"""A bounded JSONL subprocess target with one process per scenario attempt."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from pydantic_multiturn_evals.models import (
    CommandTargetSpec,
    ConversationView,
    SessionContext,
    SessionOutcome,
    TargetCompletion,
    TargetFailureEvidence,
    TargetReply,
)
from pydantic_multiturn_evals.protocol import (
    CloseRequest,
    FinishedResponse,
    FinishRequest,
    ReadyResponse,
    ReplyResponse,
    StartRequest,
    TurnRequest,
    WireSession,
)
from pydantic_multiturn_evals.targets import TargetSession


class CommandTarget:
    kind: Literal["command"] = "command"

    def __init__(self, spec: CommandTargetSpec) -> None:
        _validate_command(spec)
        self.name = spec.name
        self.version = spec.version
        self.executable = executable_identity(spec.argv)
        self._spec = spec
        self._failures: list[TargetFailureEvidence] = []

    @asynccontextmanager
    async def session(self, context: SessionContext) -> AsyncIterator[TargetSession]:
        session = _CommandSession(self._spec, context)
        try:
            await session.start()
            yield session
        except BaseException as error:
            await session.close(check_exit=False)
            self._failures.append(session.failure_evidence(error))
            raise
        else:
            try:
                await session.close(check_exit=True)
            except BaseException as error:
                self._failures.append(session.failure_evidence(error))
                raise

    def failure_evidence(self) -> tuple[TargetFailureEvidence, ...]:
        return tuple(self._failures)


class _CommandSession:
    def __init__(self, spec: CommandTargetSpec, context: SessionContext) -> None:
        self._spec = spec
        self._context = context
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr = bytearray()
        self._stderr_truncated = False
        self._turn = 0
        self._closed = False
        self._finished = False

    async def start(self) -> None:
        cwd = self._spec.cwd
        if not cwd.is_dir():
            raise ValueError(f"command target cwd is not a directory: {cwd}")
        environment = {
            name: value
            for name in self._spec.inherit_env
            if (value := os.environ.get(name)) is not None
        }
        self._process = await asyncio.create_subprocess_exec(
            *self._spec.argv,
            cwd=cwd,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=self._spec.limits.stdout_bytes_per_message + 1,
            start_new_session=True,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        await self._write(
            StartRequest(
                session=WireSession(
                    comparison_id=self._context.comparison_id,
                    arm=self._context.arm,
                    suite=self._context.suite_name,
                    target=self._context.target_name,
                    scenario_id=self._context.key.scenario_id,
                    repeat_index=self._context.key.repeat_index,
                    run_id=self._context.run_id,
                )
            )
        )
        payload = await self._read_payload(self._spec.limits.startup_seconds)
        try:
            ReadyResponse.model_validate(payload)
        except ValidationError as error:
            raise RuntimeError(f"command target sent an invalid ready message: {error}") from error

    async def reply(self, view: ConversationView) -> TargetReply:
        self._turn += 1
        await self._write(TurnRequest(id=self._turn, messages=view.messages))
        payload = await self._read_payload(self._spec.limits.turn_seconds)
        try:
            response = ReplyResponse.model_validate(payload)
        except ValidationError as error:
            raise RuntimeError(f"command target sent an invalid reply: {error}") from error
        if response.id != self._turn:
            raise RuntimeError(
                f"command target replied to turn {response.id}, expected {self._turn}"
            )
        return TargetReply(
            assistant_text=response.assistant_text,
            session_id=response.session_id,
            evidence=response.evidence,
        )

    async def finish(self, outcome: SessionOutcome) -> TargetCompletion:
        if self._finished:
            raise RuntimeError("command target session was already finished")
        self._finished = True
        await self._write(FinishRequest(outcome=outcome))
        payload = await self._read_payload(self._spec.limits.turn_seconds)
        try:
            response = FinishedResponse.model_validate(payload)
        except ValidationError as error:
            raise RuntimeError(
                f"command target sent an invalid finished message: {error}"
            ) from error
        return response.completion

    async def close(self, *, check_exit: bool) -> None:
        if self._closed:
            return
        self._closed = True
        cleanup = asyncio.create_task(self._close_process(check_exit=check_exit))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    def failure_evidence(self, error: BaseException) -> TargetFailureEvidence:
        process = self._process
        return TargetFailureEvidence(
            run_id=self._context.run_id,
            scenario_id=self._context.key.scenario_id,
            repeat_index=self._context.key.repeat_index,
            error_type=type(error).__name__,
            stderr=self._stderr.decode(errors="replace"),
            stderr_truncated=self._stderr_truncated,
            exit_code=process.returncode if process is not None else None,
        )

    async def _close_process(self, *, check_exit: bool) -> None:
        process = self._process
        if process is None:
            return
        if process.returncode is None:
            with suppress(BrokenPipeError, ConnectionResetError):
                await self._write(CloseRequest())
            if process.stdin is not None:
                process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=self._spec.limits.shutdown_seconds)
            except TimeoutError:
                self._signal_process_group(signal.SIGTERM)
                try:
                    await asyncio.wait_for(
                        process.wait(), timeout=self._spec.limits.shutdown_seconds
                    )
                except TimeoutError:
                    self._signal_process_group(signal.SIGKILL)
                    await process.wait()
        else:
            await process.wait()
        if self._stderr_task is not None:
            await self._stderr_task
        self._signal_process_group(signal.SIGTERM)
        await asyncio.sleep(0)
        self._signal_process_group(signal.SIGKILL)
        if check_exit and process.returncode != 0:
            raise self._exit_error(process.returncode)

    async def _write(self, payload: BaseModel) -> None:
        process = self._require_process()
        if process.stdin is None:
            raise RuntimeError("command target stdin is unavailable")
        if process.returncode is not None:
            raise self._exit_error(process.returncode)
        process.stdin.write((payload.model_dump_json() + "\n").encode())
        try:
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as error:
            await process.wait()
            raise self._exit_error(process.returncode) from error

    async def _read_payload(self, timeout: float) -> object:
        process = self._require_process()
        if process.stdout is None:
            raise RuntimeError("command target stdout is unavailable")
        try:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=timeout)
        except ValueError as error:
            raise RuntimeError("command target response exceeded its byte limit") from error
        except TimeoutError as error:
            raise RuntimeError(
                f"command target did not reply within {timeout:g} seconds"
            ) from error
        if len(line) > self._spec.limits.stdout_bytes_per_message:
            raise RuntimeError("command target response exceeded its byte limit")
        if not line:
            await process.wait()
            raise self._exit_error(process.returncode)
        try:
            return json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("command target returned invalid JSON") from error

    async def _drain_stderr(self) -> None:
        process = self._require_process()
        if process.stderr is None:
            return
        limit = self._spec.limits.stderr_bytes_per_session
        while chunk := await process.stderr.read(8192):
            remaining = limit - len(self._stderr)
            if remaining > 0:
                self._stderr.extend(chunk[:remaining])
            if len(chunk) > remaining:
                self._stderr_truncated = True

    def _exit_error(self, returncode: int | None) -> RuntimeError:
        suffix = f"; captured {len(self._stderr)} stderr byte(s)" if self._stderr else ""
        if self._stderr_truncated:
            suffix += " (truncated)"
        return RuntimeError(f"command target exited with status {returncode}{suffix}")

    def _signal_process_group(self, sig: signal.Signals) -> None:
        process = self._require_process()
        with suppress(ProcessLookupError):
            os.killpg(process.pid, sig)

    def _require_process(self) -> asyncio.subprocess.Process:
        if self._process is None:
            raise RuntimeError("command target process has not started")
        return self._process


def executable_identity(argv: tuple[str, ...]) -> str:
    """Return the only command identity allowed in artifacts or traces."""

    return Path(argv[0]).name


def _validate_command(spec: CommandTargetSpec) -> None:
    if not spec.cwd.is_dir():
        raise ValueError(f"command target cwd is not a directory: {spec.cwd}")
    missing = [name for name in spec.inherit_env if not os.environ.get(name)]
    if missing:
        raise ValueError(f"command target requires environment variable(s): {', '.join(missing)}")
    executable = Path(spec.argv[0])
    if executable.is_absolute() or executable.parent != Path("."):
        resolved = executable if executable.is_absolute() else spec.cwd / executable
        available = resolved.is_file() and os.access(resolved, os.X_OK)
    else:
        path = os.environ.get("PATH") if "PATH" in spec.inherit_env else None
        available = shutil.which(spec.argv[0], path=path) is not None
    if not available:
        raise ValueError(f"command target executable is not available: {spec.argv[0]}")
