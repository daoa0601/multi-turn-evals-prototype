"""A persistent JSONL target for the conversation harness."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from pydantic_multiturn_evals.harnesses._process import (
    command_environment,
    terminate_process_group,
)
from pydantic_multiturn_evals.harnesses.conversation import TargetCompletion
from pydantic_multiturn_evals.model import (
    ComponentDefinition,
    Message,
    TraceRecorder,
    Trial,
)


@dataclass(frozen=True, slots=True)
class CommandLimits:
    response_seconds: float = 30
    shutdown_seconds: float = 2
    response_bytes: int = 1_000_000
    stderr_bytes: int = 100_000


@dataclass(frozen=True, slots=True)
class CommandTarget:
    argv: tuple[str, ...]
    cwd: Path
    limits: CommandLimits = CommandLimits()
    inherit_environment: tuple[str, ...] = ()

    @property
    def component(self) -> ComponentDefinition:
        return ComponentDefinition(
            kind="jsonl-target",
            version="1",
            config={
                "argv": list(self.argv),
                "cwd": str(self.cwd),
                "inherit_environment": list(self.inherit_environment),
                "limits": {
                    "response_seconds": self.limits.response_seconds,
                    "shutdown_seconds": self.limits.shutdown_seconds,
                    "response_bytes": self.limits.response_bytes,
                    "stderr_bytes": self.limits.stderr_bytes,
                },
            },
        )

    async def open(
        self,
        trial: Trial,
        trace: TraceRecorder,
    ) -> _CommandSession:
        del trace
        if not self.argv:
            raise ValueError("command target argv cannot be empty")
        environment = command_environment(self.inherit_environment, trial.directory)
        process = await asyncio.create_subprocess_exec(
            *self.argv,
            cwd=self.cwd,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            limit=self.limits.response_bytes + 1,
        )
        session = _CommandSession(process, self.limits)
        try:
            await session.request(
                {
                    "protocol": 1,
                    "type": "start",
                    "trial": {
                        "key": trial.key.model_dump(mode="json"),
                        "attempt": trial.attempt,
                        "seed": trial.seed,
                    },
                },
                expected_type="ready",
            )
        except BaseException:
            await session.close()
            raise
        return session


class _CommandSession:
    def __init__(
        self,
        process: asyncio.subprocess.Process,
        limits: CommandLimits,
    ) -> None:
        self._process = process
        self._limits = limits
        self._turn = 0
        self._failed = False
        self._stderr = bytearray()
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def send(self, message: Message) -> tuple[Message, ...]:
        self._turn += 1
        response = await self.request(
            {
                "protocol": 1,
                "type": "turn",
                "id": self._turn,
                "message": message.model_dump(mode="json"),
            },
            expected_type="reply",
        )
        if response.get("id") != self._turn:
            raise RuntimeError(
                f"target reply id {response.get('id')!r} does not match turn {self._turn}"
            )
        try:
            return TypeAdapter(tuple[Message, ...]).validate_python(response.get("messages"))
        except ValidationError as error:
            raise RuntimeError(f"invalid reply messages: {error}") from error

    async def finish(self, messages: tuple[Message, ...]) -> TargetCompletion:
        response = await self.request(
            {
                "protocol": 1,
                "type": "finish",
                "messages": [message.model_dump(mode="json") for message in messages],
            },
            expected_type="finished",
        )
        try:
            return TargetCompletion.model_validate(response.get("completion"))
        except ValidationError as error:
            raise RuntimeError(f"invalid target completion: {error}") from error

    async def request(self, payload: object, *, expected_type: str) -> dict[str, object]:
        if self._failed:
            raise RuntimeError("target session is unusable after an earlier I/O failure")
        try:
            async with asyncio.timeout(self._limits.response_seconds):
                return await self._exchange(payload, expected_type=expected_type)
        except TimeoutError as error:
            await self._abort()
            raise TimeoutError(
                f"target did not return {expected_type!r} within "
                f"{self._limits.response_seconds:g} seconds"
            ) from error
        except BaseException:
            await self._abort()
            raise

    async def _exchange(self, payload: object, *, expected_type: str) -> dict[str, object]:
        stdin = self._process.stdin
        stdout = self._process.stdout
        if stdin is None or stdout is None:
            raise RuntimeError("target process has no pipe")
        if self._process.returncode is not None:
            raise RuntimeError(self._exit_message())
        stdin.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())
        await stdin.drain()
        try:
            line = await stdout.readline()
        except ValueError as error:
            raise RuntimeError(
                f"target response exceeds {self._limits.response_bytes} bytes"
            ) from error
        if not line:
            raise RuntimeError(self._exit_message())
        if len(line) > self._limits.response_bytes:
            raise RuntimeError(f"target response exceeds {self._limits.response_bytes} bytes")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"target returned invalid JSON: {error.msg}") from error
        if not isinstance(response, dict):
            raise RuntimeError("target response must be a JSON object")
        if response.get("protocol") != 1 or response.get("type") != expected_type:
            raise RuntimeError(
                f"expected protocol 1 {expected_type!r}, got {response.get('type')!r}"
            )
        return response

    async def close(self) -> None:
        try:
            if not self._failed and self._process.returncode is None:
                stdin = self._process.stdin
                if stdin is not None:
                    async with asyncio.timeout(self._limits.shutdown_seconds):
                        try:
                            stdin.write(b'{"protocol":1,"type":"close"}\n')
                            await stdin.drain()
                        except (BrokenPipeError, ConnectionResetError):
                            pass
                        await self._process.wait()
        except TimeoutError:
            pass
        finally:
            terminate_process_group(self._process)
            if self._process.returncode is None:
                await self._process.wait()
            await self._stderr_task

    async def _abort(self) -> None:
        self._failed = True
        terminate_process_group(self._process)
        if self._process.returncode is None:
            await self._process.wait()

    async def _drain_stderr(self) -> None:
        stderr = self._process.stderr
        if stderr is None:
            return
        while chunk := await stderr.read(4096):
            remaining = self._limits.stderr_bytes - len(self._stderr)
            if remaining > 0:
                self._stderr.extend(chunk[:remaining])

    def _exit_message(self) -> str:
        detail = self._stderr.decode(errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        return f"target exited with code {self._process.returncode}{suffix}"
