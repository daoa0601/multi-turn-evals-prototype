"""AgentENV target backed by the E2B-compatible Python SDK."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, JsonValue, ValidationError

from pydantic_multiturn_evals.models import (
    AgentEnvTargetSpec,
    ConversationView,
    EnvironmentEvidence,
    SessionContext,
    SessionOutcome,
    StrictModel,
    TargetCompletion,
    TargetFailureEvidence,
    TargetReply,
    Text,
)
from pydantic_multiturn_evals.targets import TargetSession


@dataclass(frozen=True, slots=True)
class SandboxProcessResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


class CaseSandbox(Protocol):
    @property
    def environment_id(self) -> str: ...

    async def write(self, path: str, data: bytes) -> None: ...

    async def read(self, path: str) -> bytes: ...

    async def file_size(self, path: str) -> int: ...

    async def run(
        self,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str],
        timeout_seconds: float,
    ) -> SandboxProcessResult: ...

    async def destroy(self) -> None: ...


class AgentEnvSandboxFactory(Protocol):
    async def create(self, spec: AgentEnvTargetSpec, context: SessionContext) -> CaseSandbox: ...


class _SandboxReply(StrictModel):
    protocol: Literal[1]
    type: Literal["reply"]
    assistant_text: Text
    session_id: str | None = None
    evidence: dict[str, JsonValue] = Field(default_factory=dict)


class _Verification(StrictModel):
    protocol: Literal[1]
    type: Literal["verification"]
    passed: bool
    reward: float | None = Field(default=None, ge=0, le=1)
    reason: Text
    evidence: dict[str, JsonValue] = Field(default_factory=dict)


class AgentEnvTarget:
    kind: Literal["agentenv"] = "agentenv"

    def __init__(
        self,
        spec: AgentEnvTargetSpec,
        *,
        sandbox_factory: AgentEnvSandboxFactory | None = None,
    ) -> None:
        self.name = spec.name
        self.version = spec.version
        self.executable = Path(spec.turn.argv[0]).name
        self._spec = spec
        self._sandbox_factory = sandbox_factory or E2BSandboxFactory()
        self._failures: list[TargetFailureEvidence] = []

    @asynccontextmanager
    async def session(self, context: SessionContext) -> AsyncIterator[TargetSession]:
        sandbox = await asyncio.wait_for(
            self._sandbox_factory.create(self._spec, context),
            timeout=self._spec.limits.create_seconds,
        )
        session = _AgentEnvSession(self._spec, context, sandbox)
        body_error: BaseException | None = None
        try:
            yield session
        except BaseException as error:
            body_error = error
            self._failures.append(session.failure_evidence(error))
            raise
        finally:
            cleanup = asyncio.create_task(
                asyncio.wait_for(
                    _destroy_sandbox(sandbox), timeout=self._spec.limits.destroy_seconds
                )
            )
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                try:
                    await cleanup
                except BaseException as error:
                    self._failures.append(session.failure_evidence(error))
                raise
            except BaseException as error:
                self._failures.append(session.failure_evidence(error))
                if body_error is None:
                    raise

    def failure_evidence(self) -> tuple[TargetFailureEvidence, ...]:
        return tuple(self._failures)


class _AgentEnvSession:
    def __init__(
        self,
        spec: AgentEnvTargetSpec,
        context: SessionContext,
        sandbox: CaseSandbox,
    ) -> None:
        self._spec = spec
        self._context = context
        self._sandbox = sandbox
        self._turn_index = 0
        self._finished = False
        self._stderr = ""
        self._stderr_truncated = False

    async def reply(self, view: ConversationView) -> TargetReply:
        self._turn_index += 1
        request_path = self._path(f"turn-{self._turn_index}-request.json")
        response_path = self._path(f"turn-{self._turn_index}-response.json")
        request = {
            "protocol": 1,
            "type": "turn",
            "id": self._turn_index,
            "run_id": view.run_id,
            "scenario_id": view.scenario_id,
            "messages": [message.model_dump(mode="json") for message in view.messages],
        }
        await self._write(request_path, _json_bytes(request))
        await self._run_command(
            self._spec.turn.argv,
            cwd=self._spec.turn.cwd,
            operation="turn",
            request_path=request_path,
            response_path=response_path,
            timeout_seconds=self._spec.limits.turn_seconds,
        )
        payload = await self._read_response(response_path)
        try:
            reply = _SandboxReply.model_validate(payload)
        except ValidationError as error:
            raise RuntimeError(f"AgentENV target sent an invalid reply: {error}") from error
        return TargetReply(
            assistant_text=reply.assistant_text,
            session_id=reply.session_id or self._sandbox.environment_id,
            evidence=reply.evidence,
        )

    async def finish(self, outcome: SessionOutcome) -> TargetCompletion:
        if self._finished:
            raise RuntimeError("AgentENV target session was already finished")
        self._finished = True
        request_path = self._path("outcome.json")
        response_path = self._path("verification.json")
        await self._write(request_path, outcome.model_dump_json().encode())
        await self._run_command(
            self._spec.verifier.argv,
            cwd=self._spec.verifier.cwd,
            operation="verify",
            request_path=request_path,
            response_path=response_path,
            timeout_seconds=self._spec.limits.verifier_seconds,
        )
        payload = await self._read_response(response_path)
        try:
            verification = _Verification.model_validate(payload)
        except ValidationError as error:
            raise RuntimeError(f"AgentENV verifier sent an invalid result: {error}") from error
        return TargetCompletion(
            environment=EnvironmentEvidence(
                provider="agentenv",
                environment_id=self._sandbox.environment_id,
                verifier=Path(self._spec.verifier.argv[0]).name,
                passed=verification.passed,
                reward=verification.reward,
                reason=verification.reason,
                details=verification.evidence,
            )
        )

    def failure_evidence(self, error: BaseException) -> TargetFailureEvidence:
        return TargetFailureEvidence(
            run_id=self._context.run_id,
            scenario_id=self._context.key.scenario_id,
            repeat_index=self._context.key.repeat_index,
            error_type=type(error).__name__,
            stderr=self._stderr,
            stderr_truncated=self._stderr_truncated,
        )

    async def _run_command(
        self,
        argv: tuple[str, ...],
        *,
        cwd: str | None,
        operation: Literal["turn", "verify"],
        request_path: str,
        response_path: str,
        timeout_seconds: float,
    ) -> None:
        try:
            result = await self._sandbox.run(
                shlex.join(argv),
                cwd=cwd,
                env={
                    "MULTITURN_OPERATION": operation,
                    "MULTITURN_REQUEST_PATH": request_path,
                    "MULTITURN_RESPONSE_PATH": response_path,
                },
                timeout_seconds=timeout_seconds,
            )
        except Exception as error:
            raise RuntimeError(
                f"AgentENV {operation} command failed: {type(error).__name__}"
            ) from error
        stderr_bytes = result.stderr.encode(errors="replace")
        limit = self._spec.limits.response_bytes
        self._stderr = stderr_bytes[:limit].decode(errors="replace")
        self._stderr_truncated = len(stderr_bytes) > limit
        if result.exit_code != 0:
            raise RuntimeError(
                f"AgentENV {operation} command exited with status {result.exit_code}; "
                f"captured {len(self._stderr.encode())} stderr byte(s)"
            )

    async def _read_response(self, path: str) -> object:
        try:
            size = await self._sandbox.file_size(path)
        except Exception as error:
            raise RuntimeError(
                f"AgentENV response metadata failed: {type(error).__name__}"
            ) from error
        if size > self._spec.limits.response_bytes:
            raise RuntimeError("AgentENV response exceeded its byte limit")
        try:
            data = await self._sandbox.read(path)
        except Exception as error:
            raise RuntimeError(f"AgentENV response read failed: {type(error).__name__}") from error
        if len(data) > self._spec.limits.response_bytes:
            raise RuntimeError("AgentENV response exceeded its byte limit")
        try:
            return json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("AgentENV target returned invalid JSON") from error

    def _path(self, filename: str) -> str:
        return f"/tmp/pydantic-multiturn-evals-{self._context.run_id}-{filename}"

    async def _write(self, path: str, data: bytes) -> None:
        try:
            await self._sandbox.write(path, data)
        except Exception as error:
            raise RuntimeError(f"AgentENV request write failed: {type(error).__name__}") from error


class E2BSandboxFactory:  # pragma: no cover - exercised by the opt-in AgentENV smoke
    """Create AgentENV sandboxes through its E2B-compatible endpoint."""

    async def create(self, spec: AgentEnvTargetSpec, context: SessionContext) -> CaseSandbox:
        credentials = spec.credentials
        api_url = _required_environment(credentials.api_url_env)
        sandbox_url = _required_environment(credentials.sandbox_url_env)
        api_key = _required_environment(credentials.api_key_env)
        guest_env = {name: _required_environment(name) for name in spec.guest_env}
        try:
            from e2b import Sandbox  # pyright: ignore[reportMissingImports]
        except ImportError as error:
            raise RuntimeError("AgentENV targets require: uv sync --extra agentenv") from error

        try:
            sandbox = await asyncio.to_thread(
                Sandbox.create,
                template=spec.template,
                timeout=spec.limits.sandbox_ttl_seconds,
                metadata={
                    "suite": context.suite_name,
                    "target": context.target_name,
                    "scenario": context.key.scenario_id,
                    "repeat": str(context.key.repeat_index),
                    "run": context.run_id,
                },
                envs=guest_env,
                secure=True,
                api_url=api_url,
                sandbox_url=sandbox_url,
                api_key=api_key,
                request_timeout=spec.limits.create_seconds,
            )
        except Exception as error:
            raise RuntimeError(
                f"AgentENV sandbox creation failed: {type(error).__name__}"
            ) from error
        return _E2BSandbox(sandbox)


class _E2BSandbox:  # pragma: no cover - exercised by the opt-in AgentENV smoke
    def __init__(self, sandbox: Any) -> None:
        self._sandbox = sandbox

    @property
    def environment_id(self) -> str:
        return str(self._sandbox.sandbox_id)

    async def write(self, path: str, data: bytes) -> None:
        files = self._sandbox.files
        await asyncio.to_thread(files.write, path, data)

    async def read(self, path: str) -> bytes:
        files = self._sandbox.files
        value = await asyncio.to_thread(files.read, path, format="bytes")
        return bytes(value)

    async def file_size(self, path: str) -> int:
        files = self._sandbox.files
        info = await asyncio.to_thread(files.get_info, path)
        return int(info.size)

    async def run(
        self,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str],
        timeout_seconds: float,
    ) -> SandboxProcessResult:
        commands = self._sandbox.commands
        result = await asyncio.to_thread(
            commands.run,
            command,
            cwd=cwd,
            envs=env,
            timeout=timeout_seconds,
            request_timeout=timeout_seconds + 5,
        )
        return SandboxProcessResult(
            exit_code=int(result.exit_code),
            stdout=str(result.stdout),
            stderr=str(result.stderr),
        )

    async def destroy(self) -> None:
        await asyncio.to_thread(self._sandbox.kill)


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"AgentENV requires environment variable {name}")
    return value


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


async def _destroy_sandbox(sandbox: CaseSandbox) -> None:
    try:
        await sandbox.destroy()
    except Exception as error:
        raise RuntimeError(
            f"AgentENV sandbox destruction failed: {type(error).__name__}"
        ) from error
