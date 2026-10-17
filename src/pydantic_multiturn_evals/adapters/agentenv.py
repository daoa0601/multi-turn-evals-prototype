"""AgentENV target using the E2B-compatible asynchronous SDK."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import JsonValue

from pydantic_multiturn_evals.harnesses.conversation import TargetCompletion
from pydantic_multiturn_evals.model import (
    Artifact,
    ComponentDefinition,
    LogFact,
    Message,
    Name,
    NativeCheck,
    ResourceFact,
    StrictModel,
    Text,
    TraceRecorder,
    Trial,
)


class AgentEnvArtifact(StrictModel):
    name: Name
    path: Text
    media_type: Text | None = None


class _TurnReply(StrictModel):
    protocol: int
    id: int
    messages: tuple[Message, ...]


class _VerifierReply(StrictModel):
    protocol: int
    output: JsonValue = None
    checks: tuple[NativeCheck, ...]
    artifacts: tuple[AgentEnvArtifact, ...] = ()


class _CommandResult(Protocol):
    exit_code: int
    stdout: str
    stderr: str


class _Commands(Protocol):
    async def run(
        self,
        command: str,
        *,
        envs: dict[str, str],
        timeout: float,
    ) -> _CommandResult: ...


class _Files(Protocol):
    async def write(self, path: str, data: str, *, request_timeout: float) -> object: ...

    async def read(
        self,
        path: str,
        format: str = "text",
        *,
        request_timeout: float,
    ) -> str | bytearray: ...


class _Sandbox(Protocol):
    sandbox_id: str
    commands: _Commands
    files: _Files

    async def kill(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class AgentEnvTarget:
    """Open one sandbox per conversation and verify before destroying it."""

    template: str
    turn_command: str
    verifier_command: str
    sdk_version: str = "2.46.4"
    sandbox_ttl_seconds: int = 900
    create_timeout_seconds: float = 60
    turn_timeout_seconds: float = 120
    verify_timeout_seconds: float = 180
    destroy_timeout_seconds: float = 30
    response_bytes: int = 4_000_000
    artifact_bytes: int = 64_000_000
    sandbox_environment: tuple[str, ...] = ()
    request_path: str = "/tmp/mte/request.json"
    response_path: str = "/tmp/mte/response.json"
    verify_request_path: str = "/tmp/mte/verify-request.json"
    verify_response_path: str = "/tmp/mte/verify-response.json"

    def __post_init__(self) -> None:
        values = (
            self.sandbox_ttl_seconds,
            self.create_timeout_seconds,
            self.turn_timeout_seconds,
            self.verify_timeout_seconds,
            self.destroy_timeout_seconds,
            self.response_bytes,
            self.artifact_bytes,
        )
        if any(value <= 0 for value in values):
            raise ValueError("AgentENV time and byte limits must be positive")
        if len(self.sandbox_environment) != len(set(self.sandbox_environment)):
            raise ValueError("AgentENV sandbox environment names must be unique")

    @property
    def component(self) -> ComponentDefinition:
        return ComponentDefinition(
            kind="agentenv",
            version="1",
            config={
                "template": self.template,
                "turn_command": self.turn_command,
                "verifier_command": self.verifier_command,
                "sdk_version": self.sdk_version,
                "sandbox_ttl_seconds": self.sandbox_ttl_seconds,
                "create_timeout_seconds": self.create_timeout_seconds,
                "turn_timeout_seconds": self.turn_timeout_seconds,
                "verify_timeout_seconds": self.verify_timeout_seconds,
                "destroy_timeout_seconds": self.destroy_timeout_seconds,
                "response_bytes": self.response_bytes,
                "artifact_bytes": self.artifact_bytes,
                "sandbox_environment": list(self.sandbox_environment),
                "request_path": self.request_path,
                "response_path": self.response_path,
                "verify_request_path": self.verify_request_path,
                "verify_response_path": self.verify_response_path,
            },
        )

    async def open(self, trial: Trial, trace: TraceRecorder) -> _AgentEnvSession:
        environment = _environment(self.sandbox_environment)
        async with asyncio.timeout(self.create_timeout_seconds):
            sandbox = await _create_sandbox(
                self.template,
                self.sandbox_ttl_seconds,
                environment,
            )
        state_path = trial.directory / "agentenv-state.json"
        _write_json(
            state_path,
            {
                "protocol": 1,
                "phase": "created",
                "sandbox_id": sandbox.sandbox_id,
            },
        )
        trace.record(ResourceFact(name="agentenv-sandbox", value=1, unit="sandbox"))
        return _AgentEnvSession(self, sandbox, trial, trace, state_path)


@dataclass(slots=True)
class _AgentEnvSession:
    target: AgentEnvTarget
    sandbox: _Sandbox
    trial: Trial
    trace: TraceRecorder
    state_path: Path
    history: list[Message] = field(default_factory=list)
    turn: int = 0
    closed: bool = False

    async def send(self, message: Message) -> tuple[Message, ...]:
        self.turn += 1
        self.history.append(message)
        request = {
            "protocol": 1,
            "id": self.turn,
            "messages": [value.model_dump(mode="json") for value in self.history],
            "trial": {
                "key": self.trial.key.model_dump(mode="json"),
                "attempt": self.trial.attempt,
                "seed": self.trial.seed,
            },
        }
        await self.sandbox.files.write(
            self.target.request_path,
            json.dumps(request),
            request_timeout=self.target.turn_timeout_seconds,
        )
        await self._command(
            self.target.turn_command,
            self.target.turn_timeout_seconds,
            {
                "MTE_REQUEST_PATH": self.target.request_path,
                "MTE_RESPONSE_PATH": self.target.response_path,
                "MTE_TURN_ID": str(self.turn),
            },
        )
        reply = _TurnReply.model_validate_json(
            await self._read_text(
                self.target.response_path,
                self.target.response_bytes,
                self.target.turn_timeout_seconds,
            )
        )
        if reply.protocol != 1 or reply.id != self.turn:
            raise RuntimeError(f"AgentENV returned a stale response for turn {self.turn}")
        if not reply.messages:
            raise RuntimeError("AgentENV returned no messages")
        self.history.extend(reply.messages)
        return reply.messages

    async def finish(self, messages: tuple[Message, ...]) -> TargetCompletion:
        request = {
            "protocol": 1,
            "messages": [message.model_dump(mode="json") for message in messages],
            "trial": {
                "key": self.trial.key.model_dump(mode="json"),
                "attempt": self.trial.attempt,
                "seed": self.trial.seed,
            },
        }
        await self.sandbox.files.write(
            self.target.verify_request_path,
            json.dumps(request),
            request_timeout=self.target.verify_timeout_seconds,
        )
        await self._command(
            self.target.verifier_command,
            self.target.verify_timeout_seconds,
            {
                "MTE_VERIFY_REQUEST_PATH": self.target.verify_request_path,
                "MTE_VERIFY_RESPONSE_PATH": self.target.verify_response_path,
            },
        )
        reply = _VerifierReply.model_validate_json(
            await self._read_text(
                self.target.verify_response_path,
                self.target.response_bytes,
                self.target.verify_timeout_seconds,
            )
        )
        if reply.protocol != 1:
            raise RuntimeError("AgentENV verifier must return protocol 1")
        artifacts = await self._download(reply.artifacts)
        _write_json(
            self.state_path,
            {
                "protocol": 1,
                "phase": "verified",
                "sandbox_id": self.sandbox.sandbox_id,
            },
        )
        return TargetCompletion(output=reply.output, checks=reply.checks, artifacts=artifacts)

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            async with asyncio.timeout(self.target.destroy_timeout_seconds):
                destroyed = await self.sandbox.kill()
            if not destroyed:
                sandbox_id = self.sandbox.sandbox_id
                raise RuntimeError(f"AgentENV did not confirm destruction of {sandbox_id}")
            _write_json(
                self.state_path,
                {
                    "protocol": 1,
                    "phase": "destroyed",
                    "sandbox_id": self.sandbox.sandbox_id,
                },
            )
        except BaseException:
            _write_json(
                self.state_path,
                {
                    "protocol": 1,
                    "phase": "destruction-unconfirmed",
                    "sandbox_id": self.sandbox.sandbox_id,
                },
            )
            raise

    async def _command(
        self,
        command: str,
        timeout_seconds: float,
        environment: dict[str, str],
    ) -> None:
        async with asyncio.timeout(timeout_seconds):
            result = await self.sandbox.commands.run(
                command,
                envs=environment,
                timeout=timeout_seconds,
            )
        stdout = result.stdout.encode()
        stderr = result.stderr.encode()
        if len(stdout) + len(stderr) > self.target.response_bytes:
            raise RuntimeError("AgentENV command output exceeded its byte limit")
        if result.exit_code != 0:
            detail = result.stderr.strip()[:1000]
            raise RuntimeError(
                f"AgentENV command exited with code {result.exit_code}"
                + (f": {detail}" if detail else "")
            )
        if stdout:
            self.trace.record(LogFact(level="debug", message=result.stdout[:1000]))

    async def _read_text(self, path: str, limit: int, timeout_seconds: float) -> str:
        async with asyncio.timeout(timeout_seconds):
            value = await self.sandbox.files.read(
                path,
                "bytes",
                request_timeout=timeout_seconds,
            )
        raw = bytes(value) if isinstance(value, bytearray) else value.encode()
        if len(raw) > limit:
            raise RuntimeError(f"AgentENV file {path!r} exceeds {limit} bytes")
        return raw.decode()

    async def _download(
        self,
        requested: tuple[AgentEnvArtifact, ...],
    ) -> tuple[Artifact, ...]:
        root = self.trial.directory / "agentenv-artifacts"
        root.mkdir()
        total = 0
        artifacts: list[Artifact] = []
        for value in requested:
            async with asyncio.timeout(self.target.verify_timeout_seconds):
                content = await self.sandbox.files.read(
                    value.path,
                    "bytes",
                    request_timeout=self.target.verify_timeout_seconds,
                )
            raw = bytes(content) if isinstance(content, bytearray) else content.encode()
            total += len(raw)
            if total > self.target.artifact_bytes:
                raise RuntimeError(
                    f"AgentENV artifacts exceed {self.target.artifact_bytes} bytes"
                )
            destination = root / value.name
            destination.write_bytes(raw)
            artifacts.append(
                Artifact(
                    name=value.name,
                    path=str(destination.relative_to(self.trial.directory)),
                    media_type=value.media_type,
                )
            )
        return tuple(artifacts)


async def _create_sandbox(
    template: str,
    ttl_seconds: int,
    environment: dict[str, str],
) -> _Sandbox:
    try:
        from e2b import AsyncSandbox
    except ImportError as error:
        raise RuntimeError(
            "AgentENV support requires the agentenv extra: uv sync --extra agentenv"
        ) from error
    sandbox = await AsyncSandbox.create(
        template,
        timeout=ttl_seconds,
        envs=environment,
    )
    return cast(_Sandbox, cast(Any, sandbox))


def _environment(names: tuple[str, ...]) -> dict[str, str]:
    missing = [name for name in names if name not in os.environ]
    if missing:
        raise RuntimeError(f"environment variable is not set: {', '.join(missing)}")
    return {name: os.environ[name] for name in names}


def _write_json(path: Path, value: JsonValue) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
