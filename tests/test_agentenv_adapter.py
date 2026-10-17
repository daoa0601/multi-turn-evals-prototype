from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from pydantic_multiturn_evals import (
    Campaign,
    Case,
    ComponentDefinition,
    Failed,
    Message,
    TraceRecorder,
    TrialBudget,
    Variant,
    run,
)
from pydantic_multiturn_evals.adapters import AgentEnvTarget
from pydantic_multiturn_evals.adapters import agentenv as agentenv_module
from pydantic_multiturn_evals.harnesses import Accept, Continue, ConversationHarness


@dataclass
class _Result:
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class _FileInfo:
    size: int


class _FileStream:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.downloaded = 0
        self.closed = False

    async def __aenter__(self) -> _FileStream:
        return self

    async def __aexit__(self, *error: object) -> None:
        self.closed = True

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for offset in range(0, len(self.content), 64):
            chunk = self.content[offset : offset + 64]
            self.downloaded += len(chunk)
            yield chunk


class _Files:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {"/work/proof.txt": b"real state"}
        self.padding: dict[str, int] = {}
        self.reported_sizes: dict[str, int] = {}
        self.reads: list[str] = []
        self.streams: dict[str, _FileStream] = {}

    async def get_info(self, path: str, *, request_timeout: float) -> _FileInfo:
        size = len(self.values[path]) + self.padding.get(path, 0)
        return _FileInfo(self.reported_sizes.get(path, size))

    async def write(self, path: str, data: str, *, request_timeout: float) -> object:
        self.values[path] = data.encode()
        return object()

    async def read(
        self,
        path: str,
        format: Literal["stream"],
        *,
        request_timeout: float,
    ) -> _FileStream:
        self.reads.append(path)
        value = self.values[path] + b" " * self.padding.get(path, 0)
        stream = _FileStream(value)
        self.streams[path] = stream
        return stream


class _Commands:
    def __init__(self, files: _Files, order: list[str], block: bool = False) -> None:
        self.files = files
        self.order = order
        self.block = block
        self.artifacts = [{"name": "proof", "path": "/work/proof.txt"}]

    async def run(
        self,
        command: str,
        *,
        envs: dict[str, str],
        timeout: float,
    ) -> _Result:
        if self.block:
            await asyncio.Event().wait()
        if command == "turn":
            request = json.loads(self.files.values[envs["MTE_REQUEST_PATH"]])
            turn = request["id"]
            self.files.values[envs["MTE_RESPONSE_PATH"]] = json.dumps(
                {
                    "protocol": 1,
                    "id": turn,
                    "messages": [{"role": "assistant", "content": f"reply-{turn}"}],
                }
            ).encode()
            self.order.append(f"turn-{turn}")
        else:
            self.files.values[envs["MTE_VERIFY_RESPONSE_PATH"]] = json.dumps(
                {
                    "protocol": 1,
                    "output": {"verified": True},
                    "checks": [
                        {"name": "sandbox-state", "passed": True, "reason": "state changed"}
                    ],
                    "artifacts": self.artifacts,
                }
            ).encode()
            self.order.append("verify")
        return _Result()


class _Sandbox:
    sandbox_id = "sandbox-real-boundary"

    def __init__(self, *, block: bool = False) -> None:
        self.order: list[str] = []
        self.files = _Files()
        self.commands = _Commands(self.files, self.order, block)

    async def kill(self) -> bool:
        self.order.append("destroy")
        return True


class _Actor:
    component = ComponentDefinition(kind="scripted-user", version="1")

    async def next_turn(
        self,
        case: Case,
        messages: tuple[Message, ...],
        turn: int,
        trace: TraceRecorder,
    ) -> Continue | Accept:
        del case, messages, trace
        if turn < 2:
            return Continue(
                message=Message(role="user", content=f"request-{turn + 1}"),
                reason="continue",
            )
        return Accept(reason="enough evidence")


def test_agentenv_target_keeps_one_sandbox_across_turns_and_verifies_before_destroy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _Sandbox()

    async def create(_template: str, _ttl: int, _environment: dict[str, str]) -> _Sandbox:
        return sandbox

    monkeypatch.setattr(agentenv_module, "_create_sandbox", create)
    campaign = _campaign()

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    result = report.results[0]
    assert result.passed is True
    assert sandbox.order == ["turn-1", "turn-2", "verify", "destroy"]
    request = sandbox.files.values["/tmp/mte/request.json"].decode()
    assert "secret-for-user-actor" not in request
    proof = next((tmp_path / "run").glob("trials/**/agentenv-artifacts/proof"))
    assert proof.read_text(encoding="utf-8") == "real state"


def test_agentenv_target_destroys_the_sandbox_when_a_turn_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _Sandbox(block=True)

    async def create(_template: str, _ttl: int, _environment: dict[str, str]) -> _Sandbox:
        return sandbox

    monkeypatch.setattr(agentenv_module, "_create_sandbox", create)
    campaign = _campaign(budget=TrialBudget(turns=3, seconds=0.05))

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    assert isinstance(report.results[0].terminal, Failed)
    assert report.results[0].terminal.failure.phase == "timeout"
    assert sandbox.order == ["destroy"]
    state = json.loads(next((tmp_path / "run").glob("trials/**/agentenv-state.json")).read_text())
    assert state["phase"] == "destroyed"


@pytest.mark.parametrize(
    "remote_path", ["/tmp/mte/response.json", "/tmp/mte/verify-response.json", "/work/proof.txt"]
)
@pytest.mark.parametrize("size_is_current", [True, False], ids=["known-size", "growing-file"])
def test_agentenv_rejects_oversized_files_before_downloading_the_whole_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote_path: str, size_is_current: bool
) -> None:
    sandbox = _Sandbox()
    sandbox.files.padding[remote_path] = 8192
    if not size_is_current:
        sandbox.files.reported_sizes[remote_path] = 1

    async def create(_template: str, _ttl: int, _environment: dict[str, str]) -> _Sandbox:
        return sandbox

    monkeypatch.setattr(agentenv_module, "_create_sandbox", create)

    report = asyncio.run(
        run(_campaign(response_bytes=512, artifact_bytes=512), directory=tmp_path / "run")
    )

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert remote_path in terminal.failure.message
    assert "512 bytes" in terminal.failure.message
    assert sandbox.order[-1] == "destroy"
    if size_is_current:
        assert remote_path not in sandbox.files.reads, (
            "oversized file was downloaded despite its size"
        )
    else:
        assert remote_path in sandbox.files.streams, "file was buffered instead of streamed"
        stream = sandbox.files.streams[remote_path]
        assert stream.downloaded <= 512 + 64
        assert stream.closed, "oversized download left its connection open"


def test_agentenv_checks_remaining_artifact_budget_before_the_next_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = _Sandbox()
    sandbox.files.values["/work/extra.txt"] = b"more proof"
    sandbox.commands.artifacts.append({"name": "extra", "path": "/work/extra.txt"})

    async def create(_template: str, _ttl: int, _environment: dict[str, str]) -> _Sandbox:
        return sandbox

    monkeypatch.setattr(agentenv_module, "_create_sandbox", create)

    report = asyncio.run(run(_campaign(artifact_bytes=15), directory=tmp_path / "run"))

    assert isinstance(report.results[0].terminal, Failed)
    assert "/work/proof.txt" in sandbox.files.reads
    assert "/work/extra.txt" not in sandbox.files.reads, "download exceeded the remaining budget"
    assert sandbox.order[-1] == "destroy"


def _campaign(
    *,
    budget: TrialBudget | None = None,
    response_bytes: int = 4_000_000,
    artifact_bytes: int = 64_000_000,
) -> Campaign:
    target = AgentEnvTarget(
        template="adapter-contract",
        turn_command="turn",
        verifier_command="verify",
        turn_timeout_seconds=1,
        verify_timeout_seconds=1,
        response_bytes=response_bytes,
        artifact_bytes=artifact_bytes,
    )
    return Campaign(
        name="agentenv-boundary",
        revision="1",
        cases=(
            Case(
                name="conversation",
                task="Exercise the sandbox",
                data={"secret": "secret-for-user-actor"},
                required_checks=("sandbox-state",),
                budget=budget or TrialBudget(turns=3, seconds=2),
            ),
        ),
        variants=(
            Variant(
                name="only",
                harness=ConversationHarness(target=target, actor=_Actor()),
            ),
        ),
    )
