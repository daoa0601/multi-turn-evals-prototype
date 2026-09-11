from __future__ import annotations

import asyncio
import json

import pytest

from pydantic_multiturn_evals.agentenv_target import (
    AgentEnvTarget,
    SandboxProcessResult,
)
from pydantic_multiturn_evals.models import (
    ActorAccepted,
    AgentEnvLimits,
    AgentEnvTargetSpec,
    CaseKey,
    ConversationView,
    SandboxCommandSpec,
    SessionContext,
    SessionOutcome,
    Transcript,
    UserTurn,
)


class FakeSandbox:
    environment_id = "sandbox-123"

    def __init__(self, *, oversized_reply: bool = False) -> None:
        self.files: dict[str, bytes] = {}
        self.events: list[str] = []
        self.oversized_reply = oversized_reply

    async def write(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def read(self, path: str) -> bytes:
        return self.files[path]

    async def file_size(self, path: str) -> int:
        return len(self.files[path])

    async def run(
        self,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str],
        timeout_seconds: float,
    ) -> SandboxProcessResult:
        kind = env["MULTITURN_OPERATION"]
        self.events.append(kind)
        if kind == "turn":
            request = json.loads(self.files[env["MULTITURN_REQUEST_PATH"]])
            response = {
                "protocol": 1,
                "type": "reply",
                "assistant_text": f"saw {len(request['messages'])} messages",
                "evidence": {"sandbox": "ok"},
            }
            data = json.dumps(response).encode()
            if self.oversized_reply:
                data += b" " * 4096
            self.files[env["MULTITURN_RESPONSE_PATH"]] = data
        else:
            self.files[env["MULTITURN_RESPONSE_PATH"]] = json.dumps(
                {
                    "protocol": 1,
                    "type": "verification",
                    "passed": True,
                    "reward": 0.8,
                    "reason": "Workspace is correct.",
                    "evidence": {"checks": 3},
                }
            ).encode()
        return SandboxProcessResult(exit_code=0)

    async def destroy(self) -> None:
        self.events.append("destroy")


class FakeFactory:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self.sandbox = sandbox
        self.contexts: list[SessionContext] = []

    async def create(self, spec: AgentEnvTargetSpec, context: SessionContext) -> FakeSandbox:
        self.contexts.append(context)
        return self.sandbox


def spec(*, response_limit: int = 2048) -> AgentEnvTargetSpec:
    return AgentEnvTargetSpec(
        version=1,
        name="agentenv-example",
        kind="agentenv",
        template="eval-base",
        turn=SandboxCommandSpec(argv=("python", "/opt/eval/turn.py")),
        verifier=SandboxCommandSpec(argv=("python", "/opt/eval/verify.py")),
        limits=AgentEnvLimits(response_bytes=response_limit),
    )


def context() -> SessionContext:
    return SessionContext(
        suite_name="test-suite",
        target_name="agentenv-example",
        target_kind="agentenv",
        target_version=1,
        key=CaseKey(scenario_id="help", repeat_index=2),
        run_id="run-1",
    )


def outcome() -> SessionOutcome:
    return SessionOutcome(
        transcript=Transcript(exchanges=()),
        decisions=(),
        termination=ActorAccepted(reason="Done."),
    )


def view() -> ConversationView:
    return ConversationView(
        run_id="run-1",
        scenario_id="help",
        exchanges=(),
        pending_user=UserTurn(content="Please help."),
    )


def test_agentenv_uses_one_sandbox_and_verifies_before_destroy() -> None:
    async def exercise() -> None:
        sandbox = FakeSandbox()
        factory = FakeFactory(sandbox)
        target = AgentEnvTarget(spec(), sandbox_factory=factory)

        async with target.session(context()) as session:
            reply = await session.reply(view())
            completion = await session.finish(outcome())
            assert sandbox.events == ["turn", "verify"]

        assert reply.assistant_text == "saw 1 messages"
        assert completion.environment is not None
        assert completion.environment.environment_id == "sandbox-123"
        assert completion.environment.reward == 0.8
        assert sandbox.events == ["turn", "verify", "destroy"]
        assert len(factory.contexts) == 1

    asyncio.run(exercise())


def test_agentenv_destroys_the_sandbox_when_a_turn_is_cancelled() -> None:
    class HangingSandbox(FakeSandbox):
        async def run(self, *args: object, **kwargs: object) -> SandboxProcessResult:
            await asyncio.sleep(60)
            raise AssertionError("unreachable")

    async def exercise() -> FakeSandbox:
        sandbox = HangingSandbox()
        target = AgentEnvTarget(spec(), sandbox_factory=FakeFactory(sandbox))
        with pytest.raises(TimeoutError):
            async with target.session(context()) as session:
                await asyncio.wait_for(session.reply(view()), timeout=0.01)
        return sandbox

    sandbox = asyncio.run(exercise())
    assert sandbox.events == ["destroy"]


def test_agentenv_rejects_an_oversized_response_before_parsing() -> None:
    async def exercise() -> None:
        sandbox = FakeSandbox(oversized_reply=True)
        target = AgentEnvTarget(spec(response_limit=256), sandbox_factory=FakeFactory(sandbox))
        with pytest.raises(RuntimeError, match="response exceeded its byte limit"):
            async with target.session(context()) as session:
                await session.reply(view())

    asyncio.run(exercise())


def test_agentenv_finish_is_single_use() -> None:
    async def exercise() -> None:
        target = AgentEnvTarget(spec(), sandbox_factory=FakeFactory(FakeSandbox()))
        async with target.session(context()) as session:
            await session.finish(outcome())
            with pytest.raises(RuntimeError, match="already finished"):
                await session.finish(outcome())

    asyncio.run(exercise())


def test_agentenv_records_bounded_command_failure_evidence() -> None:
    class FailingSandbox(FakeSandbox):
        async def run(self, *args: object, **kwargs: object) -> SandboxProcessResult:
            return SandboxProcessResult(exit_code=7, stderr="guest failure")

    async def exercise() -> AgentEnvTarget:
        target = AgentEnvTarget(spec(), sandbox_factory=FakeFactory(FailingSandbox()))
        with pytest.raises(RuntimeError, match="status 7"):
            async with target.session(context()) as session:
                await session.reply(view())
        return target

    target = asyncio.run(exercise())
    assert target.failure_evidence()[0].stderr == "guest failure"


def test_agentenv_sanitizes_file_metadata_failures() -> None:
    class BrokenMetadataSandbox(FakeSandbox):
        async def file_size(self, path: str) -> int:
            raise OSError("private provider detail")

    async def exercise() -> None:
        target = AgentEnvTarget(spec(), sandbox_factory=FakeFactory(BrokenMetadataSandbox()))
        with pytest.raises(RuntimeError, match="response metadata failed: OSError") as failure:
            async with target.session(context()) as session:
                await session.reply(view())
        assert "private provider detail" not in str(failure.value)

    asyncio.run(exercise())


def test_agentenv_sanitizes_request_write_failures() -> None:
    class BrokenWriteSandbox(FakeSandbox):
        async def write(self, path: str, data: bytes) -> None:
            raise OSError("private provider detail")

    async def exercise() -> None:
        target = AgentEnvTarget(spec(), sandbox_factory=FakeFactory(BrokenWriteSandbox()))
        with pytest.raises(RuntimeError, match="request write failed: OSError") as failure:
            async with target.session(context()) as session:
                await session.reply(view())
        assert "private provider detail" not in str(failure.value)

    asyncio.run(exercise())


def test_agentenv_rejects_an_invalid_reply_envelope() -> None:
    class InvalidReplySandbox(FakeSandbox):
        async def read(self, path: str) -> bytes:
            return b"{}"

    async def exercise() -> None:
        target = AgentEnvTarget(spec(), sandbox_factory=FakeFactory(InvalidReplySandbox()))
        with pytest.raises(RuntimeError, match="invalid reply"):
            async with target.session(context()) as session:
                await session.reply(view())

    asyncio.run(exercise())
