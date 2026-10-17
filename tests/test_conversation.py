from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from pydantic_multiturn_evals import (
    Campaign,
    Case,
    ComponentDefinition,
    Failed,
    Message,
    NativeCheck,
    ResourceClaim,
    TraceRecorder,
    Trial,
    TrialBudget,
    TrialKey,
    Variant,
    run,
)
from pydantic_multiturn_evals.evaluation import HarnessError
from pydantic_multiturn_evals.harnesses import (
    Accept,
    Continue,
    ConversationHarness,
    TargetCompletion,
    TargetSession,
)
from pydantic_multiturn_evals.tracing import LocalTrace


@dataclass(slots=True)
class AdaptiveActor:
    component = ComponentDefinition(kind="adaptive-actor", version="1")
    observations: list[tuple[Message, ...]] = field(default_factory=list)

    async def next_turn(
        self,
        case: Case,
        messages: tuple[Message, ...],
        turn: int,
        trace: TraceRecorder,
    ) -> Continue | Accept:
        del trace
        self.observations.append(messages)
        if turn == 0:
            return Continue(
                message=Message(role="user", content=case.task),
                reason="start",
            )
        if messages[-1].content == "need identifier":
            return Continue(
                message=Message(role="user", content="identifier 42"),
                reason="answer the question",
            )
        return Accept(reason="resolved")


class RecordingSession:
    def __init__(self) -> None:
        self.received: list[str] = []
        self.events: list[str] = []

    async def send(self, message: Message) -> tuple[Message, ...]:
        self.received.append(message.content)
        content = "need identifier" if len(self.received) == 1 else "resolved"
        return (Message(role="assistant", content=content),)

    async def finish(self, messages: tuple[Message, ...]) -> TargetCompletion:
        del messages
        self.events.append("finish")
        return TargetCompletion(
            checks=(NativeCheck(name="state", passed=True, reason="real state is correct"),)
        )

    async def close(self) -> None:
        self.events.append("close")


class BrokenCloseSession(RecordingSession):
    async def close(self) -> None:
        await super().close()
        raise RuntimeError("cleanup failed")


@dataclass(slots=True)
class RecordingTarget:
    component = ComponentDefinition(kind="recording-target", version="1")
    session: RecordingSession = field(default_factory=RecordingSession)

    async def open(
        self,
        trial: Trial,
        trace: TraceRecorder,
    ) -> TargetSession:
        del trial, trace
        return self.session


def test_conversation_adapts_and_verifies_before_cleanup(tmp_path: Path) -> None:
    actor = AdaptiveActor()
    target = RecordingTarget()
    trace = LocalTrace(_key(), 1, tmp_path / "events.jsonl")

    evidence = asyncio.run(
        ConversationHarness(target=target, actor=actor).run(
            Case(name="case", task="Help me", required_checks=("state",)),
            _trial(tmp_path),
            trace,
        )
    )
    trace.close()

    assert target.session.received == ["Help me", "identifier 42"]
    assert actor.observations[1][-1].content == "need identifier"
    assert target.session.events == ["finish", "close"]
    assert evidence.checks[0].passed is True
    assert [event.fact.kind for event in trace.events] == [
        "message",
        "message",
        "message",
        "message",
    ]


def test_cleanup_failure_keeps_evidence_produced_while_target_was_open(tmp_path: Path) -> None:
    target = RecordingTarget(session=BrokenCloseSession())
    trace = LocalTrace(_key(), 1, tmp_path / "events.jsonl")

    with pytest.raises(HarnessError, match="cleanup failed") as caught:
        asyncio.run(
            ConversationHarness(target=target, actor=AdaptiveActor()).run(
                Case(name="case", task="Help me", required_checks=("state",)),
                _trial(tmp_path),
                trace,
            )
        )
    trace.close()

    assert caught.value.phase == "cleanup"
    assert caught.value.evidence is not None
    assert caught.value.evidence.checks[0].name == "state"


@pytest.mark.parametrize("blocked_phase", ["actor", "send", "finish"])
def test_deadline_propagates_when_target_cleanup_hangs(tmp_path: Path, blocked_phase: str) -> None:
    async def exercise() -> None:
        release = asyncio.Event()

        class HangingSession(RecordingSession):
            async def send(self, message: Message) -> tuple[Message, ...]:
                if blocked_phase == "send":
                    await asyncio.Event().wait()
                return await super().send(message)

            async def finish(self, messages: tuple[Message, ...]) -> TargetCompletion:
                if blocked_phase == "finish":
                    await asyncio.Event().wait()
                return await super().finish(messages)

            async def close(self) -> None:
                self.events.append("close")
                await release.wait()

        class HangingActor(AdaptiveActor):
            async def next_turn(
                self,
                case: Case,
                messages: tuple[Message, ...],
                turn: int,
                trace: TraceRecorder,
            ) -> Continue | Accept:
                if blocked_phase == "actor":
                    await asyncio.Event().wait()
                return await super().next_turn(case, messages, turn, trace)

        target = RecordingTarget(session=HangingSession())
        harness = ConversationHarness(
            target=target, actor=HangingActor(), cleanup_timeout_seconds=0.01
        )
        campaign = Campaign(
            name="cleanup-deadline",
            revision="1",
            cases=(
                Case(
                    name="case",
                    task="Help me",
                    required_checks=("state",),
                    budget=TrialBudget(seconds=0.01),
                ),
            ),
            variants=(
                Variant(name="only", harness=harness, resources=(ResourceClaim(name="target"),)),
            ),
            samples=2,
        )
        task = asyncio.create_task(
            run(campaign, directory=tmp_path / "run", workers=2, capacities={"target": 1})
        )
        done, _ = await asyncio.wait({task}, timeout=1)
        release.set()
        report = await task

        assert done, "hung target cleanup prevented the case deadline from propagating"
        assert len(report.results) == 2, "cleanup held the resource permit for the next trial"
        for result in report.results:
            assert isinstance(result.terminal, Failed)
            assert result.terminal.failure.phase == "timeout"
        assert target.session.events[-1] == "close"
        resumed = await run(campaign, directory=tmp_path / "run", capacities={"target": 1})
        assert resumed.executed_trials == 0, "completed campaign did not release its lock"

    asyncio.run(exercise())


def _key() -> TrialKey:
    return TrialKey(campaign="test", revision="1", case="case", variant="variant", sample=1)


def _trial(path: Path) -> Trial:
    return Trial(key=_key(), attempt=1, directory=path, seed=1)
