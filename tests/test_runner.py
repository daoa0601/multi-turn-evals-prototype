from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from pydantic_multiturn_evals.models import (
    AcceptDecision,
    ActorBrief,
    ContinueDecision,
    Scenario,
    ScenarioLimits,
    SessionContext,
    TargetFailureEvidence,
    TargetReply,
    TurnLimitReached,
)
from pydantic_multiturn_evals.runner import (
    ActorView,
    ConversationView,
    RunnerServices,
    run_scenario,
)
from pydantic_multiturn_evals.storage import InMemoryStateStore


class ScriptedActor:
    def __init__(self, *decisions: ContinueDecision | AcceptDecision) -> None:
        self.decisions = list(decisions)
        self.views: list[ActorView] = []

    async def decide(self, view: ActorView) -> ContinueDecision | AcceptDecision:
        self.views.append(view)
        return self.decisions.pop(0)


class ScriptedTarget:
    name = "scripted"
    kind: Literal["command"] = "command"
    version = 1

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.views: list[ConversationView] = []

    @asynccontextmanager
    async def session(self, context: SessionContext) -> AsyncIterator[ScriptedTarget]:
        yield self

    async def reply(self, view: ConversationView) -> TargetReply:
        self.views.append(view)
        return TargetReply(assistant_text=self.replies.pop(0))

    def failure_evidence(self) -> tuple[TargetFailureEvidence, ...]:
        return ()


def scenario() -> Scenario:
    return Scenario(
        id="refund",
        first_prompt="I need a refund.",
        actor=ActorBrief(persona="A customer", goal="Learn the refund steps"),
        judge_rubric="The assistant gives accurate refund steps.",
    )


def test_actor_adapts_the_second_turn_to_the_first_reply() -> None:
    target = ScriptedTarget("What is your order number?", "Use the refund form.")
    actor = ScriptedActor(
        ContinueDecision(
            next_user_message="It is ORD-1042.",
            reason="The assistant needs the order number.",
        ),
        AcceptDecision(reason="The customer has a concrete next step."),
    )
    store = InMemoryStateStore()

    result = asyncio.run(
        run_scenario(
            scenario(),
            limits=ScenarioLimits(max_target_turns=3, timeout_seconds=5),
            target=target,
            actor=actor,
            services=RunnerServices(state_store=store),
        )
    )

    assert [exchange.user.content for exchange in result.transcript.exchanges] == [
        "I need a refund.",
        "It is ORD-1042.",
    ]
    assert actor.views[0].exchanges[0].assistant.content == "What is your order number?"
    assert [message.content for message in target.views[1].messages] == [
        "I need a refund.",
        "What is your order number?",
        "It is ORD-1042.",
    ]
    final_state = asyncio.run(store.load(result.run_id))
    assert final_state is not None
    assert final_state.termination == result.termination


def test_continue_at_the_hard_limit_cannot_create_a_dangling_turn() -> None:
    actor = ScriptedActor(
        ContinueDecision(next_user_message="One more thing.", reason="More evidence is useful.")
    )

    result = asyncio.run(
        run_scenario(
            scenario(),
            limits=ScenarioLimits(max_target_turns=1, timeout_seconds=5),
            target=ScriptedTarget("First response"),
            actor=actor,
        )
    )

    assert isinstance(result.termination, TurnLimitReached)
    assert len(result.transcript.exchanges) == 1
    assert result.transcript.exchanges[-1].assistant.content == "First response"
