"""The bounded adaptive conversation loop."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol, TypeVar
from uuid import uuid4

from pydantic_multiturn_evals.models import (
    AcceptDecision,
    ActorAccepted,
    ActorBrief,
    ActorDecision,
    ActorStopped,
    AssistantTurn,
    ContinueDecision,
    ConversationState,
    ConversationView,
    Exchange,
    Scenario,
    ScenarioLimits,
    ScenarioResult,
    StopDecision,
    Transcript,
    TurnLimitReached,
    UserTurn,
)
from pydantic_multiturn_evals.storage import InMemoryStateStore, StateStore

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ActorView:
    scenario_id: str
    brief: ActorBrief
    exchanges: tuple[Exchange, ...]
    remaining_target_turns: int


class ChatTarget(Protocol):
    async def __call__(self, view: ConversationView) -> str: ...


class AdaptiveActor(Protocol):
    async def decide(self, view: ActorView) -> ActorDecision: ...


class ScenarioTimeoutError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RunnerServices:
    state_store: StateStore

    @classmethod
    def in_memory(cls) -> RunnerServices:
        return cls(state_store=InMemoryStateStore())


async def run_scenario(
    scenario: Scenario,
    *,
    limits: ScenarioLimits,
    target: ChatTarget,
    actor: AdaptiveActor,
    services: RunnerServices | None = None,
) -> ScenarioResult:
    """Run one scenario until the actor stops it or the hard turn limit wins."""

    services = services or RunnerServices.in_memory()
    run_id = uuid4().hex
    state = ConversationState(
        run_id=run_id,
        scenario_id=scenario.id,
        pending_user=UserTurn(content=scenario.first_prompt),
    )
    await services.state_store.save(state)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limits.timeout_seconds

    async def before_deadline(awaitable: Awaitable[T]) -> T:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise ScenarioTimeoutError(
                f"scenario {scenario.id!r} exceeded {limits.timeout_seconds:g} seconds"
            )
        try:
            return await asyncio.wait_for(awaitable, timeout=remaining)
        except TimeoutError as error:
            raise ScenarioTimeoutError(
                f"scenario {scenario.id!r} exceeded {limits.timeout_seconds:g} seconds"
            ) from error

    for _ in range(limits.max_target_turns):
        pending_user = state.pending_user
        if pending_user is None:  # pragma: no cover - protected by ConversationState
            raise RuntimeError("running scenario has no pending user turn")

        view = ConversationView(
            run_id=run_id,
            scenario_id=scenario.id,
            exchanges=state.exchanges,
            pending_user=pending_user,
        )
        reply = AssistantTurn(content=await before_deadline(target(view)))
        exchanges = (*state.exchanges, Exchange(user=pending_user, assistant=reply))
        remaining_turns = limits.max_target_turns - len(exchanges)
        decision = await before_deadline(
            actor.decide(
                ActorView(
                    scenario_id=scenario.id,
                    brief=scenario.actor,
                    exchanges=exchanges,
                    remaining_target_turns=remaining_turns,
                )
            )
        )
        decisions = (*state.decisions, decision)

        if isinstance(decision, ContinueDecision) and remaining_turns > 0:
            state = ConversationState(
                run_id=run_id,
                scenario_id=scenario.id,
                exchanges=exchanges,
                pending_user=UserTurn(content=decision.next_user_message),
                decisions=decisions,
            )
            await services.state_store.save(state)
            continue

        if isinstance(decision, AcceptDecision):
            termination = ActorAccepted(reason=decision.reason)
        elif isinstance(decision, StopDecision):
            termination = ActorStopped(reason=decision.reason)
        else:
            termination = TurnLimitReached(limit=limits.max_target_turns)

        state = ConversationState(
            run_id=run_id,
            scenario_id=scenario.id,
            exchanges=exchanges,
            pending_user=None,
            decisions=decisions,
            termination=termination,
        )
        await services.state_store.save(state)
        return ScenarioResult(
            run_id=run_id,
            scenario_id=scenario.id,
            transcript=Transcript(exchanges=exchanges),
            decisions=decisions,
            termination=termination,
        )

    raise RuntimeError("scenario loop exhausted without a terminal result")  # pragma: no cover
