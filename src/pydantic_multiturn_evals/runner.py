"""The bounded adaptive conversation loop."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Protocol, TypeVar
from uuid import uuid4

from pydantic_multiturn_evals.models import (
    AcceptDecision,
    ActorAccepted,
    ActorBrief,
    ActorDecision,
    ActorStopped,
    AssistantTurn,
    CaseKey,
    ContinueDecision,
    ConversationState,
    ConversationView,
    Exchange,
    FixtureEntry,
    ModelSpec,
    Scenario,
    ScenarioLimits,
    ScenarioResult,
    SessionContext,
    SessionOutcome,
    StopDecision,
    TargetTurnEvidence,
    Transcript,
    TurnLimitReached,
    UserTurn,
)
from pydantic_multiturn_evals.observability import NO_TRACE, TraceFields, TraceRuntime
from pydantic_multiturn_evals.storage import InMemoryStateStore, StateStore
from pydantic_multiturn_evals.targets import Target

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ActorView:
    scenario_id: str
    brief: ActorBrief
    exchanges: tuple[Exchange, ...]
    remaining_target_turns: int


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
    target: Target,
    actor: AdaptiveActor,
    services: RunnerServices | None = None,
    suite_name: str = "library",
    key: CaseKey | None = None,
    comparison_id: str | None = None,
    arm: str | None = None,
    target_model: ModelSpec | None = None,
    target_instructions: str | None = None,
    fixture: tuple[FixtureEntry, ...] = (),
    trace: TraceRuntime = NO_TRACE,
) -> ScenarioResult:
    """Run one scenario until the actor stops it or the hard turn limit wins."""

    services = services or RunnerServices.in_memory()
    run_id = uuid4().hex
    case_key = key or CaseKey(scenario_id=scenario.id, repeat_index=1)
    session_context = SessionContext(
        suite_name=suite_name,
        target_name=target.name,
        target_kind=target.kind,
        target_version=target.version,
        key=case_key,
        run_id=run_id,
        target_model=target_model,
        target_instructions=target_instructions,
        fixture=fixture,
        comparison_id=comparison_id,
        arm=arm,
    )
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

    trace_fields = TraceFields(
        suite=suite_name,
        target=target.name,
        harness=target.kind,
        comparison_id=comparison_id,
        arm=arm,
        scenario_id=scenario.id,
        repeat_index=case_key.repeat_index,
        run_id=run_id,
        version=str(target.version),
    )
    evidence: list[TargetTurnEvidence] = []
    with trace.span(
        "multiturn-evals.scenario",
        trace_fields,
        input={"first_prompt": scenario.first_prompt},
        as_type="agent",
    ) as scenario_span:
        async with target.session(session_context) as target_session:
            for turn_index in range(1, limits.max_target_turns + 1):
                pending_user = state.pending_user
                if pending_user is None:  # pragma: no cover - protected by ConversationState
                    raise RuntimeError("running scenario has no pending user turn")

                view = ConversationView(
                    run_id=run_id,
                    scenario_id=scenario.id,
                    exchanges=state.exchanges,
                    pending_user=pending_user,
                )
                turn_fields = replace(trace_fields, turn_index=turn_index)
                started = perf_counter()
                with scenario_span.child(
                    "multiturn-evals.target-turn",
                    turn_fields,
                    input=[
                        {"role": message.role, "content": message.content}
                        for message in view.messages
                    ],
                    as_type="agent",
                ) as turn_span:
                    target_reply = await before_deadline(target_session.reply(view))
                    duration = perf_counter() - started
                    turn_span.update(
                        {
                            "assistant_text": target_reply.assistant_text,
                            "duration_seconds": duration,
                        }
                    )
                evidence.append(
                    TargetTurnEvidence(
                        turn_index=turn_index,
                        duration_seconds=duration,
                        session_id=target_reply.session_id,
                        executable=getattr(target, "executable", None),
                        details=target_reply.evidence,
                    )
                )
                reply = AssistantTurn(content=target_reply.assistant_text)
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
                transcript = Transcript(exchanges=exchanges)
                completion = await before_deadline(
                    target_session.finish(
                        SessionOutcome(
                            transcript=transcript,
                            decisions=decisions,
                            termination=termination,
                        )
                    )
                )
                result = ScenarioResult(
                    run_id=run_id,
                    scenario_id=scenario.id,
                    repeat_index=case_key.repeat_index,
                    transcript=transcript,
                    target_evidence=tuple(evidence),
                    decisions=decisions,
                    termination=termination,
                    completion=completion,
                )
                span_output: dict[str, object] = {
                    "termination": termination.kind,
                    "target_turns": len(exchanges),
                }
                if completion.environment is not None:
                    environment = completion.environment
                    span_output.update(
                        {
                            "environment_provider": environment.provider,
                            "environment_passed": environment.passed,
                            "environment_reward": environment.reward,
                        }
                    )
                    scenario_span.score(
                        "environment_pass", float(environment.passed), environment.reason
                    )
                    if environment.reward is not None:
                        scenario_span.score(
                            "environment_reward", environment.reward, environment.reason
                        )
                scenario_span.update(span_output)
                return result

    raise RuntimeError("scenario loop exhausted without a terminal result")  # pragma: no cover
