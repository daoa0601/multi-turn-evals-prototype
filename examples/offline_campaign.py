"""A complete local campaign using real persistent child processes."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic_multiturn_evals import (
    Campaign,
    Case,
    ComponentDefinition,
    Criterion,
    EvaluationInput,
    Evidence,
    FunctionEvaluator,
    Judgment,
    Message,
    MessageFact,
    NativeCheck,
    PublicOutcome,
    TraceRecorder,
    Trial,
    Variant,
)
from pydantic_multiturn_evals.harnesses import (
    Accept,
    CommandLimits,
    CommandTarget,
    Continue,
    ConversationHarness,
    FunctionHarness,
    OneShotCommandHarness,
)

HERE = Path(__file__).parent


@dataclass(frozen=True, slots=True)
class FollowUpActor:
    component = ComponentDefinition(kind="offline-actor", version="1")

    async def next_turn(
        self,
        case: Case,
        messages: tuple[Message, ...],
        turn: int,
        trace: TraceRecorder,
    ) -> Continue | Accept:
        del turn, trace
        if not messages:
            return Continue(
                message=Message(role="user", content=case.task, actor="simulated-user"),
                reason="start the case",
            )
        if messages[-1].content == "need identifier":
            return Continue(
                message=Message(role="user", content="identifier 42", actor="simulated-user"),
                reason="answer the target's question",
            )
        return Accept(reason="the target gave a final answer")


async def judge_resolution(value: EvaluationInput) -> tuple[Judgment, ...]:
    final_message = value.outcome.messages[-1].content
    passed = final_message.startswith("resolved:")
    return (
        Judgment(
            criterion="resolved",
            passed=passed,
            score=float(passed),
            reason="the final reply resolves the request" if passed else "the request stayed open",
        ),
    )


evaluator = FunctionEvaluator(
    component=ComponentDefinition(kind="resolution-evaluator", version="1"),
    function=judge_resolution,
)


async def run_direct(_case: Case, _trial: Trial, trace: TraceRecorder) -> Evidence:
    reply = Message(role="assistant", content="resolved: direct function", actor="offline-direct")
    trace.record(MessageFact(message=reply))
    return Evidence(
        public=PublicOutcome(messages=(reply,), output={"path": "direct"}),
        checks=(
            NativeCheck(
                name="completed",
                passed=True,
                reason="the direct function returned its evidence",
            ),
        ),
    )


direct = FunctionHarness(
    component=ComponentDefinition(kind="offline-direct", version="1"),
    function=run_direct,
)

job = OneShotCommandHarness(
    argv=(sys.executable, str(HERE / "oneshot_job.py")),
    cwd=HERE,
)


def target(profile: str) -> CommandTarget:
    return CommandTarget(
        argv=(sys.executable, str(HERE / "jsonl_target.py"), profile),
        cwd=HERE,
        limits=CommandLimits(response_seconds=5, shutdown_seconds=2),
    )


campaign = Campaign(
    name="offline-proof",
    revision="1",
    cases=(
        Case(
            name="account-access",
            task="I cannot access my account.",
            criteria=(Criterion(name="resolved", instruction="The final answer resolves access."),),
            required_checks=("completed",),
        ),
    ),
    samples=2,
    variants=(
        Variant(
            name="baseline",
            harness=ConversationHarness(target=target("baseline"), actor=FollowUpActor()),
            evaluators=(evaluator,),
        ),
        Variant(
            name="candidate",
            harness=ConversationHarness(target=target("candidate"), actor=FollowUpActor()),
            evaluators=(evaluator,),
        ),
        Variant(name="direct", harness=direct, evaluators=(evaluator,)),
        Variant(name="job", harness=job, evaluators=(evaluator,)),
    ),
)
