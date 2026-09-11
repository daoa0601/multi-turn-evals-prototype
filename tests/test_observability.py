from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Literal

import pytest
from pydantic_ai.models.test import TestModel

from pydantic_multiturn_evals.evaluation import evaluate_suite
from pydantic_multiturn_evals.models import (
    AcceptDecision,
    ActorBrief,
    Scenario,
    SessionContext,
    SuiteSpec,
    TargetFailureEvidence,
    TargetReply,
)
from pydantic_multiturn_evals.observability import (
    LangfuseTraceRuntime,
    TraceFields,
    enable_langfuse,
)
from pydantic_multiturn_evals.runner import ActorView, ConversationView


class RecordingSpan:
    def __init__(self, runtime: RecordingTrace, name: str, fields: TraceFields) -> None:
        self.runtime = runtime
        self.name = name
        self.fields = fields

    @contextmanager
    def child(
        self,
        name: str,
        fields: TraceFields,
        *,
        input: object | None = None,
        as_type: str = "span",
    ) -> Iterator[RecordingSpan]:
        span = RecordingSpan(self.runtime, name, fields)
        self.runtime.spans.append(span)
        yield span

    def update(self, output: object) -> None:
        return None

    def score(self, name: str, value: float, reason: str | None = None) -> None:
        self.runtime.scores.append((name, value))


class RecordingTrace:
    def __init__(self) -> None:
        self.spans: list[RecordingSpan] = []
        self.scores: list[tuple[str, float]] = []
        self.run_scores: list[tuple[str, str]] = []

    @contextmanager
    def span(
        self,
        name: str,
        fields: TraceFields,
        *,
        input: object | None = None,
        as_type: str = "span",
    ) -> Iterator[RecordingSpan]:
        span = RecordingSpan(self, name, fields)
        self.spans.append(span)
        yield span

    def score_run(
        self,
        run_id: str,
        *,
        case_name: str,
        score: float | None,
        assertion: bool | None,
        passed: bool,
        reason: str | None,
    ) -> None:
        self.run_scores.append((run_id, case_name))

    def shutdown(self) -> None:
        return None


class OneReplyTarget:
    name = "recorded-target"
    kind: Literal["command"] = "command"
    version = 1

    @asynccontextmanager
    async def session(self, context: SessionContext) -> AsyncIterator[OneReplyTarget]:
        yield self

    async def reply(self, view: ConversationView) -> TargetReply:
        return TargetReply(assistant_text="A safe answer.", evidence={"secret": "local-only"})

    def failure_evidence(self) -> tuple[TargetFailureEvidence, ...]:
        return ()


class AcceptingActor:
    async def decide(self, view: ActorView) -> AcceptDecision:
        return AcceptDecision(reason="Done.")


def test_trace_fields_identify_the_arm_case_and_harness_without_target_evidence() -> None:
    trace = RecordingTrace()
    suite = SuiteSpec(
        version=1,
        name="trace-suite",
        scenarios=(
            Scenario(
                id="help",
                first_prompt="Help.",
                actor=ActorBrief(persona="User", goal="Get help"),
                judge_rubric="The answer helps.",
            ),
        ),
    )

    asyncio.run(
        evaluate_suite(
            suite,
            target=OneReplyTarget(),
            actor=AcceptingActor(),
            judge_model=TestModel(
                custom_output_args={"reason": "Good.", "pass": True, "score": 0.9}
            ),
            comparison_id="comparison-1",
            arm="candidate",
            trace=trace,
            progress=False,
        )
    )

    scenario = next(span for span in trace.spans if span.name.endswith("scenario"))
    assert scenario.fields.comparison_id == "comparison-1"
    assert scenario.fields.arm == "candidate"
    assert scenario.fields.scenario_id == "help"
    assert scenario.fields.repeat_index == 1
    assert scenario.fields.harness == "command"
    assert "secret" not in repr([span.fields.metadata() for span in trace.spans])
    assert trace.run_scores[0][1] == "help"
    assert ("judge_score", 0.9) not in trace.scores


class FakeObservation:
    def __init__(self) -> None:
        self.children: list[dict[str, object]] = []
        self.updates: list[dict[str, object]] = []
        self.scores: list[dict[str, object]] = []

    @contextmanager
    def start_as_current_observation(self, **values: object) -> Iterator[FakeObservation]:
        self.children.append(values)
        yield FakeObservation()

    def update(self, **values: object) -> None:
        self.updates.append(values)

    def score_trace(self, **values: object) -> None:
        self.scores.append(values)


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.observation = FakeObservation()
        self.starts: list[dict[str, object]] = []
        self.stopped = False

    @contextmanager
    def start_as_current_observation(self, **values: object) -> Iterator[FakeObservation]:
        self.starts.append(values)
        yield self.observation

    def shutdown(self) -> None:
        self.stopped = True


def test_langfuse_adapter_emits_allowlisted_metadata_and_explicit_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langfuse

    propagated: list[dict[str, object]] = []

    @contextmanager
    def fake_propagate(**values: object) -> Iterator[None]:
        propagated.append(values)
        yield

    monkeypatch.setattr(langfuse, "propagate_attributes", fake_propagate)
    client = FakeLangfuseClient()
    runtime = LangfuseTraceRuntime(client)
    fields = TraceFields(
        suite="support",
        target="candidate",
        harness="command",
        comparison_id="comparison-1",
        arm="candidate",
        scenario_id="help",
        repeat_index=2,
        run_id="run-2",
        version="1",
    )

    with runtime.span("scenario", fields, input={"prompt": "hello"}, as_type="agent") as span:
        with span.child("turn", fields, input="hello") as child:
            child.update("answer")
            child.score("turn_ok", 1)
    runtime.score_run(
        "run-2",
        case_name="help [2/3]",
        score=0.9,
        assertion=True,
        passed=True,
        reason="Good.",
    )
    runtime.score_run(
        "unknown",
        case_name="missing",
        score=None,
        assertion=None,
        passed=False,
        reason=None,
    )
    runtime.shutdown()

    assert propagated[0]["tags"] == ["multiturn-eval", "variant:candidate"]
    assert "trace_context" not in client.starts[0]
    metadata = client.starts[0]["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["repeat_index"] == 2
    assert [score["name"] for score in client.observation.scores] == [
        "judge_score",
        "judge_pass",
        "case_gate_pass",
    ]
    assert client.stopped is True
    assert "secret" not in repr(propagated)


def test_enabling_langfuse_names_missing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY"):
        enable_langfuse()
