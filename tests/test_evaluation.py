from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic_ai import ModelSettings
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel

from pydantic_multiturn_evals.evaluation import evaluate_suite
from pydantic_multiturn_evals.models import (
    AcceptDecision,
    ActorBrief,
    CaseKey,
    EnvironmentEvidence,
    FixtureEntry,
    GatePolicy,
    PlannedCase,
    Scenario,
    SessionContext,
    SessionOutcome,
    SuiteSpec,
    TargetCompletion,
    TargetFailureEvidence,
    TargetReply,
)
from pydantic_multiturn_evals.runner import ActorView, ConversationView
from pydantic_multiturn_evals.trajectory import TrajectoryAssessment
from tests.helpers.model_binding import fake_model_binding


class AcceptingActor:
    async def decide(self, view: ActorView) -> AcceptDecision:
        return AcceptDecision(reason=f"Observed {len(view.exchanges)} complete exchange.")


class EvalTarget:
    name = "test-target"
    kind: Literal["command"] = "command"
    version = 1

    def __init__(
        self,
        reply: Callable[[ConversationView], Awaitable[str]],
        completion: TargetCompletion | None = None,
    ) -> None:
        self._reply = reply
        self._completion = completion or TargetCompletion()
        self.contexts: list[SessionContext] = []

    @asynccontextmanager
    async def session(self, context: SessionContext) -> AsyncIterator[EvalTarget]:
        self.contexts.append(context)
        yield self

    async def reply(self, view: ConversationView) -> TargetReply:
        return TargetReply(assistant_text=await self._reply(view))

    async def finish(self, outcome: SessionOutcome) -> TargetCompletion:
        return self._completion

    def failure_evidence(self) -> tuple[TargetFailureEvidence, ...]:
        return ()


async def helpful_reply(view: ConversationView) -> str:
    return f"I can help with: {view.pending_user.content}"


def helpful_target() -> EvalTarget:
    return EvalTarget(helpful_reply)


class RecordingJudge(TestModel):
    def __init__(self, *, custom_output_args: Any) -> None:
        super().__init__(custom_output_args=custom_output_args)
        self.requests: list[list[ModelMessage]] = []

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.requests.append(messages)
        return await super().request(messages, model_settings, model_request_parameters)


def suite(*, minimum_score: float = 0.7) -> SuiteSpec:
    return SuiteSpec(
        version=1,
        name="test-suite",
        gate=GatePolicy(minimum_mean_score=minimum_score),
        scenarios=(
            Scenario(
                id="help",
                first_prompt="Please help.",
                actor=ActorBrief(persona="private-persona-marker", goal="Get help"),
                judge_rubric="The assistant is helpful.",
            ),
        ),
    )


def test_pydantic_report_drives_a_passing_gate_and_artifacts(tmp_path: Path) -> None:
    judge = RecordingJudge(
        custom_output_args={"reason": "Helpful and direct.", "pass": True, "score": 0.9}
    )

    result = asyncio.run(
        evaluate_suite(
            suite(),
            target=helpful_target(),
            actor=AcceptingActor(),
            judge_binding=fake_model_binding(judge),
            progress=False,
        )
    )
    result.write_artifacts(tmp_path)

    assert result.gate.passed is True
    assert result.gate.mean_score == 0.9
    assert json.loads((tmp_path / "gate.json").read_text())["passed"] is True
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["cases"][0]["scores"]["judge_score"]["value"] == 0.9
    transcript = json.loads((tmp_path / "transcripts.jsonl").read_text())
    assert transcript["transcript"]["exchanges"][0]["user"]["content"] == "Please help."
    judge_request = repr(judge.requests)
    assert "Please help." in judge_request
    assert "private-persona-marker" not in judge_request
    assert "actor_accepted" not in judge_request


def test_gate_fails_when_the_judge_score_misses_the_suite_threshold() -> None:
    judge = TestModel(custom_output_args={"reason": "Incomplete.", "pass": True, "score": 0.5})

    result = asyncio.run(
        evaluate_suite(
            suite(minimum_score=0.8),
            target=helpful_target(),
            actor=AcceptingActor(),
            judge_binding=fake_model_binding(judge),
            progress=False,
        )
    )

    assert result.gate.passed is False
    assert result.gate.cases[0].assertion is True


def test_target_error_becomes_a_failed_pydantic_case() -> None:
    async def broken_target(view: ConversationView) -> str:
        raise RuntimeError(f"target unavailable for {view.scenario_id}")

    result = asyncio.run(
        evaluate_suite(
            suite(),
            target=EvalTarget(broken_target),
            actor=AcceptingActor(),
            judge_binding=fake_model_binding(TestModel()),
            progress=False,
        )
    )

    assert result.gate.passed is False
    assert result.gate.cases[0].errors == ("RuntimeError: target unavailable for help",)


def test_environment_verification_is_a_gate_but_not_judge_input(tmp_path: Path) -> None:
    judge = RecordingJudge(custom_output_args={"reason": "Helpful.", "pass": True, "score": 0.9})
    target = EvalTarget(
        helpful_reply,
        completion=TargetCompletion(
            environment=EnvironmentEvidence(
                provider="agentenv",
                environment_id="sandbox-secret-marker",
                verifier="tests/verify.py",
                passed=False,
                reward=0.25,
                reason="Expected file was missing.",
            )
        ),
    )

    result = asyncio.run(
        evaluate_suite(
            suite(),
            target=target,
            actor=AcceptingActor(),
            judge_binding=fake_model_binding(judge),
            progress=False,
        )
    )
    result.write_artifacts(tmp_path)

    case = result.gate.cases[0]
    assert case.passed is False
    assert case.environment_passed is False
    assert case.environment_reward == 0.25
    assert case.errors == ("environment verifier failed: Expected file was missing.",)
    assert "sandbox-secret-marker" not in repr(judge.requests)
    evidence = json.loads((tmp_path / "evidence.jsonl").read_text())
    assert evidence["completion"]["environment"]["environment_id"] == "sandbox-secret-marker"


def test_trajectory_failure_is_visible_but_cannot_fail_the_primary_gate(tmp_path: Path) -> None:
    class BrokenObserver:
        async def assess(self, **kwargs: object) -> TrajectoryAssessment:
            raise RuntimeError("observer unavailable")

    result = asyncio.run(
        evaluate_suite(
            suite(),
            target=helpful_target(),
            actor=AcceptingActor(),
            judge_binding=fake_model_binding(
                TestModel(
                    custom_output_args={"reason": "Helpful.", "pass": True, "score": 0.9}
                )
            ),
            trajectory_assessor=BrokenObserver(),
            trajectory_rubric="Judge progress and safety at this point.",
            max_trajectory_prefixes=2,
            progress=False,
        )
    )
    result.write_artifacts(tmp_path)

    assert result.gate.passed is True
    assert result.trajectories[0].assessment.status == "partial"
    assert result.trajectories[0].assessment.points[0].error == (
        "RuntimeError: observer unavailable"
    )
    saved = json.loads((tmp_path / "trajectory.jsonl").read_text())
    assert saved["assessment"]["points"][0]["status"] == "failed"


def test_evaluation_passes_prompt_and_fixture_to_each_target_session() -> None:
    target = helpful_target()

    result = asyncio.run(
        evaluate_suite(
            suite(),
            target=target,
            actor=AcceptingActor(),
            judge_binding=fake_model_binding(
                TestModel(
                    custom_output_args={"reason": "Helpful.", "pass": True, "score": 0.9}
                )
            ),
            target_instructions="Use the selected target prompt.",
            fixture=(FixtureEntry(name="account_tier", value="priority"),),
            capture_target_evidence=False,
            progress=False,
        )
    )

    assert target.contexts[0].target_instructions == "Use the selected target prompt."
    assert target.contexts[0].fixture == (
        FixtureEntry(name="account_tier", value="priority"),
    )
    assert result.report.cases[0].output.target_evidence == ()


def test_evaluation_preserves_a_caller_planned_repeat_key() -> None:
    source = suite()
    planned = PlannedCase(
        key=CaseKey(scenario_id="help", repeat_index=7),
        case_name="help [7]",
        scenario=source.scenarios[0],
    )

    result = asyncio.run(
        evaluate_suite(
            source,
            target=helpful_target(),
            actor=AcceptingActor(),
            judge_binding=fake_model_binding(
                TestModel(
                    custom_output_args={"reason": "Helpful.", "pass": True, "score": 0.9}
                )
            ),
            planned_cases=(planned,),
            progress=False,
        )
    )

    assert result.gate.cases[0].repeat_index == 7
    assert result.report.cases[0].output.repeat_index == 7
