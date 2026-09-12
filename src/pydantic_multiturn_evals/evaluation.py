"""Compile scenarios into Pydantic cases and derive the CI gate."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any

from pydantic import BaseModel
from pydantic_ai import ModelSettings
from pydantic_ai.models import Model
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    Evaluator,
    EvaluatorContext,
    EvaluatorOutput,
    LLMJudge,
)
from pydantic_evals.reporting import EvaluationReport

from pydantic_multiturn_evals.model_bindings import BoundModel, bind_model
from pydantic_multiturn_evals.models import (
    CaseGate,
    CaseKey,
    FixtureEntry,
    GatePolicy,
    GateResult,
    ModelSpec,
    PlannedCase,
    ScenarioResult,
    SuiteSpec,
    TargetFailureEvidence,
)
from pydantic_multiturn_evals.observability import NO_TRACE, TraceFields, TraceRuntime
from pydantic_multiturn_evals.providers import PydanticActor
from pydantic_multiturn_evals.runner import (
    AdaptiveActor,
    RunnerServices,
    run_scenario,
)
from pydantic_multiturn_evals.spec import load_suite
from pydantic_multiturn_evals.storage import InMemoryStateStore, StateStore
from pydantic_multiturn_evals.targets import Target
from pydantic_multiturn_evals.trajectory import (
    CaseTrajectory,
    TrajectoryAssessor,
    assess_trajectory,
)


@dataclass
class TranscriptJudge(Evaluator[PlannedCase, ScenarioResult, None]):
    """Run Pydantic's LLMJudge against visible exchanges only."""

    rubric: str
    model: Model
    settings: ModelSettings

    async def evaluate(
        self,
        ctx: EvaluatorContext[PlannedCase, ScenarioResult, None],
    ) -> EvaluatorOutput:
        judge = LLMJudge(
            rubric=self.rubric,
            model=self.model,
            model_settings=self.settings,
            include_input=False,
            include_expected_output=False,
            score={"evaluation_name": "judge_score", "include_reason": True},
            assertion={"evaluation_name": "judge_pass", "include_reason": True},
        )
        judge_context = replace(
            ctx,
            inputs=None,
            expected_output=None,
            output=ctx.output.transcript,
        )
        return await judge.evaluate(judge_context)


def build_dataset(
    suite: SuiteSpec,
    *,
    judge_binding: BoundModel,
    repeat: int = 1,
    planned_cases: tuple[PlannedCase, ...] | None = None,
) -> Dataset[PlannedCase, ScenarioResult, None]:
    cases = [
        Case[PlannedCase, ScenarioResult, None](
            name=planned.case_name,
            inputs=planned,
            evaluators=(
                TranscriptJudge(
                    rubric=planned.scenario.judge_rubric,
                    model=judge_binding.model,
                    settings=judge_binding.settings,
                ),
            ),
        )
        for planned in (planned_cases or plan_cases(suite, repeat))
    ]
    return Dataset(name=suite.name, cases=cases)


@dataclass(frozen=True, slots=True)
class SuiteResult:
    report: EvaluationReport[PlannedCase, ScenarioResult, None]
    gate: GateResult
    failure_evidence: tuple[TargetFailureEvidence, ...] = ()
    trajectories: tuple[CaseTrajectory, ...] = ()

    def write_artifacts(self, directory: str | Path) -> None:
        output_directory = Path(directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        _write_text(output_directory / "gate.json", self.gate.model_dump_json(indent=2) + "\n")
        _write_text(
            output_directory / "report.json",
            json.dumps(_report_payload(self.report), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
        )
        _write_text(
            output_directory / "report.txt",
            self.report.render(include_output=False, include_reasons=True),
        )
        transcript_lines = [
            json.dumps(
                {
                    "run_id": case.output.run_id,
                    "scenario_id": case.output.scenario_id,
                    "repeat_index": case.output.repeat_index,
                    "transcript": case.output.transcript.model_dump(mode="json"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            for case in self.report.cases
        ]
        transcript_text = "\n".join(transcript_lines)
        _write_text(
            output_directory / "transcripts.jsonl",
            transcript_text + ("\n" if transcript_text else ""),
        )
        evidence_lines = [
            json.dumps(
                {
                    "run_id": case.output.run_id,
                    "scenario_id": case.output.scenario_id,
                    "repeat_index": case.output.repeat_index,
                    "turns": [item.model_dump(mode="json") for item in case.output.target_evidence],
                    "completion": case.output.completion.model_dump(mode="json"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            for case in self.report.cases
        ]
        evidence_lines.extend(item.model_dump_json() for item in self.failure_evidence)
        evidence_text = "\n".join(evidence_lines)
        _write_text(
            output_directory / "evidence.jsonl",
            evidence_text + ("\n" if evidence_text else ""),
        )
        trajectory_text = "\n".join(item.model_dump_json() for item in self.trajectories)
        _write_text(
            output_directory / "trajectory.jsonl",
            trajectory_text + ("\n" if trajectory_text else ""),
        )


def plan_cases(suite: SuiteSpec, repeat: int) -> tuple[PlannedCase, ...]:
    if repeat < 1:
        raise ValueError("repeat must be positive")
    return tuple(
        PlannedCase(
            key=CaseKey(scenario_id=scenario.id, repeat_index=repeat_index),
            case_name=(scenario.id if repeat == 1 else f"{scenario.id} [{repeat_index}/{repeat}]"),
            scenario=scenario,
        )
        for scenario in suite.scenarios
        for repeat_index in range(1, repeat + 1)
    )


async def evaluate_suite(
    source: str | Path | SuiteSpec,
    *,
    target: Target,
    actor: AdaptiveActor | None = None,
    judge_binding: BoundModel | None = None,
    state_store: StateStore | None = None,
    max_concurrency: int = 1,
    repeat: int = 1,
    planned_cases: tuple[PlannedCase, ...] | None = None,
    progress: bool = True,
    comparison_id: str | None = None,
    arm: str | None = None,
    target_model: ModelSpec | None = None,
    target_instructions: str | None = None,
    fixture: tuple[FixtureEntry, ...] = (),
    capture_target_evidence: bool = True,
    trajectory_assessor: TrajectoryAssessor | None = None,
    trajectory_rubric: str | None = None,
    max_trajectory_prefixes: int = 4,
    trace: TraceRuntime = NO_TRACE,
) -> SuiteResult:
    """Load, run, judge, aggregate, and gate one adaptive scenario suite."""

    suite = source if isinstance(source, SuiteSpec) else load_suite(source)
    if (trajectory_assessor is None) != (trajectory_rubric is None):
        raise ValueError("trajectory assessor and rubric must be provided together")
    adaptive_actor = actor or PydanticActor(suite.actor)
    evaluator_binding = judge_binding or bind_model(suite.judge.model)
    services = RunnerServices(state_store=state_store or InMemoryStateStore())
    dataset = build_dataset(
        suite,
        judge_binding=evaluator_binding,
        repeat=repeat,
        planned_cases=planned_cases,
    )

    async def task(planned: PlannedCase) -> ScenarioResult:
        result = await run_scenario(
            planned.scenario,
            limits=suite.limits_for(planned.scenario),
            target=target,
            actor=adaptive_actor,
            services=services,
            suite_name=suite.name,
            key=planned.key,
            comparison_id=comparison_id,
            arm=arm,
            target_model=target_model,
            target_instructions=target_instructions,
            fixture=fixture,
            trace=trace,
        )
        if capture_target_evidence:
            return result
        return result.model_copy(update={"target_evidence": ()})

    fields = TraceFields(
        suite=suite.name,
        target=target.name,
        harness=target.kind,
        comparison_id=comparison_id,
        arm=arm,
        version=str(target.version),
    )
    with trace.span("multiturn-evals.arm", fields, as_type="evaluator") as arm_span:
        report = await dataset.evaluate(
            task,
            name=suite.name,
            task_name="adaptive_conversation",
            max_concurrency=max_concurrency,
            repeat=1,
            progress=progress,
            metadata={
                "suite": suite.name,
                "schema_version": suite.version,
                "comparison_id": comparison_id,
                "arm": arm,
                "target": target.name,
                "harness": target.kind,
            },
        )
        gate = derive_gate(report, suite.gate)
        arm_span.score("case_pass_rate", gate.case_pass_rate)
        arm_span.score("mean_score", gate.mean_score)
        arm_span.score("gate_pass", float(gate.passed))
        arm_span.update(gate.model_dump(mode="json"))
    gates = {case.case_name: case for case in gate.cases}
    for case in report.cases:
        outcome = gates[case.name]
        trace.score_run(
            case.output.run_id,
            case_name=case.name,
            score=outcome.score,
            assertion=outcome.assertion,
            passed=outcome.passed,
            reason=outcome.reason,
        )
    trajectories = await _assess_trajectories(
        report,
        assessor=trajectory_assessor,
        rubric=trajectory_rubric,
        max_prefixes=max_trajectory_prefixes,
        max_concurrency=max_concurrency,
    )
    return SuiteResult(
        report=report,
        gate=gate,
        failure_evidence=target.failure_evidence(),
        trajectories=trajectories,
    )


async def _assess_trajectories(
    report: EvaluationReport[PlannedCase, ScenarioResult, None],
    *,
    assessor: TrajectoryAssessor | None,
    rubric: str | None,
    max_prefixes: int,
    max_concurrency: int,
) -> tuple[CaseTrajectory, ...]:
    if assessor is None or rubric is None:
        return ()
    if max_prefixes < 1:
        raise ValueError("max_trajectory_prefixes must be positive")

    semaphore = asyncio.Semaphore(max_concurrency)

    async def assess_case(case: Any) -> CaseTrajectory:
        async with semaphore:
            assessment = await assess_trajectory(
                case.output.transcript,
                rubric=rubric,
                assessor=assessor,
                max_prefixes=max_prefixes,
            )
        return CaseTrajectory(
            case_name=case.name,
            scenario_id=case.output.scenario_id,
            repeat_index=case.output.repeat_index,
            assessment=assessment,
        )

    return tuple(await asyncio.gather(*(assess_case(case) for case in report.cases)))


def derive_gate(
    report: EvaluationReport[PlannedCase, ScenarioResult, None],
    policy: GatePolicy,
) -> GateResult:
    outcomes: list[CaseGate] = []
    scores: list[float] = []

    for case in report.cases:
        errors = tuple(
            f"{failure.name}: {failure.error_message}" for failure in case.evaluator_failures
        )
        score_result = case.scores.get("judge_score")
        assertion_result = case.assertions.get("judge_pass")
        score = float(score_result.value) if score_result is not None else None
        assertion = bool(assertion_result.value) if assertion_result is not None else None
        environment = case.output.completion.environment
        validation_errors = list(errors)
        if score is None or not math.isfinite(score):
            validation_errors.append("judge_score is missing or not finite")
            score = None
        else:
            scores.append(score)
        if assertion is None:
            validation_errors.append("judge_pass is missing")
        if environment is not None and not environment.passed:
            explanation = environment.reason or "no reason was provided"
            validation_errors.append(f"environment verifier failed: {explanation}")
        reason = None
        if assertion_result is not None:
            reason = assertion_result.reason
        elif score_result is not None:
            reason = score_result.reason
        outcomes.append(
            CaseGate(
                case_name=case.name,
                scenario_id=case.inputs.key.scenario_id,
                repeat_index=case.inputs.key.repeat_index,
                passed=assertion is True and not validation_errors,
                score=score,
                assertion=assertion,
                environment_passed=(environment.passed if environment is not None else None),
                environment_reward=(environment.reward if environment is not None else None),
                reason=reason,
                errors=tuple(validation_errors),
            )
        )

    for failure in report.failures:
        outcomes.append(
            CaseGate(
                case_name=failure.name,
                scenario_id=failure.inputs.key.scenario_id,
                repeat_index=failure.inputs.key.repeat_index,
                passed=False,
                errors=(failure.error_message,),
            )
        )

    total = len(outcomes)
    passed_count = sum(outcome.passed for outcome in outcomes)
    pass_rate = passed_count / total if total else 0.0
    mean_score = fmean(scores) if scores else 0.0
    has_errors = any(outcome.errors for outcome in outcomes) or bool(
        report.report_evaluator_failures
    )
    passed = (
        total > 0
        and not has_errors
        and pass_rate >= policy.minimum_case_pass_rate
        and mean_score >= policy.minimum_mean_score
    )
    return GateResult(
        passed=passed,
        case_pass_rate=pass_rate,
        mean_score=mean_score,
        cases=tuple(outcomes),
    )


def _report_payload(
    report: EvaluationReport[PlannedCase, ScenarioResult, None],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": report.name,
        "experiment_metadata": report.experiment_metadata,
        "trace_id": report.trace_id,
        "span_id": report.span_id,
        "cases": [
            {
                "name": case.name,
                "source_case_name": case.source_case_name,
                "input": case.inputs.scenario.model_dump(mode="json"),
                "output": {
                    "run_id": case.output.run_id,
                    "scenario_id": case.output.scenario_id,
                    "repeat_index": case.output.repeat_index,
                    "transcript": case.output.transcript.model_dump(mode="json"),
                    "evidence_file": "evidence.jsonl",
                },
                "scores": _evaluation_values(case.scores),
                "assertions": _evaluation_values(case.assertions),
                "metrics": case.metrics,
                "attributes": case.attributes,
                "task_duration": case.task_duration,
                "total_duration": case.total_duration,
                "trace_id": case.trace_id,
                "span_id": case.span_id,
                "evaluator_failures": [
                    {
                        "name": failure.name,
                        "error_type": failure.error_type,
                        "error_message": failure.error_message,
                    }
                    for failure in case.evaluator_failures
                ],
            }
            for case in report.cases
        ],
        "task_failures": [
            {
                "name": failure.name,
                "source_case_name": failure.source_case_name,
                "input": _json_value(failure.inputs),
                "error_message": failure.error_message,
                "error_stacktrace": failure.error_stacktrace,
                "trace_id": failure.trace_id,
                "span_id": failure.span_id,
            }
            for failure in report.failures
        ],
        "report_evaluator_failures": [
            {"name": failure.name, "error_message": failure.error_message}
            for failure in report.report_evaluator_failures
        ],
    }


def _evaluation_values(values: dict[str, Any]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for name, result in values.items():
        payload[name] = {
            "value": result.value,
            "reason": result.reason,
            "evaluator_version": result.evaluator_version,
        }
    return payload


def _json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
