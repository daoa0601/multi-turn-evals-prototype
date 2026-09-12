"""Exactly-two-arm comparison orchestration and artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic_multiturn_evals.evaluation import evaluate_suite, plan_cases
from pydantic_multiturn_evals.model_bindings import BoundModel, bind_model
from pydantic_multiturn_evals.models import (
    CaseGate,
    CaseKey,
    ExecutionKind,
    GateResult,
    StrictModel,
    SuiteSpec,
)
from pydantic_multiturn_evals.observability import NO_TRACE, TraceFields, TraceRuntime
from pydantic_multiturn_evals.providers import PydanticActor
from pydantic_multiturn_evals.runner import AdaptiveActor
from pydantic_multiturn_evals.spec import LoadedTarget, load_suite
from pydantic_multiturn_evals.targets import build_target


class ArmSummary(StrictModel):
    role: Literal["baseline", "candidate"]
    target_name: str
    target_kind: ExecutionKind
    passed: bool
    case_pass_rate: float
    mean_score: float


class PairResult(StrictModel):
    key: CaseKey
    case_name: str
    baseline: CaseGate
    candidate: CaseGate
    score_delta_candidate_minus_baseline: float | None


class ComparisonSummary(StrictModel):
    schema_version: Literal[1] = 1
    comparison_id: str
    suite: str
    passed: bool
    baseline: ArmSummary
    candidate: ArmSummary
    case_pass_rate_delta_candidate_minus_baseline: float
    mean_score_delta_candidate_minus_baseline: float
    pairs: tuple[PairResult, ...]


class ArmResult(Protocol):
    @property
    def gate(self) -> GateResult: ...

    @property
    def report(self) -> Any: ...

    def write_artifacts(self, directory: str | Path) -> None: ...


@dataclass(frozen=True, slots=True)
class CompletedArm:
    role: Literal["baseline", "candidate"]
    target_name: str
    target_kind: ExecutionKind
    result: ArmResult

    def summary(self) -> ArmSummary:
        gate = self.result.gate
        return ArmSummary(
            role=self.role,
            target_name=self.target_name,
            target_kind=self.target_kind,
            passed=gate.passed,
            case_pass_rate=gate.case_pass_rate,
            mean_score=gate.mean_score,
        )


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    comparison_id: str
    suite_name: str
    baseline: CompletedArm
    candidate: CompletedArm
    pairs: tuple[PairResult, ...]

    @property
    def passed(self) -> bool:
        return self.baseline.result.gate.passed and self.candidate.result.gate.passed

    def summary(self) -> ComparisonSummary:
        baseline = self.baseline.summary()
        candidate = self.candidate.summary()
        return ComparisonSummary(
            comparison_id=self.comparison_id,
            suite=self.suite_name,
            passed=self.passed,
            baseline=baseline,
            candidate=candidate,
            case_pass_rate_delta_candidate_minus_baseline=(
                candidate.case_pass_rate - baseline.case_pass_rate
            ),
            mean_score_delta_candidate_minus_baseline=(candidate.mean_score - baseline.mean_score),
            pairs=self.pairs,
        )

    def write_artifacts(self, directory: str | Path) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        self.baseline.result.write_artifacts(output / "baseline")
        self.candidate.result.write_artifacts(output / "candidate")
        summary = self.summary()
        _write_text(output / "comparison.json", summary.model_dump_json(indent=2) + "\n")
        _write_text(
            output / "gate.json",
            json.dumps({"schema_version": 1, "passed": summary.passed}, indent=2) + "\n",
        )
        _write_text(output / "comparison.txt", _render_summary(summary))


ActorFactory = Callable[[SuiteSpec], AdaptiveActor]
JudgeBindingFactory = Callable[[SuiteSpec], BoundModel]


async def compare_suite(
    source: str | Path | SuiteSpec,
    *,
    baseline: LoadedTarget,
    candidate: LoadedTarget,
    actor_factory: ActorFactory | None = None,
    judge_binding_factory: JudgeBindingFactory | None = None,
    max_concurrency: int = 1,
    repeat: int = 1,
    progress: bool = True,
    trace: TraceRuntime = NO_TRACE,
) -> ComparisonResult:
    """Run isolated baseline and candidate arms, then pair their planned cases."""

    suite = source if isinstance(source, SuiteSpec) else load_suite(source)
    if baseline.spec.name == candidate.spec.name:
        raise ValueError("baseline and candidate need distinct target names")
    baseline_target = build_target(baseline)
    candidate_target = build_target(candidate)
    make_actor = actor_factory or (lambda value: PydanticActor(value.actor))
    make_judge = judge_binding_factory or (lambda value: bind_model(value.judge.model))
    comparison_id = uuid4().hex
    fields = TraceFields(suite=suite.name, comparison_id=comparison_id)

    with trace.span("multiturn-evals.comparison", fields, as_type="evaluator") as span:
        baseline_result = await evaluate_suite(
            suite,
            target=baseline_target,
            actor=make_actor(suite),
            judge_binding=make_judge(suite),
            max_concurrency=max_concurrency,
            repeat=repeat,
            progress=progress,
            comparison_id=comparison_id,
            arm="baseline",
            trace=trace,
        )
        candidate_result = await evaluate_suite(
            suite,
            target=candidate_target,
            actor=make_actor(suite),
            judge_binding=make_judge(suite),
            max_concurrency=max_concurrency,
            repeat=repeat,
            progress=progress,
            comparison_id=comparison_id,
            arm="candidate",
            trace=trace,
        )
        baseline_arm = CompletedArm(
            role="baseline",
            target_name=baseline_target.name,
            target_kind=baseline_target.kind,
            result=baseline_result,
        )
        candidate_arm = CompletedArm(
            role="candidate",
            target_name=candidate_target.name,
            target_kind=candidate_target.kind,
            result=candidate_result,
        )
        pairs = pair_results(suite, repeat, baseline_result, candidate_result)
        result = ComparisonResult(
            comparison_id=comparison_id,
            suite_name=suite.name,
            baseline=baseline_arm,
            candidate=candidate_arm,
            pairs=pairs,
        )
        summary = result.summary()
        span.score("mean_score_delta", summary.mean_score_delta_candidate_minus_baseline)
        span.score(
            "case_pass_rate_delta",
            summary.case_pass_rate_delta_candidate_minus_baseline,
        )
        span.score("comparison_gate_pass", float(summary.passed))
        span.update(summary.model_dump(mode="json"))
        return result


def pair_results(
    suite: SuiteSpec,
    repeat: int,
    baseline: ArmResult,
    candidate: ArmResult,
) -> tuple[PairResult, ...]:
    baseline_cases = {case.case_name: case for case in baseline.gate.cases}
    candidate_cases = {case.case_name: case for case in candidate.gate.cases}
    planned = plan_cases(suite, repeat)
    expected = {case.case_name for case in planned}
    if set(baseline_cases) != expected or set(candidate_cases) != expected:
        raise RuntimeError("comparison arms did not return every planned case")
    pairs: list[PairResult] = []
    for case in planned:
        baseline_case = baseline_cases[case.case_name]
        candidate_case = candidate_cases[case.case_name]
        delta = None
        if baseline_case.score is not None and candidate_case.score is not None:
            delta = candidate_case.score - baseline_case.score
        pairs.append(
            PairResult(
                key=case.key,
                case_name=case.case_name,
                baseline=baseline_case,
                candidate=candidate_case,
                score_delta_candidate_minus_baseline=delta,
            )
        )
    return tuple(pairs)


def _render_summary(summary: ComparisonSummary) -> str:
    return (
        f"Comparison {summary.comparison_id}\n"
        f"Suite: {summary.suite}\n"
        f"Baseline ({summary.baseline.target_name}): "
        f"score={summary.baseline.mean_score:.3f}, "
        f"pass_rate={summary.baseline.case_pass_rate:.3f}\n"
        f"Candidate ({summary.candidate.target_name}): "
        f"score={summary.candidate.mean_score:.3f}, "
        f"pass_rate={summary.candidate.case_pass_rate:.3f}\n"
        f"Candidate - baseline mean score: "
        f"{summary.mean_score_delta_candidate_minus_baseline:+.3f}\n"
        f"Passed: {summary.passed}\n"
    )


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
