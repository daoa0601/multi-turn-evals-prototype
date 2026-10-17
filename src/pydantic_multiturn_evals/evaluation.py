"""Validate harness evidence and run independent evaluators."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter

from pydantic_multiturn_evals.model import (
    Case,
    ComponentDefinition,
    EvaluationArtifact,
    EvaluationInput,
    Evaluator,
    Evidence,
    Judgment,
    TraceRecorder,
)


class EvaluationError(RuntimeError):
    pass


class HarnessError(RuntimeError):
    def __init__(self, phase: str, message: str, *, evidence: Evidence | None = None) -> None:
        super().__init__(message)
        self.phase = phase
        self.evidence = evidence


EvaluatorFunction = Callable[[EvaluationInput], Awaitable[tuple[Judgment, ...]]]


@dataclass(frozen=True, slots=True)
class FunctionEvaluator:
    component: ComponentDefinition
    function: EvaluatorFunction

    async def evaluate(
        self,
        value: EvaluationInput,
        trace: TraceRecorder,
    ) -> tuple[Judgment, ...]:
        del trace
        return await self.function(value)


async def evaluate(
    case: Case,
    evidence: Evidence,
    evaluators: tuple[Evaluator, ...],
    directory: Path,
    trace: TraceRecorder,
) -> tuple[Judgment, ...]:
    """Return exactly one judgment for each authored criterion."""

    _require_native_checks(case, evidence)
    value = EvaluationInput(
        task=case.task,
        criteria=case.criteria,
        outcome=evidence.public,
        artifacts=tuple(
            EvaluationArtifact(
                name=artifact.name,
                path=(directory / artifact.path).resolve(),
                media_type=artifact.media_type,
            )
            for artifact in evidence.artifacts
        ),
    )
    judgments: list[Judgment] = []
    for evaluator in evaluators:
        try:
            output = await evaluator.evaluate(value, trace)
            judgments.extend(TypeAdapter(tuple[Judgment, ...]).validate_python(output))
        except Exception as error:
            identity = f"{evaluator.component.kind}@{evaluator.component.version}"
            raise EvaluationError(f"evaluator {identity} failed: {error}") from error

    expected = {criterion.name for criterion in case.criteria}
    actual = [judgment.criterion for judgment in judgments]
    duplicates = sorted(name for name in set(actual) if actual.count(name) > 1)
    if duplicates:
        raise EvaluationError(f"duplicate judgments: {', '.join(duplicates)}")
    missing = sorted(expected - set(actual))
    extra = sorted(set(actual) - expected)
    if missing:
        raise EvaluationError(f"missing judgments: {', '.join(missing)}")
    if extra:
        raise EvaluationError(f"unexpected judgments: {', '.join(extra)}")
    return tuple(judgments)


def _require_native_checks(case: Case, evidence: Evidence) -> None:
    present = {check.name for check in evidence.checks}
    missing = sorted(set(case.required_checks) - present)
    if missing:
        raise EvaluationError(f"harness omitted required checks: {', '.join(missing)}")
