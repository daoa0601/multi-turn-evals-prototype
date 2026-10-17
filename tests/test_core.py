from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pydantic_multiturn_evals import (
    Artifact,
    Campaign,
    Case,
    ComponentDefinition,
    Criterion,
    EvaluationInput,
    Evidence,
    FunctionEvaluator,
    Judgment,
    NativeCheck,
    TrialKey,
    Variant,
)
from pydantic_multiturn_evals.evaluation import EvaluationError, evaluate
from pydantic_multiturn_evals.harnesses import FunctionHarness
from pydantic_multiturn_evals.model import TraceRecorder, Trial
from pydantic_multiturn_evals.tracing import LocalTrace


def test_required_native_check_is_a_harness_contract(tmp_path: Path) -> None:
    case = Case(name="state", task="Change state", required_checks=("state-changed",))
    evidence = Evidence()

    with pytest.raises(EvaluationError, match="harness omitted required checks: state-changed"):
        _evaluate(case, evidence, (), tmp_path)


def test_evaluators_must_cover_authored_criteria_once(tmp_path: Path) -> None:
    case = Case(
        name="answer",
        task="Answer clearly",
        criteria=(Criterion(name="clear", instruction="The answer is clear."),),
    )

    async def no_judgments(_value: EvaluationInput) -> tuple[Judgment, ...]:
        return ()

    evaluator = FunctionEvaluator(
        component=ComponentDefinition(kind="empty-evaluator", version="1"),
        function=no_judgments,
    )
    with pytest.raises(EvaluationError, match="missing judgments: clear"):
        _evaluate(case, Evidence(), (evaluator,), tmp_path)


def test_failing_native_check_is_quality_evidence_not_an_execution_error(tmp_path: Path) -> None:
    case = Case(name="state", task="Change state", required_checks=("state-changed",))
    evidence = Evidence(
        checks=(NativeCheck(name="state-changed", passed=False, reason="state is unchanged"),)
    )

    assert _evaluate(case, evidence, (), tmp_path) == ()


def test_evaluator_sees_public_output_and_resolved_artifacts_but_not_private_data(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "patch.diff"
    artifact_path.write_text("a useful patch", encoding="utf-8")
    seen: list[EvaluationInput] = []

    async def inspect(value: EvaluationInput) -> tuple[Judgment, ...]:
        seen.append(value)
        return (Judgment(criterion="useful", passed=True, reason="patch is useful"),)

    case = Case(
        name="patch",
        task="Repair the project",
        data={"secret": "actor-only"},
        criteria=(Criterion(name="useful", instruction="The patch fixes the bug."),),
    )
    evidence = Evidence(artifacts=(Artifact(name="patch", path="patch.diff"),))
    evaluator = FunctionEvaluator(
        component=ComponentDefinition(kind="patch-evaluator", version="1"),
        function=inspect,
    )

    judgments = _evaluate(case, evidence, (evaluator,), tmp_path)

    assert judgments[0].passed is True
    assert seen[0].artifacts[0].path.read_text(encoding="utf-8") == "a useful patch"
    assert not hasattr(seen[0], "data")
    assert not hasattr(seen[0], "checks")


def test_campaign_rejects_an_identifier_that_trial_receipts_cannot_use() -> None:
    async def direct(_case: Case, _trial: Trial, _trace: TraceRecorder) -> Evidence:
        return Evidence(checks=(NativeCheck(name="done", passed=True, reason="done"),))

    with pytest.raises(ValueError, match="invalid campaign identifier"):
        Campaign(
            name="-invalid",
            revision="1",
            cases=(Case(name="case", task="run", required_checks=("done",)),),
            variants=(
                Variant(
                    name="direct",
                    harness=FunctionHarness(
                        component=ComponentDefinition(kind="direct", version="1"),
                        function=direct,
                    ),
                ),
            ),
        )


def _evaluate(
    case: Case,
    evidence: Evidence,
    evaluators: tuple[FunctionEvaluator, ...],
    directory: Path,
) -> tuple[Judgment, ...]:
    trace = LocalTrace(
        TrialKey(
            campaign="evaluation-test",
            revision="1",
            case=case.name,
            variant="only",
            sample=1,
        ),
        1,
        directory / "events.jsonl",
    )
    try:
        return asyncio.run(evaluate(case, evidence, evaluators, directory, trace))
    finally:
        trace.close()
