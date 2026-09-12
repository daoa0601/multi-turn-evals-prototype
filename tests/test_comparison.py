from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from pydantic_ai.models.test import TestModel

from pydantic_multiturn_evals.comparison import compare_suite
from pydantic_multiturn_evals.models import (
    AcceptDecision,
    ActorBrief,
    CommandTargetSpec,
    GatePolicy,
    Scenario,
    SuiteSpec,
)
from pydantic_multiturn_evals.runner import ActorView
from pydantic_multiturn_evals.spec import LoadedTarget
from tests.helpers.model_binding import fake_model_binding

ROOT = Path(__file__).parents[1]
HARNESS = ROOT / "tests" / "helpers" / "command_harness.py"


class AcceptingActor:
    async def decide(self, view: ActorView) -> AcceptDecision:
        return AcceptDecision(reason=f"Observed {len(view.exchanges)} exchange.")


def loaded(name: str, variant: str) -> LoadedTarget:
    return LoadedTarget(
        spec=CommandTargetSpec(
            version=1,
            name=name,
            kind="command",
            argv=(sys.executable, str(HARNESS), name),
            cwd=ROOT,
            inherit_env=("PATH", variant),
        ),
        source_directory=ROOT,
    )


def suite() -> SuiteSpec:
    return SuiteSpec(
        version=1,
        name="comparison-suite",
        gate=GatePolicy(minimum_case_pass_rate=1, minimum_mean_score=0.7),
        scenarios=(
            Scenario(
                id="help",
                first_prompt="Help me.",
                actor=ActorBrief(persona="A user", goal="Get help"),
                judge_rubric="The reply helps.",
            ),
        ),
    )


def test_comparison_runs_both_arms_and_pairs_every_repeat(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("BASELINE_VARIANT", "baseline")
    monkeypatch.setenv("CANDIDATE_VARIANT", "candidate")
    judge_outputs = iter(
        [
            {"reason": "Weak.", "pass": True, "score": 0.4},
            {"reason": "Strong.", "pass": True, "score": 0.9},
        ]
    )

    result = asyncio.run(
        compare_suite(
            suite(),
            baseline=loaded("baseline", "BASELINE_VARIANT"),
            candidate=loaded("candidate", "CANDIDATE_VARIANT"),
            actor_factory=lambda _suite: AcceptingActor(),
            judge_binding_factory=lambda _suite: fake_model_binding(
                TestModel(custom_output_args=next(judge_outputs))
            ),
            repeat=2,
            progress=False,
        )
    )
    result.write_artifacts(tmp_path)

    assert result.baseline.result.gate.passed is False
    assert result.candidate.result.gate.passed is True
    assert (
        result.baseline.result.report.cases[0]
        .output.transcript.exchanges[0]
        .assistant.content.startswith("baseline")
    )
    assert (
        result.candidate.result.report.cases[0]
        .output.transcript.exchanges[0]
        .assistant.content.startswith("candidate")
    )
    assert [pair.key.repeat_index for pair in result.pairs] == [1, 2]
    assert [pair.score_delta_candidate_minus_baseline for pair in result.pairs] == [0.5, 0.5]
    assert (tmp_path / "baseline" / "gate.json").exists()
    assert (tmp_path / "candidate" / "gate.json").exists()
    comparison = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["baseline"]["target_name"] == "baseline"
    assert comparison["candidate"]["target_name"] == "candidate"


def test_comparison_rejects_the_same_target_name() -> None:
    target = loaded("same", "SAME_VARIANT")

    try:
        asyncio.run(compare_suite(suite(), baseline=target, candidate=target, progress=False))
    except ValueError as error:
        assert "distinct target names" in str(error)
    else:
        raise AssertionError("comparison accepted duplicate target names")
