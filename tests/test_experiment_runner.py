from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from pydantic_multiturn_evals.experiment import compile_experiment
from pydantic_multiturn_evals.experiment_runner import (
    ExperimentCaseResult,
    execute_experiment,
    load_saved_experiment,
    preflight_experiment,
    save_new_experiment,
    summarize_experiment,
)

ROOT = Path(__file__).parents[1]


def plan():
    return compile_experiment(ROOT / "experiments" / "support-ab.yaml")


def test_preflight_reports_every_missing_provider_credential_before_running() -> None:
    with pytest.raises(ValueError) as caught:
        preflight_experiment(plan(), environment={})

    message = str(caught.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "DEEPSEEK_API_KEY" in message
    assert "OPENAI_API_KEY" in message
    assert "ZAI_API_KEY" in message


def test_completed_cases_resume_without_replay_and_comparisons_pair_corpus_keys(
    tmp_path: Path,
) -> None:
    experiment = plan()
    output = tmp_path / "run"
    save_new_experiment(experiment, output)
    calls: list[str] = []

    async def fake_executor(plan, arm, case, directory):
        calls.append(case.key.model_dump_json())
        score = 0.9 if arm.name == "target-prompt" else 0.7
        return ExperimentCaseResult(
            key=case.key,
            passed=True,
            score=score,
            run_id=f"run-{len(calls)}",
            duration_seconds=0.01,
        )

    first = asyncio.run(execute_experiment(experiment, output, executor=fake_executor))
    restored = load_saved_experiment(output)
    second = asyncio.run(execute_experiment(restored, output, executor=fake_executor))

    assert first.completed == experiment.reservation.case_count
    assert second.completed == first.completed
    assert len(calls) == experiment.reservation.case_count
    comparison = next(item for item in first.comparisons if item.name == "concise-vs-current")
    assert comparison.complete is True
    assert comparison.mean_score_delta_candidate_minus_baseline == pytest.approx(0.2)
    assert all(pair.corpus == pair.baseline.key.corpus for pair in comparison.pairs)


def test_previous_nonterminal_receipt_is_marked_interrupted_not_replayed(
    tmp_path: Path,
) -> None:
    experiment = plan()
    output = tmp_path / "run"
    save_new_experiment(experiment, output)
    case = experiment.arms[0].cases[0]
    case_directory = output / "cases" / "000001-baseline-cancel-account-r0001"
    case_directory.mkdir(parents=True)
    (case_directory / "started.json").write_text("{}", encoding="utf-8")
    calls = 0

    async def fake_executor(plan, arm, case, directory):
        nonlocal calls
        calls += 1
        return ExperimentCaseResult(
            key=case.key,
            passed=True,
            score=1,
            duration_seconds=0.01,
        )

    asyncio.run(execute_experiment(experiment, output, executor=fake_executor))
    summary = summarize_experiment(experiment, output)

    assert case.key.arm == "baseline"
    assert summary.interrupted == 1
    assert calls == experiment.reservation.case_count - 1
    assert (case_directory / "interrupted.json").exists()


def test_new_plan_refuses_an_existing_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "user-file.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="output directory is not empty"):
        save_new_experiment(plan(), output)

    assert (output / "user-file.txt").read_text(encoding="utf-8") == "keep"


def test_plan_and_summary_are_readable_json_artifacts(tmp_path: Path) -> None:
    experiment = plan()
    output = tmp_path / "run"
    save_new_experiment(experiment, output)

    summary = summarize_experiment(experiment, output)

    saved_plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    saved_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved_plan["experiment_name"] == "support-ab"
    assert summary.pending == experiment.reservation.case_count
    assert saved_summary["pending"] == experiment.reservation.case_count
