from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from pydantic_multiturn_evals import experiment_runner
from pydantic_multiturn_evals.experiment import compile_experiment
from pydantic_multiturn_evals.experiment_runner import (
    ExperimentCaseResult,
    execute_experiment,
    load_saved_experiment,
    preflight_experiment,
    save_new_experiment,
    summarize_experiment,
)
from tests.helpers.model_binding import fake_model_binding

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
    command_arm = next(arm for arm in saved_plan["arms"] if arm["target"]["kind"] == "command")
    frozen_cwd = Path(command_arm["target"]["cwd"])
    assert frozen_cwd.is_relative_to(output)
    assert (frozen_cwd / "examples" / "glm_jsonl_harness.py").exists()
    assert summary.pending == experiment.reservation.case_count
    assert saved_summary["pending"] == experiment.reservation.case_count


def test_execution_failure_is_terminal_across_resume(tmp_path: Path) -> None:
    experiment = plan()
    output = tmp_path / "run"
    save_new_experiment(experiment, output)
    calls = 0

    async def broken_executor(plan, arm, case, directory):
        nonlocal calls
        calls += 1
        raise RuntimeError("provider unavailable")

    first = asyncio.run(execute_experiment(experiment, output, executor=broken_executor))
    second = asyncio.run(execute_experiment(experiment, output, executor=broken_executor))

    assert first.execution_failed == experiment.reservation.case_count
    assert second.execution_failed == first.execution_failed
    assert calls == experiment.reservation.case_count


def test_run_deadline_marks_active_cases_interrupted(tmp_path: Path) -> None:
    experiment = plan()
    experiment = experiment.model_copy(
        update={"bounds": experiment.bounds.model_copy(update={"run_seconds": 0.01})}
    )
    output = tmp_path / "run"
    save_new_experiment(experiment, output)

    async def slow_executor(plan, arm, case, directory):
        await asyncio.sleep(1)
        raise AssertionError("deadline did not cancel the case")

    summary = asyncio.run(execute_experiment(experiment, output, executor=slow_executor))

    assert summary.interrupted == experiment.bounds.max_concurrency
    assert (output / "timeout.json").exists()


def test_default_executor_runs_target_actor_judge_and_observer_from_the_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_plan = plan()
    arm = next(item for item in full_plan.arms if item.name == "harness")
    arm = arm.model_copy(update={"cases": (arm.cases[0],)})
    experiment = full_plan.model_copy(update={"arms": (arm,), "comparisons": ()})
    bindings = iter(
        (
            fake_model_binding(
                TestModel(
                    custom_output_args={
                        "kind": "accept",
                        "reason": "The answer is enough.",
                        "next_user_message": None,
                    }
                )
            ),
            fake_model_binding(
                TestModel(custom_output_args={"reason": "Helpful.", "pass": True, "score": 0.9})
            ),
            fake_model_binding(
                TestModel(custom_output_args={"score": 0.8, "passed": True, "reason": "Progress."})
            ),
            fake_model_binding(TestModel(custom_output_text="Use the account settings page.")),
        )
    )
    monkeypatch.setattr(experiment_runner, "bind_model", lambda spec: next(bindings))
    output = tmp_path / "run"
    save_new_experiment(experiment, output)

    summary = asyncio.run(execute_experiment(experiment, output))

    assert summary.completed == 1
    assert summary.task_failed == 0
    result_path = output / "cases" / "000001-harness-cancel-account-r0001"
    receipt = ExperimentCaseResult.model_validate_json(
        (result_path / "complete.json").read_text(encoding="utf-8")
    )
    assert receipt.trajectory_status == "complete"
    assert [model.role for model in receipt.requested_models] == [
        "actor",
        "target",
        "judge",
        "observer",
    ]
    assert (
        json.loads((result_path / "trajectory.jsonl").read_text())["assessment"]["status"]
        == "complete"
    )
