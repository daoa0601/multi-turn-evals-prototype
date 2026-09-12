from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from pydantic_multiturn_evals.experiment import (
    ArmCaseKey,
    CorpusCaseKey,
    ExperimentBounds,
    compile_experiment,
    load_experiment,
)
from pydantic_multiturn_evals.models import (
    AgentEnvTargetSpec,
    CommandTargetSpec,
    PydanticAITargetSpec,
)

ROOT = Path(__file__).parents[1]


def test_checked_in_experiment_can_ablate_each_choice_independently() -> None:
    plan = compile_experiment(load_experiment(ROOT / "experiments" / "support-ab.yaml"))
    arms = {arm.name: arm for arm in plan.arms}
    baseline = arms["baseline"]

    expected_changes = {
        "actor-model": {"actor"},
        "target-model": {"target"},
        "judge-model": {"judge"},
        "observer-model": {"observer"},
        "target-prompt": {"prompts"},
        "fixture": {"fixture"},
        "task-selection": {"tasks"},
        "harness": {"harness"},
        "execution": {"execution"},
        "limits": {"limits"},
    }
    for arm_name, changed_fields in expected_changes.items():
        arm = arms[arm_name]
        actual_changes = {
            field
            for field in type(baseline.choices).model_fields
            if getattr(baseline.choices, field) != getattr(arm.choices, field)
        }
        assert actual_changes == changed_fields

    assert isinstance(baseline.target, CommandTargetSpec)
    assert baseline.suite.actor.model.provider.kind == "zai"
    assert arms["actor-model"].suite.actor.model.provider.kind == "openai"
    assert baseline.target_model.provider.kind == "zai"
    assert arms["target-model"].target_model.provider.kind == "openai-compatible"
    assert baseline.suite.judge.model.provider.kind == "zai"
    assert arms["judge-model"].suite.judge.model.provider.kind == "anthropic"
    assert baseline.observer_model.provider.kind == "zai"
    assert arms["observer-model"].observer_model.provider.kind == "anthropic"
    assert arms["target-prompt"].prompts.target != baseline.prompts.target
    assert arms["fixture"].fixture.values != baseline.fixture.values
    assert len(arms["task-selection"].cases) == 1
    assert isinstance(arms["harness"].target, PydanticAITargetSpec)
    assert arms["execution"].execution != baseline.execution
    assert arms["limits"].suite.limits != baseline.suite.limits


def test_matrix_expansion_is_deterministic_and_stops_above_the_arm_bound(tmp_path: Path) -> None:
    experiment_path = tmp_path / "matrix.yaml"
    experiment = yaml.safe_load((ROOT / "experiments" / "support-ab.yaml").read_text())
    experiment["design"] = {
        "kind": "matrix",
        "axes": {
            "prompts.target": ["support-current", "support-concise"],
            "actor": ["patient-user", "openai-user"],
        },
        "comparisons": [],
    }
    experiment["corpus"] = str((ROOT / "corpora" / "support.yaml").resolve())
    experiment["catalog"] = str((ROOT / "components" / "support.yaml").resolve())
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")

    first = compile_experiment(load_experiment(experiment_path))
    second = compile_experiment(load_experiment(experiment_path))

    assert [arm.name for arm in first.arms] == [
        "actor-patient-user--prompts-target-support-current",
        "actor-patient-user--prompts-target-support-concise",
        "actor-openai-user--prompts-target-support-current",
        "actor-openai-user--prompts-target-support-concise",
    ]
    assert first.model_dump_json() == second.model_dump_json()

    experiment["bounds"]["max_arms"] = 3
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[BOUND_ARMS\] design.axes"):
        compile_experiment(load_experiment(experiment_path))


def test_static_compilation_collects_reference_and_selection_issues(tmp_path: Path) -> None:
    experiment = yaml.safe_load((ROOT / "experiments" / "support-ab.yaml").read_text())
    catalog = yaml.safe_load((ROOT / "components" / "support.yaml").read_text())
    catalog["task_selectors"]["empty"] = {"kind": "ids", "ids": []}
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog), encoding="utf-8")
    experiment["corpus"] = str((ROOT / "corpora" / "support.yaml").resolve())
    experiment["catalog"] = str(catalog_path)
    experiment["defaults"]["actor"] = "missing-actor"
    experiment["defaults"]["tasks"] = "empty"
    experiment_path = tmp_path / "broken.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        compile_experiment(load_experiment(experiment_path))

    lines = str(caught.value).splitlines()[1:]
    assert lines == sorted(lines, key=lambda line: line.split("] ", 1)[1])
    assert any("[SELECTION_EMPTY] catalog.task_selectors.empty" in line for line in lines)
    assert any("[REF_UNKNOWN] defaults.actor" in line for line in lines)


def test_comparison_pairs_exact_corpus_keys() -> None:
    plan = compile_experiment(load_experiment(ROOT / "experiments" / "support-ab.yaml"))
    comparison = plan.comparisons[0]

    assert comparison.name == "concise-vs-current"
    assert comparison.pairs
    assert all(pair.baseline.corpus == pair.candidate.corpus for pair in comparison.pairs)
    assert isinstance(comparison.pairs[0].baseline, ArmCaseKey)
    assert isinstance(comparison.pairs[0].baseline.corpus, CorpusCaseKey)
    assert {pair.baseline.arm for pair in comparison.pairs} == {"baseline"}
    assert {pair.candidate.arm for pair in comparison.pairs} == {"target-prompt"}


def test_exact_comparison_rejects_different_task_selections(tmp_path: Path) -> None:
    experiment = yaml.safe_load((ROOT / "experiments" / "support-ab.yaml").read_text())
    experiment["corpus"] = str((ROOT / "corpora" / "support.yaml").resolve())
    experiment["catalog"] = str((ROOT / "components" / "support.yaml").resolve())
    experiment["design"]["comparisons"] = [
        {
            "name": "bad-pair",
            "baseline": "baseline",
            "candidate": "task-selection",
            "pairing": "exact",
        }
    ]
    path = tmp_path / "bad-comparison.yaml"
    path.write_text(yaml.safe_dump(experiment), encoding="utf-8")

    with pytest.raises(ValueError, match=r"\[COMPARISON_CASE_MISMATCH\]"):
        compile_experiment(load_experiment(path))


def test_command_and_agentenv_targets_are_resolved_and_embedded(tmp_path: Path) -> None:
    experiment = yaml.safe_load((ROOT / "experiments" / "support-ab.yaml").read_text())
    experiment["corpus"] = str((ROOT / "corpora" / "support.yaml").resolve())
    experiment["catalog"] = str((ROOT / "components" / "support.yaml").resolve())
    experiment["design"] = {
        "kind": "arms",
        "arms": [
            {
                "name": "command",
                "select": {
                    "harness": "command-jsonl",
                },
            },
            {
                "name": "agentenv",
                "select": {
                    "harness": "agentenv-session",
                    "execution": "agentenv-remote",
                },
            },
        ],
        "comparisons": [],
    }
    path = tmp_path / "external.yaml"
    path.write_text(yaml.safe_dump(experiment), encoding="utf-8")

    plan = compile_experiment(load_experiment(path))

    assert isinstance(plan.arms[0].target, CommandTargetSpec)
    assert plan.arms[0].target.cwd.is_absolute()
    assert isinstance(plan.arms[1].target, AgentEnvTargetSpec)
    serialized = json.loads(plan.model_dump_json())
    assert serialized["arms"][0]["target"]["argv"]
    assert serialized["arms"][1]["target"]["turn"]["argv"]


def test_static_compilation_does_not_read_provider_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("ZAI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    plan = compile_experiment(load_experiment(ROOT / "experiments" / "support-ab.yaml"))

    assert len(plan.arms) == 11
    assert plan.reservation.case_count == sum(len(arm.cases) for arm in plan.arms)


def test_case_bound_applies_after_task_selection_and_repeats(tmp_path: Path) -> None:
    experiment = yaml.safe_load((ROOT / "experiments" / "support-ab.yaml").read_text())
    experiment["corpus"] = str((ROOT / "corpora" / "support.yaml").resolve())
    experiment["catalog"] = str((ROOT / "components" / "support.yaml").resolve())
    experiment["bounds"] = ExperimentBounds(
        max_arms=20,
        max_cases=1,
        max_concurrency=1,
        max_model_requests=100,
        max_output_tokens=100_000,
        run_seconds=60,
    ).model_dump(mode="json")
    experiment_path = tmp_path / "bounded.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")

    with pytest.raises(ValueError, match=r"\[BOUND_CASES\] bounds.max_cases"):
        compile_experiment(load_experiment(experiment_path))
