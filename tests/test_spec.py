from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pydantic_multiturn_evals.harbor_runner import load_harbor_arm
from pydantic_multiturn_evals.model_bindings import (
    CODING_ZAI_BASE_URL,
    GENERAL_ZAI_BASE_URL,
    bind_model,
    zai_base_url,
)
from pydantic_multiturn_evals.models import (
    ActorBrief,
    AgentEnvTargetSpec,
    PydanticAITargetSpec,
    Scenario,
    SuiteSpec,
    ZAIProviderSpec,
)
from pydantic_multiturn_evals.spec import load_suite, load_target
from pydantic_multiturn_evals.targets import build_target

ROOT = Path(__file__).parents[1]


def test_example_files_are_valid() -> None:
    suite = load_suite(ROOT / "scenarios" / "support.yaml")
    target = load_target(ROOT / "targets" / "support.yaml")
    candidate = load_target(ROOT / "targets" / "support-candidate.yaml")
    command = load_target(ROOT / "targets" / "command-example.yaml")
    agentenv = load_target(ROOT / "targets" / "agentenv-example.yaml")
    agentenv_baseline = load_target(ROOT / "targets" / "agentenv-baseline.yaml")
    agentenv_candidate = load_target(ROOT / "targets" / "agentenv-candidate.yaml")
    harbor = load_harbor_arm(ROOT / "harbor" / "baseline.example.yaml")

    assert len(suite.scenarios) == 2
    assert isinstance(target.spec, PydanticAITargetSpec)
    assert isinstance(target.spec.model.provider, ZAIProviderSpec)
    assert target.spec.model.provider.endpoint_plan == "coding"
    assert candidate.spec.name == "support-candidate"
    assert command.spec.kind == "command"
    assert isinstance(agentenv.spec, AgentEnvTargetSpec)
    assert agentenv.spec.template == "pydantic-multiturn-eval-v1"
    assert build_target(agentenv).kind == "agentenv"
    assert agentenv_baseline.spec.name != agentenv_candidate.spec.name
    assert harbor.spec.kind == "harbor"
    assert harbor.spec.base_config == (ROOT / "harbor" / "base-job.baseline.example.yaml").resolve()


def test_command_target_cwd_is_resolved_from_its_yaml(tmp_path: Path) -> None:
    target_file = tmp_path / "nested" / "target.yaml"
    target_file.parent.mkdir()
    target_file.write_text(
        """\
version: 1
name: demo
kind: command
argv: [python, harness.py]
cwd: ..
inherit_env: [PATH]
""",
        encoding="utf-8",
    )

    loaded = load_target(target_file)

    assert loaded.spec.kind == "command"
    assert loaded.spec.cwd == tmp_path.resolve()
    assert loaded.source_directory == target_file.parent.resolve()


def test_duplicate_scenario_ids_are_rejected() -> None:
    repeated = Scenario(
        id="same",
        first_prompt="Hello",
        actor=ActorBrief(persona="User", goal="Get help"),
        judge_rubric="The response helps.",
    )

    with pytest.raises(ValidationError, match="scenario ids must be unique"):
        SuiteSpec(version=1, name="duplicates", scenarios=(repeated, repeated))


def test_endpoint_plan_selects_one_fixed_url() -> None:
    assert zai_base_url("general") == GENERAL_ZAI_BASE_URL
    assert zai_base_url("coding") == CODING_ZAI_BASE_URL


def test_missing_key_names_the_variable_without_exposing_a_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EVAL_TEST_ZAI_KEY", raising=False)
    suite = load_suite(ROOT / "scenarios" / "support.yaml")
    spec = suite.actor.model.model_copy(
        update={"provider": ZAIProviderSpec(api_key_env="EVAL_TEST_ZAI_KEY")}
    )

    with pytest.raises(ValueError, match="EVAL_TEST_ZAI_KEY is not set"):
        bind_model(spec)
