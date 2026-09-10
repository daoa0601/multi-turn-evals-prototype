from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pydantic_multiturn_evals.models import (
    ActorBrief,
    Scenario,
    SuiteSpec,
    ZAIProviderSpec,
)
from pydantic_multiturn_evals.providers import (
    CODING_ZAI_BASE_URL,
    GENERAL_ZAI_BASE_URL,
    build_model,
    zai_base_url,
)
from pydantic_multiturn_evals.spec import load_suite, load_target

ROOT = Path(__file__).parents[1]


def test_example_files_are_valid() -> None:
    suite = load_suite(ROOT / "scenarios" / "support.yaml")
    target = load_target(ROOT / "targets" / "support.yaml")

    assert len(suite.scenarios) == 2
    assert target.model.provider.endpoint_plan == "coding"


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
        build_model(spec)
