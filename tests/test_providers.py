from __future__ import annotations

import asyncio

import pytest
from pydantic_ai.models.test import TestModel

import pydantic_multiturn_evals.providers as providers
from pydantic_multiturn_evals.models import (
    ActorBrief,
    ActorSpec,
    AssistantTurn,
    ConversationView,
    Exchange,
    ModelSpec,
    PydanticAITargetSpec,
    UserTurn,
    ZAIProviderSpec,
)
from pydantic_multiturn_evals.runner import ActorView


def test_pydantic_actor_returns_a_validated_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = TestModel(
        custom_output_args={
            "kind": "accept",
            "reason": "The user has a concrete next step.",
            "next_user_message": None,
        }
    )
    monkeypatch.setattr(providers, "build_model", lambda spec: fake_model)
    actor = providers.PydanticActor(ActorSpec())

    decision = asyncio.run(
        actor.decide(
            ActorView(
                scenario_id="help",
                brief=ActorBrief(persona="User", goal="Get help"),
                exchanges=(
                    Exchange(
                        user=UserTurn(content="Help me."),
                        assistant=AssistantTurn(content="Follow this step."),
                    ),
                ),
                remaining_target_turns=1,
            )
        )
    )

    assert decision.kind == "accept"


def test_pydantic_target_accepts_complete_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        providers,
        "build_model",
        lambda spec: TestModel(custom_output_text="A response from the fake target."),
    )
    target = providers.PydanticAITarget(
        PydanticAITargetSpec(version=1, kind="pydantic_ai", instructions="Help the user.")
    )
    view = ConversationView(
        run_id="run-1",
        scenario_id="help",
        exchanges=(
            Exchange(
                user=UserTurn(content="First question"),
                assistant=AssistantTurn(content="First answer"),
            ),
        ),
        pending_user=UserTurn(content="Follow-up question"),
    )

    assert asyncio.run(target(view)) == "A response from the fake target."


def test_build_model_uses_the_named_environment_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_TEST_ZAI_KEY", "not-a-real-key")
    model = providers.build_model(
        ModelSpec(provider=ZAIProviderSpec(endpoint_plan="coding", api_key_env="EVAL_TEST_ZAI_KEY"))
    )

    assert model.model_name == "glm-5.3-flash"


def test_unknown_endpoint_plan_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported Z.AI endpoint plan"):
        providers.zai_base_url("unknown")
