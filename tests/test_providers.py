from __future__ import annotations

import asyncio

from pydantic_ai.models.test import TestModel

import pydantic_multiturn_evals.providers as providers
from pydantic_multiturn_evals.models import (
    ActorBrief,
    ActorSpec,
    AssistantTurn,
    CaseKey,
    ConversationView,
    Exchange,
    PydanticAITargetSpec,
    SessionContext,
    UserTurn,
)
from pydantic_multiturn_evals.runner import ActorView
from tests.helpers.model_binding import fake_model_binding


def test_pydantic_actor_returns_a_validated_decision() -> None:
    fake_model = TestModel(
        custom_output_args={
            "kind": "accept",
            "reason": "The user has a concrete next step.",
            "next_user_message": None,
        }
    )
    actor = providers.PydanticActor(ActorSpec(), model_binding=fake_model_binding(fake_model))

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


def test_pydantic_target_accepts_complete_history() -> None:
    fake_model = TestModel(custom_output_text="A response from the fake target.")
    target = providers.PydanticAITarget(
        PydanticAITargetSpec(
            version=1,
            name="test-target",
            kind="pydantic_ai",
            instructions="Help the user.",
        ),
        model_binding=fake_model_binding(fake_model),
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

    async def exercise() -> str:
        context = SessionContext(
            suite_name="test-suite",
            target_name="test-target",
            target_kind="pydantic_ai",
            target_version=1,
            key=CaseKey(scenario_id="help", repeat_index=1),
            run_id="run-1",
        )
        async with target.session(context) as session:
            return (await session.reply(view)).assistant_text

    assert asyncio.run(exercise()) == "A response from the fake target."
