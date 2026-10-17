from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic_ai import ModelSettings
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_multiturn_evals import (
    Campaign,
    Case,
    ComponentDefinition,
    Failed,
    Message,
    TraceRecorder,
    TrialResult,
    Variant,
    run,
)
from pydantic_multiturn_evals.harnesses import Accept, Continue, ConversationHarness
from pydantic_multiturn_evals.providers import (
    CODING_ZAI_BASE_URL,
    BoundModel,
    ModelSpec,
    PydanticTarget,
    ZAIProviderSpec,
    bind_model,
)


def test_zai_binding_uses_the_explicit_coding_route() -> None:
    binding = bind_model(
        ModelSpec(name="glm-5.3-flash", provider=ZAIProviderSpec(endpoint="coding")),
        environment={"ZAI_API_KEY": "test-key"},
    )

    assert binding.model.base_url is not None
    assert binding.model.base_url.rstrip("/") == CODING_ZAI_BASE_URL


def test_model_binding_names_the_missing_credential() -> None:
    with pytest.raises(ValueError, match="ZAI_API_KEY is not set"):
        bind_model(
            ModelSpec(name="glm-5.3-flash", provider=ZAIProviderSpec()),
            environment={},
        )


class _OneMessageActor:
    component = ComponentDefinition(kind="one-message-actor", version="1")

    async def next_turn(
        self,
        case: Case,
        messages: tuple[Message, ...],
        turn: int,
        trace: TraceRecorder,
    ) -> Continue | Accept:
        del case, messages, trace
        if turn == 0:
            return Continue(message=Message(role="user", content="hello"), reason="start")
        return Accept(reason="done")


def test_failed_pydantic_request_survives_in_the_trial_receipt(tmp_path: Path) -> None:
    async def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise RuntimeError("provider unavailable")

    target = PydanticTarget(
        BoundModel(
            model=FunctionModel(fail, model_name="failing-model"),
            settings=ModelSettings(),
            provider="function",
            interface="function",
            definition=ComponentDefinition(kind="failing-model", version="1"),
        ),
        instructions="Reply to the user.",
    )
    campaign = Campaign(
        name="failed-model-call",
        revision="1",
        cases=(Case(name="case", task="Run", required_checks=("never-reached",)),),
        variants=(
            Variant(
                name="only",
                harness=ConversationHarness(target=target, actor=_OneMessageActor()),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path))

    result = report.results[0]
    assert isinstance(result.terminal, Failed)
    assert result.terminal.failure.phase == "execute"
    receipt_path = next(tmp_path.glob("trials/**/result.json"))
    saved = TrialResult.model_validate_json(receipt_path.read_text(encoding="utf-8"))
    assert len(saved.model_calls) == 1
    call = saved.model_calls[0]
    assert call.role == "target"
    assert call.status == "failed"
    assert call.error_type == "RuntimeError"
    assert call.error_message == "provider unavailable"
