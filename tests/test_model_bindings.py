from __future__ import annotations

import pytest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.models.zai import ZaiModel

from pydantic_multiturn_evals.model_bindings import bind_model
from pydantic_multiturn_evals.models import (
    AnthropicProviderSpec,
    ModelOptions,
    ModelSpec,
    OpenAICompatibleProviderSpec,
    OpenAIProviderSpec,
    ZAIProviderSpec,
)


def test_zai_coding_binding_keeps_model_and_settings_together() -> None:
    binding = bind_model(
        ModelSpec(
            name="glm-5.3-flash",
            provider=ZAIProviderSpec(
                kind="zai",
                endpoint_plan="coding",
                api_key_env="TEST_ZAI_API_KEY",
            ),
            options=ModelOptions(thinking="low", max_tokens=700),
        ),
        environment={"TEST_ZAI_API_KEY": "not-a-real-key"},
    )

    assert isinstance(binding.model, ZaiModel)
    assert str(binding.model.base_url).rstrip("/") == "https://api.z.ai/api/coding/paas/v4"
    assert binding.settings.get("thinking") == "low"
    assert "extra_body" not in binding.settings
    assert binding.identity.provider == "zai"
    assert binding.required_environment == ("TEST_ZAI_API_KEY",)


def test_openai_binding_does_not_receive_zai_request_fields() -> None:
    binding = bind_model(
        ModelSpec(
            name="gpt-5.4",
            provider=OpenAIProviderSpec(
                kind="openai",
                interface="responses",
                api_key_env="TEST_OPENAI_API_KEY",
            ),
            options=ModelOptions(temperature=None, top_p=None, thinking="high"),
        ),
        environment={"TEST_OPENAI_API_KEY": "not-a-real-key"},
    )

    assert isinstance(binding.model, OpenAIResponsesModel)
    assert binding.settings.get("thinking") == "high"
    assert "temperature" not in binding.settings
    assert "top_p" not in binding.settings
    assert "extra_body" not in binding.settings
    assert binding.identity.provider == "openai"


def test_anthropic_binding_uses_its_explicit_credential() -> None:
    binding = bind_model(
        ModelSpec(
            name="claude-sonnet-4-5",
            provider=AnthropicProviderSpec(
                kind="anthropic",
                api_key_env="TEST_ANTHROPIC_API_KEY",
            ),
            options=ModelOptions(temperature=0.2, top_p=None, thinking=False),
        ),
        environment={"TEST_ANTHROPIC_API_KEY": "not-a-real-key"},
    )

    assert isinstance(binding.model, AnthropicModel)
    assert binding.settings.get("temperature") == 0.2
    assert binding.settings.get("thinking") is False
    assert "top_p" not in binding.settings
    assert binding.identity.provider == "anthropic"


def test_openai_compatible_binding_can_target_a_keyless_local_server() -> None:
    binding = bind_model(
        ModelSpec(
            name="local-model",
            provider=OpenAICompatibleProviderSpec.model_validate(
                {
                    "kind": "openai-compatible",
                    "interface": "chat",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "api_key_env": None,
                }
            ),
        ),
        environment={},
    )

    assert isinstance(binding.model, OpenAIChatModel)
    assert str(binding.model.base_url).rstrip("/") == "http://127.0.0.1:8000/v1"
    assert binding.required_environment == ()
    assert binding.identity.provider == "openai-compatible"


def test_binding_fails_before_provider_io_when_credential_is_missing() -> None:
    spec = ModelSpec(
        name="gpt-5.4",
        provider=OpenAIProviderSpec(
            kind="openai",
            api_key_env="MISSING_OPENAI_API_KEY",
        ),
    )

    with pytest.raises(ValueError, match="MISSING_OPENAI_API_KEY is not set"):
        bind_model(spec, environment={})
