"""Resolve one authored model choice into matched Pydantic AI runtime objects."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from pydantic_ai import ModelSettings
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelName
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIModelName,
    OpenAIResponsesModel,
)
from pydantic_ai.models.zai import ZaiModel, ZaiModelName
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from pydantic_multiturn_evals.models import (
    AnthropicProviderSpec,
    Identifier,
    ModelOptions,
    ModelSpec,
    OpenAICompatibleProviderSpec,
    OpenAIProviderSpec,
    StrictModel,
    Text,
    ZAIProviderSpec,
)

GENERAL_ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"
CODING_ZAI_BASE_URL = "https://api.z.ai/api/coding/paas/v4"


class ModelIdentity(StrictModel):
    provider: Identifier
    requested_model: Text
    interface: Identifier


@dataclass(frozen=True, slots=True)
class BoundModel:
    model: Model
    settings: ModelSettings
    identity: ModelIdentity
    required_environment: tuple[str, ...]


def bind_model(
    spec: ModelSpec,
    *,
    environment: Mapping[str, str] | None = None,
) -> BoundModel:
    """Build a model and only the settings accepted by its authored provider route."""

    values = os.environ if environment is None else environment
    provider = spec.provider
    settings = _model_settings(spec.options)

    if isinstance(provider, ZAIProviderSpec):
        api_key = _required_environment(values, provider.api_key_env)
        base_url = zai_base_url(provider.endpoint_plan)
        model_provider = OpenAIProvider(base_url=base_url, api_key=api_key)
        model = ZaiModel(cast(ZaiModelName, spec.name), provider=model_provider)
        interface = "chat"
    elif isinstance(provider, OpenAIProviderSpec):
        api_key = _required_environment(values, provider.api_key_env)
        model_provider = OpenAIProvider(api_key=api_key)
        model = _openai_model(spec.name, provider.interface, model_provider)
        interface = provider.interface
    elif isinstance(provider, AnthropicProviderSpec):
        api_key = _required_environment(values, provider.api_key_env)
        model = AnthropicModel(
            cast(AnthropicModelName, spec.name),
            provider=AnthropicProvider(api_key=api_key),
        )
        interface = "messages"
    elif isinstance(provider, OpenAICompatibleProviderSpec):
        api_key = (
            _required_environment(values, provider.api_key_env)
            if provider.api_key_env is not None
            else "local-server"
        )
        model_provider = OpenAIProvider(base_url=str(provider.base_url), api_key=api_key)
        model = _openai_model(spec.name, provider.interface, model_provider)
        interface = provider.interface
    else:  # pragma: no cover - ProviderSpec is exhaustive
        raise AssertionError(f"unhandled provider: {provider}")

    return BoundModel(
        model=model,
        settings=settings,
        identity=ModelIdentity(
            provider=provider.kind,
            requested_model=spec.name,
            interface=interface,
        ),
        required_environment=required_environment(spec),
    )


def required_environment(spec: ModelSpec) -> tuple[str, ...]:
    name = spec.provider.api_key_env
    return () if name is None else (name,)


def zai_base_url(endpoint_plan: str) -> str:
    if endpoint_plan == "general":
        return GENERAL_ZAI_BASE_URL
    if endpoint_plan == "coding":
        return CODING_ZAI_BASE_URL
    raise ValueError(f"unsupported Z.AI endpoint plan: {endpoint_plan}")


def _model_settings(options: ModelOptions) -> ModelSettings:
    settings = ModelSettings(
        max_tokens=options.max_tokens,
        timeout=options.timeout_seconds,
    )
    if options.temperature is not None:
        settings["temperature"] = options.temperature
    if options.top_p is not None:
        settings["top_p"] = options.top_p
    if options.thinking is not None:
        settings["thinking"] = options.thinking
    return settings


def _openai_model(
    name: str,
    interface: str,
    provider: OpenAIProvider,
) -> Model:
    model_name = cast(OpenAIModelName, name)
    if interface == "responses":
        return OpenAIResponsesModel(model_name, provider=provider)
    if interface == "chat":
        return OpenAIChatModel(model_name, provider=provider)
    raise ValueError(f"unsupported OpenAI interface: {interface}")


def _required_environment(values: Mapping[str, str], name: str) -> str:
    value = values.get(name)
    if not value:
        raise ValueError(f"{name} is not set")
    return value
