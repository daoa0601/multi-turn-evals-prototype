"""Pydantic AI components and explicit provider binding."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias, TypeVar

from pydantic import AnyHttpUrl, Field, JsonValue, TypeAdapter, model_validator
from pydantic_ai import Agent, ModelSettings, format_as_xml
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.models.zai import ZaiModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.run import AgentRunResult
from pydantic_ai.usage import RunUsage, UsageLimits

from pydantic_multiturn_evals.harnesses.conversation import (
    Accept,
    ActorDecision,
    Continue,
    TargetCompletion,
    TargetSession,
)
from pydantic_multiturn_evals.model import (
    Case,
    ComponentDefinition,
    EvaluationInput,
    Judgment,
    Message,
    ModelCall,
    ModelCallFact,
    ModelUsage,
    StrictModel,
    TraceRecorder,
    Trial,
)

OutputT = TypeVar("OutputT")

GENERAL_ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"
CODING_ZAI_BASE_URL = "https://api.z.ai/api/coding/paas/v4"


class ZAIProviderSpec(StrictModel):
    kind: Literal["zai"] = "zai"
    endpoint: Literal["general", "coding"] = "coding"
    api_key_env: str = "ZAI_API_KEY"


class OpenAIProviderSpec(StrictModel):
    kind: Literal["openai"] = "openai"
    interface: Literal["responses", "chat"] = "responses"
    api_key_env: str = "OPENAI_API_KEY"


class AnthropicProviderSpec(StrictModel):
    kind: Literal["anthropic"] = "anthropic"
    api_key_env: str = "ANTHROPIC_API_KEY"


class OpenAICompatibleProviderSpec(StrictModel):
    kind: Literal["openai-compatible"] = "openai-compatible"
    interface: Literal["responses", "chat"] = "chat"
    base_url: AnyHttpUrl
    api_key_env: str | None = None


ProviderSpec: TypeAlias = Annotated[
    ZAIProviderSpec | OpenAIProviderSpec | AnthropicProviderSpec | OpenAICompatibleProviderSpec,
    Field(discriminator="kind"),
]


class ModelOptions(StrictModel):
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int = Field(default=2048, ge=1)
    timeout_seconds: float = Field(default=90, gt=0)


class ModelSpec(StrictModel):
    name: str
    provider: ProviderSpec
    options: ModelOptions = ModelOptions()


@dataclass(frozen=True, slots=True)
class BoundModel:
    model: Model
    settings: ModelSettings
    provider: str
    interface: str
    definition: ComponentDefinition


def bind_model(
    spec: ModelSpec,
    *,
    environment: Mapping[str, str] | None = None,
) -> BoundModel:
    values = os.environ if environment is None else environment
    provider = spec.provider
    settings = _settings(spec.options)
    if isinstance(provider, ZAIProviderSpec):
        api_key = _required(values, provider.api_key_env)
        base_url = CODING_ZAI_BASE_URL if provider.endpoint == "coding" else GENERAL_ZAI_BASE_URL
        model: Model = ZaiModel(
            spec.name,
            provider=OpenAIProvider(base_url=base_url, api_key=api_key),
        )
        interface = "chat"
    elif isinstance(provider, OpenAIProviderSpec):
        api_key = _required(values, provider.api_key_env)
        model = _openai_model(
            spec.name,
            provider.interface,
            OpenAIProvider(api_key=api_key),
        )
        interface = provider.interface
    elif isinstance(provider, AnthropicProviderSpec):
        api_key = _required(values, provider.api_key_env)
        model = AnthropicModel(spec.name, provider=AnthropicProvider(api_key=api_key))
        interface = "messages"
    else:
        api_key = _required(values, provider.api_key_env) if provider.api_key_env else "local"
        model = _openai_model(
            spec.name,
            provider.interface,
            OpenAIProvider(base_url=str(provider.base_url), api_key=api_key),
        )
        interface = provider.interface
    return BoundModel(
        model=model,
        settings=settings,
        provider=provider.kind,
        interface=interface,
        definition=ComponentDefinition(
            kind="provider-model",
            version="1",
            config=spec.model_dump(mode="json"),
        ),
    )


class _ActorPayload(StrictModel):
    kind: Literal["continue", "accept"]
    reason: str
    message: str | None = None

    @model_validator(mode="after")
    def match_message_to_kind(self) -> _ActorPayload:
        if self.kind == "continue" and not self.message:
            raise ValueError("continue requires a message")
        if self.kind == "accept" and self.message is not None:
            raise ValueError("accept cannot include a message")
        return self


class PydanticActor:
    def __init__(
        self,
        model: ModelSpec | BoundModel,
        *,
        instructions: str = "Stay in character.",
    ) -> None:
        binding = _binding(model)
        self.component = ComponentDefinition(
            kind="pydantic-actor",
            version="1",
            config={
                "model": binding.definition.model_dump(mode="json"),
                "instructions": instructions,
            },
        )
        self._settings = binding.settings
        self._binding = binding
        self._agent = Agent(
            binding.model,
            output_type=_ActorPayload,
            instructions=(
                "Act as the user in an evaluation. Continue with one natural user message when "
                "more interaction is useful. Accept when the task is resolved. " + instructions
            ),
        )

    async def next_turn(
        self,
        case: Case,
        messages: tuple[Message, ...],
        turn: int,
        trace: TraceRecorder,
    ) -> ActorDecision:
        result = await _recorded_run(
            binding=self._binding,
            trace=trace,
            role="actor",
            operation=f"decision-{turn + 1}",
            call=lambda: self._agent.run(
                format_as_xml(
                    {
                        "task": case.task,
                        "private_case_data": case.data,
                        "messages": [message.model_dump(mode="json") for message in messages],
                        "turn": turn,
                        "turn_budget": case.budget.turns,
                    }
                ),
                model_settings=self._settings,
                usage_limits=UsageLimits(request_limit=2),
            ),
        )
        if result.output.kind == "accept":
            return Accept(reason=result.output.reason)
        if result.output.message is None:
            raise RuntimeError("validated actor continuation has no message")
        return Continue(
            message=Message(role="user", content=result.output.message, actor="simulated-user"),
            reason=result.output.reason,
        )


class _JudgmentBatch(StrictModel):
    judgments: tuple[Judgment, ...]


class PydanticJudge:
    def __init__(self, model: ModelSpec | BoundModel) -> None:
        binding = _binding(model)
        self.component = ComponentDefinition(
            kind="pydantic-judge",
            version="1",
            config={"model": binding.definition.model_dump(mode="json")},
        )
        self._settings = binding.settings
        self._binding = binding
        self._agent = Agent(
            binding.model,
            output_type=_JudgmentBatch,
            instructions=(
                "Judge each supplied criterion from only the public outcome. Return exactly one "
                "judgment per criterion. Do not infer private state or external side effects."
            ),
        )

    async def evaluate(
        self,
        value: EvaluationInput,
        trace: TraceRecorder,
    ) -> tuple[Judgment, ...]:
        if not value.criteria:
            return ()
        result = await _recorded_run(
            binding=self._binding,
            trace=trace,
            role="judge",
            operation="criteria",
            call=lambda: self._agent.run(
                format_as_xml(
                    {
                        "task": value.task,
                        "criteria": [
                            criterion.model_dump(mode="json") for criterion in value.criteria
                        ],
                        "public_outcome": value.outcome.model_dump(mode="json"),
                    }
                ),
                model_settings=self._settings,
                usage_limits=UsageLimits(request_limit=2),
            ),
        )
        return result.output.judgments


class PydanticTarget:
    def __init__(self, model: ModelSpec | BoundModel, *, instructions: str) -> None:
        binding = _binding(model)
        self.component = ComponentDefinition(
            kind="pydantic-target",
            version="1",
            config={
                "model": binding.definition.model_dump(mode="json"),
                "instructions": instructions,
            },
        )
        self._agent = Agent(binding.model, output_type=str, instructions=instructions)
        self._settings = binding.settings
        self._binding = binding

    async def open(
        self,
        trial: Trial,
        trace: TraceRecorder,
    ) -> TargetSession:
        del trial
        return _PydanticTargetSession(self._agent, self._binding, trace)


class _PydanticTargetSession:
    def __init__(
        self,
        agent: Agent[None, str],
        binding: BoundModel,
        trace: TraceRecorder,
    ) -> None:
        self._agent = agent
        self._binding = binding
        self._trace = trace
        self._messages: list[Message] = []
        self._turn = 0

    async def send(self, message: Message) -> tuple[Message, ...]:
        if message.role != "user":
            raise ValueError("Pydantic target accepts user messages")
        self._turn += 1
        result = await _recorded_run(
            binding=self._binding,
            trace=self._trace,
            role="target",
            operation=f"reply-{self._turn}",
            call=lambda: self._agent.run(
                message.content,
                message_history=_model_history(self._messages),
                model_settings=self._binding.settings,
            ),
        )
        reply = Message(role="assistant", content=result.output, actor="target")
        self._messages.extend((message, reply))
        return (reply,)

    async def finish(self, messages: tuple[Message, ...]) -> TargetCompletion:
        del messages
        return TargetCompletion()

    async def close(self) -> None:
        return None


def _settings(options: ModelOptions) -> ModelSettings:
    settings = ModelSettings(max_tokens=options.max_tokens, timeout=options.timeout_seconds)
    if options.temperature is not None:
        settings["temperature"] = options.temperature
    if options.top_p is not None:
        settings["top_p"] = options.top_p
    return settings


def _binding(model: ModelSpec | BoundModel) -> BoundModel:
    return bind_model(model) if isinstance(model, ModelSpec) else model


async def _recorded_run(
    *,
    binding: BoundModel,
    trace: TraceRecorder,
    role: str,
    operation: str,
    call: Callable[[], Awaitable[AgentRunResult[OutputT]]],
) -> AgentRunResult[OutputT]:
    started = time.monotonic()
    try:
        result = await call()
    except asyncio.CancelledError:
        trace.record(
            ModelCallFact(
                call=_model_call(
                    binding,
                    role,
                    operation,
                    duration_seconds=time.monotonic() - started,
                    status="cancelled",
                )
            )
        )
        raise
    except Exception as error:
        trace.record(
            ModelCallFact(
                call=_model_call(
                    binding,
                    role,
                    operation,
                    duration_seconds=time.monotonic() - started,
                    status="failed",
                    error=error,
                )
            )
        )
        raise
    trace.record(
        ModelCallFact(
            call=_model_call(
                binding,
                role,
                operation,
                duration_seconds=time.monotonic() - started,
                status="completed",
                result=result,
            )
        )
    )
    return result


def _model_call(
    binding: BoundModel,
    role: str,
    operation: str,
    *,
    duration_seconds: float,
    status: Literal["completed", "failed", "cancelled"],
    result: AgentRunResult[OutputT] | None = None,
    error: Exception | None = None,
) -> ModelCall:
    usage = _usage(result.usage) if result is not None else None
    response = result.response if result is not None else None
    return ModelCall(
        role=role,
        operation=operation,
        requested=binding.definition,
        reported_model=response.model_name if response is not None else None,
        provider=binding.provider,
        interface=binding.interface,
        sampling=TypeAdapter(JsonValue).validate_python(dict(binding.settings)),
        duration_seconds=duration_seconds,
        status=status,
        usage=usage,
        provider_response_id=(response.provider_response_id if response is not None else None),
        error_type=type(error).__name__ if error is not None else None,
        error_message=(str(error) or repr(error))[:1000] if error is not None else None,
    )


def _usage(value: RunUsage) -> ModelUsage:
    return ModelUsage(
        requests=value.requests,
        tool_calls=value.tool_calls,
        input_tokens=value.input_tokens,
        cache_write_tokens=value.cache_write_tokens,
        cache_read_tokens=value.cache_read_tokens,
        output_tokens=value.output_tokens,
        input_audio_tokens=value.input_audio_tokens,
        cache_audio_read_tokens=value.cache_audio_read_tokens,
        output_audio_tokens=value.output_audio_tokens,
        cost_usd=float(value.cost) if value.cost is not None else None,
        details=value.details,
    )


def _openai_model(name: str, interface: str, provider: OpenAIProvider) -> Model:
    if interface == "responses":
        return OpenAIResponsesModel(name, provider=provider)
    return OpenAIChatModel(name, provider=provider)


def _required(values: Mapping[str, str], name: str | None) -> str:
    if name is None:
        return "local"
    value = values.get(name)
    if not value:
        raise ValueError(f"{name} is not set")
    return value


def _model_history(messages: list[Message]) -> tuple[ModelMessage, ...]:
    history: list[ModelMessage] = []
    for message in messages:
        if message.role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content=message.content)]))
        elif message.role == "assistant":
            history.append(ModelResponse(parts=[TextPart(content=message.content)]))
    return tuple(history)
