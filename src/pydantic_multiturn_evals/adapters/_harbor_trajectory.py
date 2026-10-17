"""Private ATIF trajectory import for the Harbor adapter.

Mirrors the Agent Trajectory Interchange Format models accepted by the pinned
Harbor version so a saved agent trajectory can become typed evidence without
importing Harbor itself. Only text content is imported: image sources and
references to other trajectory files are validated for shape but never read.

The importer handles exactly one segment. The adapter follows ordered
``continued_trajectory_ref`` chains and imports every segment once, so steps
flagged ``is_copied_context`` (context replayed from an earlier segment) are
skipped entirely instead of being recorded twice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from pydantic_multiturn_evals.model import (
    ComponentDefinition,
    JsonValue,
    LogFact,
    Message,
    MessageFact,
    ModelCall,
    ModelCallFact,
    ModelUsage,
    ResourceFact,
    StrictModel,
    ToolFact,
    TraceFact,
)

SUPPORTED_ATIF_VERSIONS: tuple[str, ...] = (
    "ATIF-v1.0",
    "ATIF-v1.1",
    "ATIF-v1.2",
    "ATIF-v1.3",
    "ATIF-v1.4",
    "ATIF-v1.5",
    "ATIF-v1.6",
)

_AtifVersion = Literal[
    "ATIF-v1.0",
    "ATIF-v1.1",
    "ATIF-v1.2",
    "ATIF-v1.3",
    "ATIF-v1.4",
    "ATIF-v1.5",
    "ATIF-v1.6",
]


class _AtifImageSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    path: str


class _AtifContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "image"]
    text: str | None = None
    source: _AtifImageSource | None = None

    @model_validator(mode="after")
    def require_fields_for_type(self) -> Self:
        if self.type == "text":
            if self.text is None:
                raise ValueError("'text' field is required when type='text'")
            if self.source is not None:
                raise ValueError("'source' field is not allowed when type='text'")
        else:
            if self.source is None:
                raise ValueError("'source' field is required when type='image'")
            if self.text is not None:
                raise ValueError("'text' field is not allowed when type='image'")
        return self


class _AtifToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    function_name: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class _AtifSubagentRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    trajectory_path: str | None = None
    extra: dict[str, JsonValue] | None = None


class _AtifObservationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_call_id: str | None = None
    content: str | list[_AtifContentPart] | None = None
    subagent_trajectory_ref: list[_AtifSubagentRef] | None = None


class _AtifObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[_AtifObservationResult]


class _AtifMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    extra: dict[str, JsonValue] | None = None


class _AtifFinalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_cached_tokens: int | None = None
    total_cost_usd: float | None = None
    total_steps: Annotated[int, Field(ge=0)] | None = None
    extra: dict[str, JsonValue] | None = None


class _AtifAgent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    model_name: str | None = None
    tool_definitions: list[dict[str, JsonValue]] | None = None
    extra: dict[str, JsonValue] | None = None


class _AtifStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: Annotated[int, Field(ge=1)]
    timestamp: str | None = None
    source: Literal["system", "user", "agent"]
    model_name: str | None = None
    reasoning_effort: str | float | None = None
    message: str | list[_AtifContentPart]
    reasoning_content: str | None = None
    tool_calls: list[_AtifToolCall] | None = None
    observation: _AtifObservation | None = None
    metrics: _AtifMetrics | None = None
    is_copied_context: bool | None = None
    extra: dict[str, JsonValue] | None = None

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(f"Invalid ISO 8601 timestamp: {error}") from error
        return value

    @model_validator(mode="after")
    def restrict_agent_fields(self) -> Self:
        if self.source != "agent":
            for name in (
                "model_name",
                "reasoning_effort",
                "reasoning_content",
                "tool_calls",
                "metrics",
            ):
                if getattr(self, name) is not None:
                    raise ValueError(
                        f"Field '{name}' is only applicable when source is 'agent', "
                        f"but source is '{self.source}'"
                    )
        return self


class AtifTrajectory(StrictModel):
    """One Harbor agent trajectory restricted to versions Harbor 0.1.44 accepts."""

    schema_version: _AtifVersion = "ATIF-v1.6"
    session_id: str
    agent: _AtifAgent
    steps: Annotated[list[_AtifStep], Field(min_length=1)]
    notes: str | None = None
    final_metrics: _AtifFinalMetrics | None = None
    continued_trajectory_ref: str | None = None
    extra: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def require_sequential_steps(self) -> Self:
        for index, step in enumerate(self.steps):
            if step.step_id != index + 1:
                raise ValueError(
                    f"steps[{index}].step_id: expected {index + 1} "
                    f"(sequential from 1), got {step.step_id}"
                )
        return self

    @model_validator(mode="after")
    def require_local_observation_references(self) -> Self:
        for step in self.steps:
            if step.observation is None:
                continue
            call_ids = {call.tool_call_id for call in step.tool_calls or ()}
            for result in step.observation.results:
                if result.source_call_id is not None and result.source_call_id not in call_ids:
                    raise ValueError(
                        f"Observation result references source_call_id "
                        f"'{result.source_call_id}' which is not found in "
                        f"step {step.step_id}'s tool_calls"
                    )
        return self


@dataclass(frozen=True, slots=True)
class TrajectoryReceipt:
    """Public dialogue plus typed facts imported from one saved trajectory."""

    messages: tuple[Message, ...]
    facts: tuple[TraceFact, ...]


def parse_trajectory(data: bytes) -> AtifTrajectory:
    """Validate one ATIF trajectory without following external references."""

    try:
        return AtifTrajectory.model_validate_json(data.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError("trajectory.json is not valid UTF-8") from error
    except ValidationError as error:
        raise ValueError(_validation_message(error)) from error


def import_trajectory(trajectory: AtifTrajectory, *, receipt: str) -> TrajectoryReceipt:
    """Turn one validated trajectory segment into text and typed facts.

    System steps, agent reasoning, and unreported data stay out of the import.
    Steps flagged as copied context are skipped entirely — message, tool, and
    model facts — because the adapter imports every segment of the
    continuation chain, where that step already exists as fresh provenance.
    """

    messages: list[Message] = []
    facts: list[TraceFact] = []
    call_names: dict[str, str] = {}
    skipped_images = 0
    for step in trajectory.steps:
        if step.source == "system" or step.is_copied_context:
            continue
        text, images = _text_content(step.message)
        skipped_images += images
        if text is not None:
            message = Message(
                role="user" if step.source == "user" else "assistant",
                content=text,
                actor=f"harbor-{step.source}",
            )
            messages.append(message)
            facts.append(MessageFact(message=message))
        for call in step.tool_calls or ():
            call_names[call.tool_call_id] = call.function_name
            facts.append(ToolFact(name=call.function_name, input=call.arguments))
        for result in step.observation.results if step.observation is not None else ():
            output, images = _text_content(result.content)
            skipped_images += images
            if output is None:
                continue
            facts.append(
                ToolFact(
                    name=call_names.get(result.source_call_id or "", "harbor-observation"),
                    output=output,
                )
            )
            message = Message(role="tool", content=output, actor="harbor-tool")
            messages.append(message)
            facts.append(MessageFact(message=message))
        if step.source == "agent" and step.metrics is not None:
            call = _model_call(
                step,
                metrics=step.metrics,
                agent=trajectory.agent,
                schema_version=trajectory.schema_version,
                receipt=receipt,
            )
            facts.append(ModelCallFact(call=call))
            if call.usage is None:
                facts.extend(_partial_usage_facts(step.metrics))
    if skipped_images:
        facts.append(
            LogFact(
                level="warning",
                message=(
                    f"Harbor trajectory contained {skipped_images} image content "
                    "parts that were not imported"
                ),
            )
        )
    return TrajectoryReceipt(messages=tuple(messages), facts=tuple(facts))


def _model_call(
    step: _AtifStep,
    *,
    metrics: _AtifMetrics,
    agent: _AtifAgent,
    schema_version: str,
    receipt: str,
) -> ModelCall:
    model = step.model_name or agent.model_name
    return ModelCall(
        role="target",
        operation=f"{_receipt_token(receipt)}-step-{step.step_id}",
        requested=ComponentDefinition(
            kind="harbor-agent-model",
            version="1",
            config={
                "agent": agent.name,
                "agent_version": agent.version,
                "model": model,
                "schema_version": schema_version,
                "receipt": receipt,
                "source": "imported-receipt",
            },
        ),
        reported_model=model,
        provider="harbor",
        interface="atif-trajectory",
        sampling=(
            {"reasoning_effort": step.reasoning_effort}
            if step.reasoning_effort is not None
            else None
        ),
        duration_seconds=None,
        status="completed",
        usage=_usage(metrics, tool_calls=len(step.tool_calls or ())),
    )


def _usage(metrics: _AtifMetrics, *, tool_calls: int) -> ModelUsage | None:
    """Build usage only from counts the trajectory actually reported.

    ATIF token fields are optional and unconstrained, so a missing count stays
    unknown (``usage=None``) instead of becoming a reported zero, and any
    negative count makes the metric block too untrustworthy to import.
    Arbitrary ``metrics.extra`` values are never usage counters.
    """
    input_tokens = metrics.prompt_tokens
    output_tokens = metrics.completion_tokens
    if input_tokens is None or output_tokens is None:
        return None
    if input_tokens < 0 or output_tokens < 0:
        return None
    if metrics.cached_tokens is not None and metrics.cached_tokens < 0:
        return None
    return ModelUsage(
        requests=1,
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        cache_read_tokens=metrics.cached_tokens if metrics.cached_tokens is not None else 0,
        output_tokens=output_tokens,
        cost_usd=_valid_cost(metrics.cost_usd),
    )


_PARTIAL_USAGE_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("prompt_tokens", "harbor-trajectory-prompt-tokens", "tokens"),
    ("completion_tokens", "harbor-trajectory-completion-tokens", "tokens"),
    ("cached_tokens", "harbor-trajectory-cached-tokens", "tokens"),
)


def _partial_usage_facts(metrics: _AtifMetrics) -> list[TraceFact]:
    """Keep reported values as plain facts when no trustworthy usage exists.

    Only literal values the trajectory reported are recorded, so partial
    metrics (for example a cost with unknown token counts) are preserved
    without fabricating a full ``ModelUsage`` and without deriving anything
    from aggregate ``final_metrics``.
    """
    facts: list[TraceFact] = []
    for attribute, name, unit in _PARTIAL_USAGE_FIELDS:
        count = getattr(metrics, attribute)
        if _reported(count):
            facts.append(ResourceFact(name=name, value=float(count), unit=unit))
    cost = _valid_cost(metrics.cost_usd)
    if cost is not None:
        facts.append(ResourceFact(name="harbor-trajectory-cost", value=cost, unit="usd"))
    return facts


def _reported(count: int | None) -> bool:
    return count is not None and count >= 0


def _valid_cost(cost: float | None) -> float | None:
    if cost is None or not math.isfinite(cost) or cost < 0:
        return None
    return cost


def _receipt_token(receipt: str) -> str:
    """Encode unsafe characters without merging distinct receipt paths."""
    return "receipt-" + "".join(
        character
        if character.isascii() and (character.isalnum() or character in ".-")
        else f"_{ord(character):x}_"
        for character in receipt
    )


def _text_content(content: str | list[_AtifContentPart] | None) -> tuple[str | None, int]:
    if content is None:
        return None, 0
    if isinstance(content, str):
        return (content, 0) if content.strip() else (None, 0)
    parts: list[str] = []
    images = 0
    for part in content:
        if part.type == "image":
            images += 1
        elif part.text and part.text.strip():
            parts.append(part.text)
    return ("\n".join(parts), images) if parts else (None, images)


def _validation_message(error: ValidationError) -> str:
    reasons = []
    for entry in error.errors()[:5]:
        location = ".".join(str(part) for part in entry["loc"])
        reasons.append(f"{location or 'trajectory'}: {entry['msg']}")
    return "; ".join(reasons)
