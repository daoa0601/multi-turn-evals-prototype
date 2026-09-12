"""Pydantic AI implementations of the adaptive actor and chat target."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from pydantic import model_validator
from pydantic_ai import Agent, format_as_xml
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.usage import UsageLimits

from pydantic_multiturn_evals.model_bindings import BoundModel, bind_model
from pydantic_multiturn_evals.models import (
    AcceptDecision,
    ActorDecision,
    ActorSpec,
    AssistantTurn,
    ContinueDecision,
    ConversationView,
    PydanticAITargetSpec,
    SessionContext,
    SessionOutcome,
    StopDecision,
    StrictModel,
    TargetCompletion,
    TargetFailureEvidence,
    TargetReply,
    Text,
    UserTurn,
)
from pydantic_multiturn_evals.runner import ActorView


class ActorDecisionPayload(StrictModel):
    """Flat provider output that is converted to the tagged domain union."""

    kind: Literal["continue", "accept", "stop"]
    reason: Text
    next_user_message: Text | None = None

    @model_validator(mode="after")
    def validate_message(self) -> ActorDecisionPayload:
        if self.kind == "continue" and self.next_user_message is None:
            raise ValueError("continue requires next_user_message")
        if self.kind != "continue" and self.next_user_message is not None:
            raise ValueError("only continue may include next_user_message")
        return self

    def to_domain(self) -> ActorDecision:
        if self.kind == "continue":
            if self.next_user_message is None:  # pragma: no cover - protected by validation
                raise RuntimeError("validated continue decision has no next message")
            return ContinueDecision(
                next_user_message=self.next_user_message,
                reason=self.reason,
            )
        if self.kind == "accept":
            return AcceptDecision(reason=self.reason)
        return StopDecision(reason=self.reason)


class PydanticActor:
    def __init__(self, spec: ActorSpec, *, model_binding: BoundModel | None = None) -> None:
        binding = model_binding or bind_model(spec.model)
        self._settings = binding.settings
        self._request_limit = spec.model_request_limit
        self._agent = Agent(
            binding.model,
            output_type=ActorDecisionPayload,
            instructions=(
                "You are the simulated user in an evaluation. Read the conversation, follow the "
                "private user brief, and return exactly one structured decision. Continue with the "
                "next user message when more evidence is useful. Accept when the user's goal is "
                "satisfied. Stop only when continuing would be impossible or unsafe. "
                f"Additional instructions: {spec.instructions}"
            ),
        )

    async def decide(self, view: ActorView) -> ActorDecision:
        prompt = format_as_xml(
            {
                "scenario_id": view.scenario_id,
                "user_brief": view.brief,
                "conversation": view.exchanges,
                "remaining_target_turns": view.remaining_target_turns,
            }
        )
        result = await self._agent.run(
            prompt,
            model_settings=self._settings,
            usage_limits=UsageLimits(request_limit=self._request_limit),
            metadata={"scenario_id": view.scenario_id, "eval_role": "adaptive_actor"},
        )
        return result.output.to_domain()


class PydanticAITarget:
    kind: Literal["pydantic_ai"] = "pydantic_ai"

    def __init__(
        self,
        spec: PydanticAITargetSpec,
        *,
        model_binding: BoundModel | None = None,
    ) -> None:
        binding = model_binding or bind_model(spec.model)
        self.name = spec.name
        self.version = spec.version
        self._settings = binding.settings
        self._agent = Agent(binding.model, output_type=str, instructions=spec.instructions)

    @asynccontextmanager
    async def session(self, context: SessionContext) -> AsyncIterator[PydanticAITarget]:
        yield self

    async def reply(self, view: ConversationView) -> TargetReply:
        result = await self._agent.run(
            view.pending_user.content,
            message_history=_message_history(view),
            conversation_id=view.run_id,
            model_settings=self._settings,
            metadata={"scenario_id": view.scenario_id, "eval_role": "target"},
        )
        return TargetReply(assistant_text=result.output)

    async def finish(self, outcome: SessionOutcome) -> TargetCompletion:
        return TargetCompletion()

    def failure_evidence(self) -> tuple[TargetFailureEvidence, ...]:
        return ()


def _message_history(view: ConversationView) -> tuple[ModelMessage, ...]:
    messages: list[ModelMessage] = []
    for exchange in view.exchanges:
        messages.append(_user_message(exchange.user))
        messages.append(_assistant_message(exchange.assistant))
    return tuple(messages)


def _user_message(turn: UserTurn) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=turn.content)])


def _assistant_message(turn: AssistantTurn) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=turn.content)])
