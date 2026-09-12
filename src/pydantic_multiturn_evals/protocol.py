"""Strict wire messages for command target protocol version 2."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from pydantic_multiturn_evals.models import (
    AssistantTurn,
    FixtureEntry,
    SessionOutcome,
    StrictModel,
    TargetCompletion,
    Text,
    UserTurn,
)


class WireSession(StrictModel):
    comparison_id: str | None
    arm: str | None
    suite: str
    target: str
    scenario_id: str
    repeat_index: int = Field(ge=1)
    run_id: str
    target_instructions: str | None = None
    fixture: tuple[FixtureEntry, ...] = ()


class StartRequest(StrictModel):
    protocol: Literal[2] = 2
    type: Literal["start"] = "start"
    session: WireSession


class ReadyResponse(StrictModel):
    protocol: Literal[2]
    type: Literal["ready"]


class TurnRequest(StrictModel):
    protocol: Literal[2] = 2
    type: Literal["turn"] = "turn"
    id: int = Field(ge=1)
    messages: tuple[UserTurn | AssistantTurn, ...]


class ReplyResponse(StrictModel):
    protocol: Literal[2]
    type: Literal["reply"]
    id: int = Field(ge=1)
    assistant_text: Text
    session_id: str | None = None
    evidence: dict[str, JsonValue] = Field(default_factory=dict)


class FinishRequest(StrictModel):
    protocol: Literal[2] = 2
    type: Literal["finish"] = "finish"
    outcome: SessionOutcome


class FinishedResponse(StrictModel):
    protocol: Literal[2]
    type: Literal["finished"]
    completion: TargetCompletion = TargetCompletion()


class CloseRequest(StrictModel):
    protocol: Literal[2] = 2
    type: Literal["close"] = "close"
