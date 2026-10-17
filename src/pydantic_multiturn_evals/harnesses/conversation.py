"""An adaptive conversation harness composed from an actor and a target."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol

from pydantic import Field

from pydantic_multiturn_evals.evaluation import HarnessError
from pydantic_multiturn_evals.model import (
    Artifact,
    Case,
    ComponentDefinition,
    Evidence,
    JsonValue,
    Message,
    MessageFact,
    NativeCheck,
    PublicOutcome,
    StrictModel,
    TraceRecorder,
    Trial,
)


class Continue(StrictModel):
    kind: Literal["continue"] = "continue"
    message: Message
    reason: str


class Accept(StrictModel):
    kind: Literal["accept"] = "accept"
    reason: str


class Stop(StrictModel):
    kind: Literal["stop"] = "stop"
    reason: str


ActorDecision = Annotated[Continue | Accept | Stop, Field(discriminator="kind")]


class TargetCompletion(StrictModel):
    output: JsonValue = None
    checks: tuple[NativeCheck, ...] = ()
    artifacts: tuple[Artifact, ...] = ()


class Actor(Protocol):
    @property
    def component(self) -> ComponentDefinition: ...

    async def next_turn(
        self,
        case: Case,
        messages: tuple[Message, ...],
        turn: int,
        trace: TraceRecorder,
    ) -> ActorDecision: ...


class TargetSession(Protocol):
    async def send(self, message: Message) -> tuple[Message, ...]: ...

    async def finish(self, messages: tuple[Message, ...]) -> TargetCompletion: ...

    async def close(self) -> None: ...


class Target(Protocol):
    @property
    def component(self) -> ComponentDefinition: ...

    async def open(self, trial: Trial, trace: TraceRecorder) -> TargetSession: ...


@dataclass(frozen=True, slots=True)
class ConversationHarness:
    target: Target
    actor: Actor
    cleanup_timeout_seconds: float = 30

    def __post_init__(self) -> None:
        if self.cleanup_timeout_seconds <= 0:
            raise ValueError("conversation cleanup timeout must be positive")

    @property
    def component(self) -> ComponentDefinition:
        return ComponentDefinition(
            kind="conversation",
            version="1",
            config={
                "target": self.target.component.model_dump(mode="json"),
                "actor": self.actor.component.model_dump(mode="json"),
                "cleanup_timeout_seconds": self.cleanup_timeout_seconds,
            },
        )

    async def run(self, case: Case, trial: Trial, trace: TraceRecorder) -> Evidence:
        messages: list[Message] = []
        try:
            session = await self.target.open(trial, trace)
        except Exception as error:
            raise HarnessError("start", str(error)) from error

        primary_error: BaseException | None = None
        finished_evidence: Evidence | None = None
        try:
            for turn in range(case.budget.turns):
                try:
                    decision = await self.actor.next_turn(case, tuple(messages), turn, trace)
                except Exception as error:
                    raise HarnessError(
                        "actor",
                        str(error),
                        evidence=_partial_evidence(messages),
                    ) from error

                if isinstance(decision, Accept):
                    break
                if isinstance(decision, Stop):
                    raise HarnessError(
                        "actor",
                        decision.reason,
                        evidence=_partial_evidence(messages),
                    )

                messages.append(decision.message)
                trace.record(MessageFact(message=decision.message))
                try:
                    replies = await session.send(decision.message)
                except Exception as error:
                    raise HarnessError(
                        "execute",
                        str(error),
                        evidence=_partial_evidence(messages),
                    ) from error
                if not replies:
                    raise HarnessError(
                        "execute",
                        "target returned no messages",
                        evidence=_partial_evidence(messages),
                    )
                for reply in replies:
                    messages.append(reply)
                    trace.record(MessageFact(message=reply))

            try:
                completion = await session.finish(tuple(messages))
            except Exception as error:
                raise HarnessError(
                    "verify",
                    str(error),
                    evidence=_partial_evidence(messages),
                ) from error
            finished_evidence = Evidence(
                public=PublicOutcome(messages=tuple(messages), output=completion.output),
                checks=completion.checks,
                artifacts=completion.artifacts,
            )
            return finished_evidence
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                async with asyncio.timeout(self.cleanup_timeout_seconds):
                    await session.close()
            except Exception as error:
                if primary_error is None:
                    raise HarnessError(
                        "cleanup",
                        str(error)
                        or f"target cleanup exceeded {self.cleanup_timeout_seconds:g} seconds",
                        evidence=finished_evidence or _partial_evidence(messages),
                    ) from error


def _partial_evidence(messages: list[Message]) -> Evidence:
    return Evidence(public=PublicOutcome(messages=tuple(messages)))
