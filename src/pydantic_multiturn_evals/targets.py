"""Target lifecycle and the exhaustive target factory."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Literal, Protocol

from pydantic_multiturn_evals.models import (
    CommandTargetSpec,
    ConversationView,
    PydanticAITargetSpec,
    SessionContext,
    TargetFailureEvidence,
    TargetReply,
)
from pydantic_multiturn_evals.spec import LoadedTarget


class TargetSession(Protocol):
    async def reply(self, view: ConversationView) -> TargetReply: ...


class Target(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> Literal["pydantic_ai", "command"]: ...

    @property
    def version(self) -> int: ...

    def session(self, context: SessionContext) -> AbstractAsyncContextManager[TargetSession]: ...

    def failure_evidence(self) -> tuple[TargetFailureEvidence, ...]: ...


def build_target(loaded: LoadedTarget) -> Target:
    """Build one supported target without exposing lifecycle work to callers."""

    spec = loaded.spec
    if isinstance(spec, PydanticAITargetSpec):
        from pydantic_multiturn_evals.providers import PydanticAITarget

        return PydanticAITarget(spec)
    if isinstance(spec, CommandTargetSpec):
        from pydantic_multiturn_evals.command_target import CommandTarget

        return CommandTarget(spec)
    raise AssertionError(f"unhandled target kind: {spec}")
