"""Target lifecycle and the exhaustive target factory."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Protocol

from pydantic_multiturn_evals.models import (
    AgentEnvTargetSpec,
    CommandTargetSpec,
    ConversationView,
    PydanticAITargetSpec,
    SessionContext,
    SessionOutcome,
    TargetCompletion,
    TargetFailureEvidence,
    TargetKind,
    TargetReply,
)
from pydantic_multiturn_evals.spec import LoadedTarget


class TargetSession(Protocol):
    async def reply(self, view: ConversationView) -> TargetReply: ...

    async def finish(self, outcome: SessionOutcome) -> TargetCompletion: ...


class Target(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> TargetKind: ...

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
    if isinstance(spec, AgentEnvTargetSpec):
        from pydantic_multiturn_evals.agentenv_target import AgentEnvTarget

        return AgentEnvTarget(spec)
    raise AssertionError(f"unhandled target kind: {spec}")
