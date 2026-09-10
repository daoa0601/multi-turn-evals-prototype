"""Short-lived state storage for one evaluation process."""

from __future__ import annotations

from typing import Protocol

from pydantic_multiturn_evals.models import ConversationState


class StateStore(Protocol):
    async def save(self, state: ConversationState) -> None: ...

    async def load(self, run_id: str) -> ConversationState | None: ...


class InMemoryStateStore:
    """Process-local state. Each saved value is copied to prevent shared mutation."""

    def __init__(self) -> None:
        self._states: dict[str, ConversationState] = {}

    async def save(self, state: ConversationState) -> None:
        self._states[state.run_id] = state.model_copy(deep=True)

    async def load(self, run_id: str) -> ConversationState | None:
        state = self._states.get(run_id)
        return state.model_copy(deep=True) if state is not None else None
