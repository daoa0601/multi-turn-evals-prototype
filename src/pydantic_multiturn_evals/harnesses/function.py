"""A direct harness for in-process systems and SDKs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic_multiturn_evals.model import (
    Case,
    ComponentDefinition,
    Evidence,
    TraceRecorder,
    Trial,
)

HarnessFunction = Callable[[Case, Trial, TraceRecorder], Awaitable[Evidence]]


@dataclass(frozen=True, slots=True)
class FunctionHarness:
    component: ComponentDefinition
    function: HarnessFunction

    async def run(self, case: Case, trial: Trial, trace: TraceRecorder) -> Evidence:
        return await self.function(case, trial, trace)
