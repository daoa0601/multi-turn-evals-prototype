"""One-shot adaptive-actor worker used by the isolated Harbor runtime."""

from __future__ import annotations

import asyncio
import sys

from pydantic import Field

from pydantic_multiturn_evals.models import (
    ActorBrief,
    ActorDecision,
    ActorSpec,
    Exchange,
    StrictModel,
)
from pydantic_multiturn_evals.providers import PydanticActor
from pydantic_multiturn_evals.runner import ActorView

MAX_REQUEST_BYTES = 4 * 1024 * 1024


class ActorWorkerRequest(StrictModel):
    actor: ActorSpec
    scenario_id: str
    brief: ActorBrief
    exchanges: tuple[Exchange, ...]
    remaining_target_turns: int = Field(ge=0)


async def decide(request: ActorWorkerRequest) -> ActorDecision:
    actor = PydanticActor(request.actor)
    return await actor.decide(
        ActorView(
            scenario_id=request.scenario_id,
            brief=request.brief,
            exchanges=request.exchanges,
            remaining_target_turns=request.remaining_target_turns,
        )
    )


def main() -> int:
    data = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(data) > MAX_REQUEST_BYTES:
        raise ValueError("actor worker request exceeded its byte limit")
    request = ActorWorkerRequest.model_validate_json(data)
    decision = asyncio.run(decide(request))
    sys.stdout.write(decision.model_dump_json() + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())
