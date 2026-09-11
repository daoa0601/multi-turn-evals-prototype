from __future__ import annotations

import asyncio
import io
from typing import Any

from pydantic_multiturn_evals import harbor_actor_worker
from pydantic_multiturn_evals.harbor_actor_worker import ActorWorkerRequest
from pydantic_multiturn_evals.models import (
    AcceptDecision,
    ActorBrief,
    ActorSpec,
    AssistantTurn,
    Exchange,
    UserTurn,
)


def request() -> ActorWorkerRequest:
    return ActorWorkerRequest(
        actor=ActorSpec(),
        scenario_id="help",
        brief=ActorBrief(persona="A user", goal="Get help"),
        exchanges=(
            Exchange(
                user=UserTurn(content="Help."),
                assistant=AssistantTurn(content="I can help."),
            ),
        ),
        remaining_target_turns=1,
    )


def test_actor_worker_uses_the_existing_pydantic_actor(monkeypatch: Any) -> None:
    class FakeActor:
        def __init__(self, spec: ActorSpec) -> None:
            self.spec = spec

        async def decide(self, view: object) -> AcceptDecision:
            return AcceptDecision(reason="Done.")

    monkeypatch.setattr(harbor_actor_worker, "PydanticActor", FakeActor)

    decision = asyncio.run(harbor_actor_worker.decide(request()))

    assert decision == AcceptDecision(reason="Done.")


def test_actor_worker_main_has_a_bounded_json_contract(monkeypatch: Any) -> None:
    async def fake_decide(value: ActorWorkerRequest) -> AcceptDecision:
        return AcceptDecision(reason=f"Handled {value.scenario_id}.")

    output = io.StringIO()
    monkeypatch.setattr(
        harbor_actor_worker.sys,
        "stdin",
        type("Input", (), {"buffer": io.BytesIO(request().model_dump_json().encode())})(),
    )
    monkeypatch.setattr(harbor_actor_worker.sys, "stdout", output)
    monkeypatch.setattr(harbor_actor_worker, "decide", fake_decide)

    assert harbor_actor_worker.main() == 0
    assert '"kind":"accept"' in output.getvalue()
