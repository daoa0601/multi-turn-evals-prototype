"""Run the checked-in GLM Pydantic target through command protocol 2."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from pydantic_multiturn_evals.models import (
    AssistantTurn,
    CaseKey,
    ConversationView,
    Exchange,
    SessionContext,
    UserTurn,
)
from pydantic_multiturn_evals.protocol import (
    CloseRequest,
    FinishRequest,
    ReplyResponse,
    StartRequest,
    TurnRequest,
)
from pydantic_multiturn_evals.spec import load_target
from pydantic_multiturn_evals.targets import build_target

ROOT = Path(__file__).parents[1]


async def main() -> int:
    target = build_target(load_target(ROOT / "targets" / "glm-wide.yaml"))
    start = StartRequest.model_validate(await _read())
    wire = start.session
    context = SessionContext(
        suite_name=wire.suite,
        target_name=target.name,
        target_kind=target.kind,
        target_version=target.version,
        key=CaseKey(
            scenario_id=wire.scenario_id,
            repeat_index=wire.repeat_index,
        ),
        run_id=wire.run_id,
        target_instructions=wire.target_instructions,
        fixture=wire.fixture,
        comparison_id=wire.comparison_id,
        arm=wire.arm,
    )

    async with target.session(context) as session:
        _send({"protocol": 2, "type": "ready"})
        while True:
            payload = await _read()
            message_type = payload.get("type")
            if message_type == "turn":
                request = TurnRequest.model_validate(payload)
                reply = await session.reply(
                    _conversation_view(wire.run_id, wire.scenario_id, request)
                )
                _send(
                    ReplyResponse(
                        protocol=2,
                        type="reply",
                        id=request.id,
                        assistant_text=reply.assistant_text,
                        session_id=reply.session_id,
                        evidence={"adapter": "glm-pydantic-jsonl", **reply.evidence},
                    ).model_dump(mode="json")
                )
                continue
            if message_type == "finish":
                request = FinishRequest.model_validate(payload)
                completion = await session.finish(request.outcome)
                _send(
                    {
                        "protocol": 2,
                        "type": "finished",
                        "completion": completion.model_dump(mode="json"),
                    }
                )
                continue
            CloseRequest.model_validate(payload)
            return 0


def _conversation_view(run_id: str, scenario_id: str, request: TurnRequest) -> ConversationView:
    messages = request.messages
    if not messages or not isinstance(messages[-1], UserTurn):
        raise ValueError("turn messages must end with a user message")
    history = messages[:-1]
    if len(history) % 2 != 0:
        raise ValueError("turn history must contain complete exchanges")
    exchanges: list[Exchange] = []
    for index in range(0, len(history), 2):
        user = history[index]
        assistant = history[index + 1]
        if not isinstance(user, UserTurn) or not isinstance(assistant, AssistantTurn):
            raise ValueError("turn history must alternate user and assistant messages")
        exchanges.append(Exchange(user=user, assistant=assistant))
    return ConversationView(
        run_id=run_id,
        scenario_id=scenario_id,
        exchanges=tuple(exchanges),
        pending_user=messages[-1],
    )


async def _read() -> dict[str, object]:
    line = await asyncio.to_thread(sys.stdin.readline)
    if not line:
        raise RuntimeError("command protocol input closed unexpectedly")
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("command protocol message must be an object")
    return payload


def _send(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
