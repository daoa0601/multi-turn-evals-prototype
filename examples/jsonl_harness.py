"""Minimal command-target adapter for the multiturn-evals JSONL protocol."""

from __future__ import annotations

import json
import sys


def send(payload: dict[str, object]) -> None:
    print(json.dumps(payload), flush=True)


for line in sys.stdin:
    request = json.loads(line)
    if request["type"] == "start":
        send({"protocol": 1, "type": "ready"})
        continue
    if request["type"] == "close":
        break

    latest_user_message = request["messages"][-1]["content"]
    send(
        {
            "protocol": 1,
            "type": "reply",
            "id": request["id"],
            "assistant_text": f"Example harness received: {latest_user_message}",
            "evidence": {"adapter": "jsonl-example"},
        }
    )
