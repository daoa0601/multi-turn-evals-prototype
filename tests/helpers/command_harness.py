from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def send(payload: dict[str, object]) -> None:
    print(json.dumps(payload), flush=True)


mode = os.environ.get("HARNESS_MODE", "ok")
variant = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HARNESS_VARIANT", "target")
pid_file = os.environ.get("HARNESS_PID_FILE")
if pid_file:
    Path(pid_file).write_text(str(os.getpid()), encoding="utf-8")

for raw_line in sys.stdin:
    request = json.loads(raw_line)
    if request["type"] == "start":
        send({"protocol": 1, "type": "ready"})
        continue
    if request["type"] == "close":
        break
    if mode == "hang":
        time.sleep(60)
    if mode == "malformed":
        print("not-json", flush=True)
        continue
    if mode == "oversized":
        print("x" * 4096, flush=True)
        continue
    if mode == "wrong-id":
        send(
            {
                "protocol": 1,
                "type": "reply",
                "id": request["id"] + 1,
                "assistant_text": "wrong turn",
            }
        )
        continue
    if mode == "exit":
        print("command harness failed", file=sys.stderr, flush=True)
        raise SystemExit(23)
    if mode == "heavy-stderr":
        print("e" * 1024 * 1024, file=sys.stderr, flush=True)

    turn_id = request["id"]
    send(
        {
            "protocol": 1,
            "type": "reply",
            "id": turn_id,
            "assistant_text": f"{variant} reply {turn_id}",
            "session_id": f"{variant}-{os.getpid()}",
            "evidence": {"message_count": len(request["messages"])},
        }
    )
