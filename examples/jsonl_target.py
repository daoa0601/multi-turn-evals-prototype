"""Small persistent target used by the offline end-to-end campaign."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def send(payload: object) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def main() -> int:
    profile = sys.argv[1]
    workspace = Path(os.environ["MTE_TRIAL_DIRECTORY"])
    started = workspace / "target-started.json"
    if profile == "orphan":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        (workspace / "target-child-pid").write_text(str(child.pid), encoding="utf-8")

    start = json.loads(sys.stdin.readline())
    if start["type"] != "start":
        raise ValueError("first request must start the target")
    started.write_text(
        json.dumps({"profile": profile, "start_keys": sorted(start)}) + "\n",
        encoding="utf-8",
    )
    send({"protocol": 1, "type": "ready"})
    turn = 0
    for line in sys.stdin:
        message = json.loads(line)
        if message["type"] == "turn":
            turn += 1
            if profile == "invalid":
                print("not-json", flush=True)
                continue
            if profile == "hang":
                (workspace / "target-pid").write_text(str(os.getpid()), encoding="utf-8")
                time.sleep(60)
            content = (
                "cannot resolve"
                if profile == "baseline"
                else "need identifier"
                if turn == 1
                else f"resolved: {message['message']['content']}"
            )
            send(
                {
                    "protocol": 1,
                    "type": "reply",
                    "id": message["id"],
                    "messages": [
                        {"role": "assistant", "content": content, "actor": "offline-target"}
                    ],
                }
            )
        elif message["type"] == "finish":
            send(
                {
                    "protocol": 1,
                    "type": "finished",
                    "completion": {
                        "output": {"turns": turn},
                        "checks": [
                            {
                                "name": "completed",
                                "passed": True,
                                "reason": "the live target reached its finish request",
                            }
                        ],
                        "artifacts": [{"name": "startup", "path": started.name}],
                    },
                }
            )
        elif message["type"] == "close":
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
