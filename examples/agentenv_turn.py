"""Example turn command to bake into an AgentENV template."""

from __future__ import annotations

import json
import os
from pathlib import Path

request_path = Path(os.environ["MULTITURN_REQUEST_PATH"])
response_path = Path(os.environ["MULTITURN_RESPONSE_PATH"])
request = json.loads(request_path.read_text(encoding="utf-8"))
latest_message = request["messages"][-1]["content"]
response_path.parent.mkdir(parents=True, exist_ok=True)
response_path.write_text(
    json.dumps(
        {
            "protocol": 1,
            "type": "reply",
            "assistant_text": f"AgentENV adapter received: {latest_message}",
            "evidence": {"adapter": "agentenv-example"},
        }
    ),
    encoding="utf-8",
)
