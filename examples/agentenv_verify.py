"""Example live-state verifier to bake into an AgentENV template."""

from __future__ import annotations

import json
import os
from pathlib import Path

outcome_path = Path(os.environ["MULTITURN_REQUEST_PATH"])
response_path = Path(os.environ["MULTITURN_RESPONSE_PATH"])
outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
has_exchange = bool(outcome["transcript"]["exchanges"])
response_path.write_text(
    json.dumps(
        {
            "protocol": 1,
            "type": "verification",
            "passed": has_exchange,
            "reward": 1.0 if has_exchange else 0.0,
            "reason": "The sandbox recorded a complete exchange."
            if has_exchange
            else "No complete exchange was recorded.",
            "evidence": {"exchange_count": len(outcome["transcript"]["exchanges"])},
        }
    ),
    encoding="utf-8",
)
