from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_multiturn_harbor_agent import (
    CASE_END,
    CASE_START,
    PydanticAdaptiveUserAgent,
)


class SlowEnvironment:
    async def exec(self, command: str) -> SimpleNamespace:
        await asyncio.sleep(1)
        return SimpleNamespace(return_code=0, stdout="late reply")


def test_scenario_timeout_bounds_the_complete_harbor_conversation(tmp_path: Path) -> None:
    case = {
        "schema_version": 1,
        "run_id": "run-1",
        "key": {"scenario_id": "slow", "repeat_index": 1},
        "scenario": {
            "id": "slow",
            "first_prompt": "Help.",
            "actor": {"persona": "User", "goal": "Get help", "rules": []},
            "judge_rubric": "The answer helps.",
            "tags": [],
            "limits": None,
        },
        "actor": {},
        "limits": {"max_target_turns": 1, "timeout_seconds": 0.05},
    }
    instruction = f"{CASE_START}\n{json.dumps(case)}\n{CASE_END}"
    agent = PydanticAdaptiveUserAgent(
        tmp_path,
        actor_command=[sys.executable, "-c", "raise SystemExit(0)"],
    )

    with pytest.raises(RuntimeError, match="exceeded 0.05 seconds"):
        asyncio.run(agent.run(instruction, SlowEnvironment(), SimpleNamespace(metadata={})))
