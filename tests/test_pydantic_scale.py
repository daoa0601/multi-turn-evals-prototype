from __future__ import annotations

import asyncio
import json
from pathlib import Path

from scripts.run_pydantic_scale import ScaleConfig, execute


def test_scale_runner_drives_concurrent_pydantic_trials_and_resumes(tmp_path: Path) -> None:
    summary = asyncio.run(
        execute(
            ScaleConfig(
                directory=tmp_path,
                cases=2,
                samples=2,
                workers=4,
                model_delay_ms=5,
            )
        )
    )

    assert summary["trials"] == 8
    assert summary["passed"] == 8
    assert summary["peak_model_concurrency"] == 4
    assert summary["model_calls"] == {"actor": 16, "judge": 8, "target": 8}
    assert summary["recorded_model_calls"] == 32
    assert summary["recorded_calls_by_role"] == {"actor": 16, "judge": 8, "target": 8}
    assert summary["resume_executed_trials"] == 0
    saved = json.loads((tmp_path / "scale-summary.json").read_text(encoding="utf-8"))
    assert saved["receipts"] == 8
    assert saved["traces"] == 8
    receipt = json.loads(next(tmp_path.glob("trials/**/result.json")).read_text(encoding="utf-8"))
    assert [call["role"] for call in receipt["model_calls"]] == [
        "actor",
        "target",
        "actor",
        "judge",
    ]
    assert all(call["status"] == "completed" for call in receipt["model_calls"])
    assert all(call["usage"]["requests"] == 1 for call in receipt["model_calls"])
