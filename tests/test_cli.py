from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from pydantic_multiturn_evals import (
    Campaign,
    Case,
    ComponentDefinition,
    Evidence,
    TraceRecorder,
    Trial,
    Variant,
    run,
)
from pydantic_multiturn_evals.cli import main
from pydantic_multiturn_evals.harnesses import FunctionHarness


def test_cli_preserves_quality_failure_when_reading_a_resumed_run(tmp_path: Path) -> None:
    reference = "examples.offline_campaign:campaign"

    assert main(["run", reference, "--directory", str(tmp_path), "--workers", "3"]) == 1
    assert main(["run", reference, "--directory", str(tmp_path), "--workers", "3"]) == 1
    assert main(["report", str(tmp_path), "--compare", "baseline", "candidate"]) == 1


def test_cli_preserves_execution_failure_from_a_saved_report(tmp_path: Path) -> None:
    async def broken(_case: Case, _trial: Trial, _trace: TraceRecorder) -> Evidence:
        raise RuntimeError("target unavailable")

    campaign = Campaign(
        name="execution-failure",
        revision="1",
        cases=(Case(name="case", task="run", required_checks=("done",)),),
        variants=(
            Variant(
                name="direct",
                harness=FunctionHarness(
                    component=ComponentDefinition(kind="broken-target", version="1"),
                    function=broken,
                ),
            ),
        ),
    )
    asyncio.run(run(campaign, directory=tmp_path))

    assert main(["report", str(tmp_path)]) == 2


def test_cli_runs_separate_static_shards_and_merges_exact_pairs(tmp_path: Path) -> None:
    reference = "examples.offline_campaign:campaign"
    roots = (tmp_path / "shard-0", tmp_path / "shard-1")

    for index, root in enumerate(roots):
        assert (
            main(
                [
                    "run",
                    reference,
                    "--directory",
                    str(root),
                    "--shard",
                    f"{index}/2",
                ]
            )
            == 1
        )
    assert (
        main(
            [
                "merge",
                reference,
                *(str(root) for root in roots),
                "--compare",
                "baseline",
                "candidate",
            ]
        )
        == 1
    )


def test_installed_cli_loads_campaign_from_working_directory(tmp_path: Path) -> None:
    (tmp_path / "authored.py").write_text(
        "from pydantic_multiturn_evals import "
        "Campaign, Case, ComponentDefinition, Evidence, NativeCheck, Variant\n"
        "from pydantic_multiturn_evals.harnesses import FunctionHarness\n"
        "async def task(case, trial, trace):\n"
        "    return Evidence(checks=(NativeCheck(name='done', passed=True, reason='done'),))\n"
        "campaign = Campaign(name='local', revision='1', "
        "cases=(Case(name='case', task='Run', required_checks=('done',)),), "
        "variants=(Variant(name='direct', harness=FunctionHarness("
        "component=ComponentDefinition(kind='local', version='1'), function=task)),))\n",
        encoding="utf-8",
    )
    executable = Path(sys.executable).parent / "multiturn-evals"
    environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}

    result = subprocess.run(
        [str(executable), "run", "authored:campaign", "--directory", "results"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["results"][0]["terminal"]["evidence"]["checks"][0]["passed"] is True
