from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from scripts.run_harbor_benchmark_pack import BenchmarkSpec, PackConfig, execute


def test_benchmark_pack_runs_the_authored_task_and_resumes_without_replay(tmp_path: Path) -> None:
    executable = tmp_path / "harbor"
    calls = tmp_path / "calls.txt"
    executable.write_text(
        f"#!{sys.executable}\n"
        + f"CALLS = {str(calls)!r}\n"
        + r"""
import json
import pathlib
import sys

arguments = sys.argv[1:]
task = arguments[arguments.index("--task-name") + 1]
if task != "fixed-tool-task" or "--n-tasks" in arguments:
    raise SystemExit("Expected the authored task, not a dataset sample")
with open(CALLS, "a") as output:
    output.write(task + "\n")
jobs = pathlib.Path(arguments[arguments.index("--jobs-dir") + 1])
name = arguments[arguments.index("--job-name") + 1]
trial = jobs / name / "native-trial"
trial.mkdir(parents=True)
(trial / "result.json").write_text(json.dumps({
    "task_name": task,
    "trial_name": "native-trial",
    "agent_info": {"name": "oracle", "version": "fixture"},
    "verifier_result": {"rewards": {"reward": 1}},
}))
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    config = PackConfig(
        directory=tmp_path / "run",
        executable=executable,
        harbor_version="boundary-fixture",
        benchmarks=(BenchmarkSpec("tools", "tools@1.0", "fixed-tool-task", "tool use"),),
    )

    first = asyncio.run(execute(config))
    resumed = asyncio.run(execute(config))

    assert first["passed"] == 1
    assert resumed["results"][0]["task_name"] == "fixed-tool-task"
    assert calls.read_text().splitlines() == ["fixed-tool-task"]
    manifest = json.loads((config.directory / "campaign.json").read_text())
    binding = manifest["variants"][0]["harness"]["config"]["tasks"][0]
    assert binding["task"] == "fixed-tool-task"
