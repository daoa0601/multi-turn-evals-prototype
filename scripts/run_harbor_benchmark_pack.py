"""Run one real Harbor task from each domain in the benchmark canary pack."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic_multiturn_evals import (
    Campaign,
    Case,
    Completed,
    ResourceClaim,
    TrialBudget,
    TrialResult,
    Variant,
    run,
)
from pydantic_multiturn_evals.adapters import HarborDatasetTask, HarborHarness


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    case: str
    dataset: str
    task: str
    domain: str


PACK = (
    BenchmarkSpec("terminal", "terminal-bench-sample@2.0", "chess-best-move", "terminal operation"),
    BenchmarkSpec("tool-use", "bfcl@1.0", "bfcl-live-simple-22-5-0", "function and tool use"),
    BenchmarkSpec("data-analysis", "dabstep@1.0", "dabstep-1517", "multi-step data analysis"),
    BenchmarkSpec(
        "cooperation", "cooperbench@1.0", "cb-dspy-t8587-f2-3", "multi-agent cooperation"
    ),
    BenchmarkSpec(
        "general-assistant",
        "gaia@1.0",
        "cca530fc-4052-43b2-b130-b30968d8aa44",
        "general assistant reasoning",
    ),
)


@dataclass(frozen=True, slots=True)
class PackConfig:
    directory: Path
    executable: Path
    harbor_version: str
    benchmarks: tuple[BenchmarkSpec, ...]
    agent: str = "oracle"
    model: str | None = None
    workers: int = 2
    docker_capacity: int = 2
    docker_platform: str | None = "linux/amd64"
    timeout_seconds: float = 1_800
    inherit_environment: tuple[str, ...] = ("PATH", "HOME")

    def __post_init__(self) -> None:
        if not self.benchmarks:
            raise ValueError("the Harbor pack needs at least one benchmark")
        if self.workers < 1 or self.docker_capacity < 1:
            raise ValueError("workers and Docker capacity must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout must be positive")


async def execute(config: PackConfig) -> dict[str, Any]:
    tasks = tuple(
        HarborDatasetTask(
            case=benchmark.case,
            dataset=benchmark.dataset,
            task=benchmark.task,
            revision=benchmark.dataset.rsplit("@", maxsplit=1)[-1],
        )
        for benchmark in config.benchmarks
    )
    harness = HarborHarness(
        executable=config.executable,
        harbor_version=config.harbor_version,
        tasks=tasks,
        agent=config.agent,
        model=config.model,
        environment="docker",
        docker_platform=config.docker_platform,
        timeout_seconds=config.timeout_seconds,
        inherit_environment=config.inherit_environment,
    )
    campaign = Campaign(
        name="harbor-benchmark-pack",
        revision="fixed-tasks-1",
        cases=tuple(
            Case(
                name=benchmark.case,
                task=f"Run one native task from {benchmark.dataset}.",
                required_checks=("harbor.verified",),
                budget=TrialBudget(seconds=config.timeout_seconds + 60),
            )
            for benchmark in config.benchmarks
        ),
        variants=(
            Variant(
                name=config.agent,
                harness=harness,
                resources=(ResourceClaim(name="docker"),),
            ),
        ),
    )
    report = await run(
        campaign,
        directory=config.directory,
        workers=config.workers,
        capacities={"docker": config.docker_capacity},
    )
    benchmarks_by_case = {benchmark.case: benchmark for benchmark in config.benchmarks}
    results = [
        _result_summary(result, benchmarks_by_case[result.key.case]) for result in report.results
    ]
    summary: dict[str, Any] = {
        "campaign": campaign.name,
        "revision": campaign.revision,
        "directory": str(config.directory.resolve()),
        "mode": "adapter-canary" if config.agent == "oracle" else "model-evaluation",
        "agent": config.agent,
        "model": config.model,
        "harbor_version": config.harbor_version,
        "workers": config.workers,
        "docker_capacity": config.docker_capacity,
        "docker_platform": config.docker_platform,
        "trials": len(results),
        "completed": sum(result["terminal"] == "completed" for result in results),
        "passed": sum(bool(result["passed"]) for result in results),
        "execution_failures": sum(result["terminal"] != "completed" for result in results),
        "results": results,
        "source_state": _source_state(),
    }
    _write_json(config.directory / "pack-summary.json", summary)
    return summary


def _result_summary(result: TrialResult, benchmark: BenchmarkSpec) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "case": result.key.case,
        "dataset": benchmark.dataset,
        "domain": benchmark.domain,
        "terminal": result.terminal.kind,
        "passed": result.passed,
        "duration_seconds": round(result.duration_seconds, 6),
    }
    if isinstance(result.terminal, Completed):
        output = result.terminal.evidence.public.output
        if isinstance(output, dict):
            summary["task_name"] = output.get("task_name")
            summary["trial_name"] = output.get("trial_name")
        summary["native_checks"] = [
            check.model_dump(mode="json") for check in result.terminal.evidence.checks
        ]
    else:
        summary["failure"] = result.terminal.model_dump(mode="json")
    return summary


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_state() -> str:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return "dirty-worktree" if result.stdout else "clean-worktree"


def _harbor_version(executable: Path) -> str:
    result = subprocess.run(
        (str(executable), "--version"),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _selected(names: list[str] | None) -> tuple[BenchmarkSpec, ...]:
    if names is None:
        return PACK
    selected = set(names)
    return tuple(benchmark for benchmark in PACK if benchmark.case in selected)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--case", action="append", choices=[item.case for item in PACK])
    parser.add_argument("--agent", default="oracle")
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--docker-capacity", type=int, default=2)
    parser.add_argument("--docker-platform", default="linux/amd64")
    parser.add_argument("--timeout-seconds", type=float, default=1_800)
    parser.add_argument("--inherit", action="append", default=[])
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    executable_name = shutil.which("harbor")
    if executable_name is None:
        raise SystemExit("Harbor is not installed")
    docker = shutil.which("docker")
    if docker is None:
        raise SystemExit("Docker is not installed")
    subprocess.run(
        (docker, "info"),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    directory = arguments.directory
    if directory is None:
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
        directory = Path("outputs") / "harbor-benchmark-pack" / timestamp
    inherited = tuple(
        dict.fromkeys(name for name in ("PATH", "HOME", *arguments.inherit) if name in os.environ)
    )
    config = PackConfig(
        directory=directory,
        executable=Path(executable_name),
        harbor_version=_harbor_version(Path(executable_name)),
        benchmarks=_selected(arguments.case),
        agent=arguments.agent,
        model=arguments.model,
        workers=arguments.workers,
        docker_capacity=arguments.docker_capacity,
        docker_platform=arguments.docker_platform,
        timeout_seconds=arguments.timeout_seconds,
        inherit_environment=inherited,
    )
    summary = asyncio.run(execute(config))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if config.agent == "oracle" and summary["passed"] != summary["trials"]:
        raise SystemExit("Harbor oracle did not pass every benchmark canary")


if __name__ == "__main__":
    main()
