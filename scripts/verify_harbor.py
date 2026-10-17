"""Run the real Harbor adapter against Docker with positive and negative agents."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path

from pydantic_multiturn_evals import (
    Campaign,
    Case,
    Completed,
    Failed,
    Report,
    ResourceClaim,
    TrialBudget,
    Variant,
    run,
)
from pydantic_multiturn_evals.adapters import HarborHarness, HarborLocalTask

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "integration" / "harbor" / "task"
CANCEL_TASK = ROOT / "integration" / "harbor" / "cancel-task"


def main() -> None:
    executable = shutil.which("harbor")
    if executable is None:
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
    with tempfile.TemporaryDirectory(prefix="mte-harbor-") as temporary:
        root = Path(temporary)
        report = asyncio.run(_run(Path(executable), root / "outcomes"))
        cancelled = asyncio.run(_cancel(Path(executable), root / "cancellation"))
        projects = {
            path.parent.name.lower()
            for path in root.glob("*/trials/*/*/sample-*/attempt-*/harbor/attempt-*/*/config.json")
        }
    oracle = [result for result in report.results if result.key.variant == "oracle"]
    nop = [result for result in report.results if result.key.variant == "nop"]
    if len(oracle) != 2 or any(
        not isinstance(result.terminal, Completed) or not result.passed for result in oracle
    ):
        raise SystemExit("Harbor oracle did not pass its real verifier")
    if len(nop) != 2 or any(
        not isinstance(result.terminal, Completed) or result.passed for result in nop
    ):
        raise SystemExit("Harbor nop did not produce a real quality failure")
    cancellation = cancelled.results[0].terminal
    if not isinstance(cancellation, Failed) or cancellation.failure.phase != "timeout":
        raise SystemExit("Harbor cancellation did not produce a timeout receipt")
    remaining = _containers(docker, projects)
    if remaining:
        raise SystemExit(f"Harbor left its own containers after cancellation: {sorted(remaining)}")
    print("Harbor adapter: oracle passed, nop failed, cancellation cleaned up")


async def _run(executable: Path, directory: Path) -> Report:
    task = HarborLocalTask(case="adapter-smoke", path=TASK, revision="1.0.0")
    case = Case(
        name="adapter-smoke",
        task="Create the exact answer file.",
        required_checks=("harbor.verified",),
    )

    def harness(agent: str) -> HarborHarness:
        return HarborHarness(
            executable=executable,
            harbor_version="0.1.44",
            tasks=(task,),
            agent=agent,
            environment="docker",
            inherit_environment=("PATH", "HOME"),
            timeout_seconds=300,
        )

    campaign = Campaign(
        name="harbor-adapter-smoke",
        revision="1",
        cases=(case,),
        variants=(
            Variant(
                name="oracle",
                harness=harness("oracle"),
                resources=(ResourceClaim(name="docker"),),
            ),
            Variant(
                name="nop",
                harness=harness("nop"),
                resources=(ResourceClaim(name="docker"),),
            ),
        ),
        samples=2,
    )
    return await run(campaign, directory=directory, workers=4, capacities={"docker": 2})


async def _cancel(executable: Path, directory: Path) -> Report:
    campaign = Campaign(
        name="harbor-adapter-cancel",
        revision="1",
        cases=(
            Case(
                name="cancel-smoke",
                task="Start work that will exceed the deadline.",
                required_checks=("harbor.verified",),
                budget=TrialBudget(seconds=5),
            ),
        ),
        variants=(
            Variant(
                name="oracle",
                harness=HarborHarness(
                    executable=executable,
                    harbor_version="0.1.44",
                    tasks=(
                        HarborLocalTask(
                            case="cancel-smoke",
                            path=CANCEL_TASK,
                            revision="1.0.0",
                        ),
                    ),
                    agent="oracle",
                    environment="docker",
                    inherit_environment=("PATH", "HOME"),
                    timeout_seconds=300,
                    cancel_grace_seconds=30,
                ),
                resources=(ResourceClaim(name="docker"),),
            ),
        ),
    )
    return await run(campaign, directory=directory, capacities={"docker": 1})


def _containers(docker: str, projects: set[str]) -> set[str]:
    output = subprocess.run(
        (docker, "ps", "-a", "--format", '{{.ID}} {{.Label "com.docker.compose.project"}}'),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        parts[0]
        for line in output.splitlines()
        if len(parts := line.split()) == 2 and parts[1] in projects
    }


if __name__ == "__main__":
    main()
