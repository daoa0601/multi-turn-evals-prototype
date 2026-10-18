"""Run the real SWE-bench evaluator on one gold and one irrelevant patch."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic_multiturn_evals import (
    Campaign,
    Case,
    Completed,
    Report,
    ResourceClaim,
    TrialBudget,
    Variant,
    run,
)
from pydantic_multiturn_evals.adapters import SweBenchHarness, SweBenchTask

INSTANCE = "sympy__sympy-20590"
IRRELEVANT_PATCH = """diff --git a/mte_irrelevant.txt b/mte_irrelevant.txt
new file mode 100644
index 0000000..8e27be7
--- /dev/null
+++ b/mte_irrelevant.txt
@@ -0,0 +1 @@
+irrelevant
"""


def main() -> None:
    uvx = shutil.which("uvx")
    docker = shutil.which("docker")
    if uvx is None or docker is None:
        raise SystemExit("uvx and Docker are required")
    subprocess.run(
        (docker, "info"),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with tempfile.TemporaryDirectory(prefix="mte-swebench-") as temporary:
        report = asyncio.run(_run(uvx, Path(temporary)))
    by_variant = {result.key.variant: result for result in report.results}
    gold = by_variant["gold"]
    wrong = by_variant["wrong"]
    if not isinstance(gold.terminal, Completed) or not gold.passed:
        raise SystemExit("SWE-bench did not resolve its official reference patch")
    if not isinstance(wrong.terminal, Completed) or wrong.passed:
        raise SystemExit("SWE-bench did not reject the irrelevant patch")
    print("SWE-bench adapter: gold passed, irrelevant patch failed, reports preserved")


async def _run(uvx: str, directory: Path) -> Report:
    task = SweBenchTask(
        case="sympy-20590",
        dataset="verified",
        split="test",
        instance=INSTANCE,
        revision="SWE-bench_Verified-test",
    )
    evaluator = (uvx, "--from", "swebench==5.0.2", "swebench")
    solver = (
        sys.executable,
        "-c",
        "import json,sys; json.loads(sys.stdin.readline()); "
        f"print(json.dumps({{'protocol':1,'patch':{IRRELEVANT_PATCH!r}}}))",
    )

    def harness(
        solver_argv: tuple[str, ...] | None = None,
        solver_name: str = "solver",
    ) -> SweBenchHarness:
        return SweBenchHarness(
            evaluator_argv=evaluator,
            package_version="5.0.2",
            tasks=(task,),
            evaluator_environment=("PATH", "HOME"),
            verifier_timeout_seconds=600,
            solver_argv=solver_argv,
            solver_cwd=Path.cwd(),
            solver_name=solver_name,
            solver_revision="irrelevant-v1" if solver_argv is not None else None,
        )

    campaign = Campaign(
        name="swebench-adapter-smoke",
        revision="5.0.2",
        cases=(
            Case(
                name="sympy-20590",
                task="Verify a repair for the selected instance.",
                required_checks=("swebench.resolved",),
                budget=TrialBudget(seconds=700),
            ),
        ),
        variants=(
            Variant(
                name="gold",
                harness=harness(),
                resources=(ResourceClaim(name="docker"),),
            ),
            Variant(
                name="wrong",
                harness=harness(solver, "irrelevant-patch"),
                resources=(ResourceClaim(name="docker"),),
            ),
        ),
    )
    return await run(campaign, directory=directory, workers=2, capacities={"docker": 2})


if __name__ == "__main__":
    main()
