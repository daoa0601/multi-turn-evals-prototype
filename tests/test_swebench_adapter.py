from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from pydantic_multiturn_evals import Campaign, Case, Completed, Variant, run
from pydantic_multiturn_evals.adapters import SweBenchHarness, SweBenchTask


def test_swebench_adapter_grades_the_solver_patch_in_a_separate_process(tmp_path: Path) -> None:
    solver = _executable(
        tmp_path / "solver.py",
        """
import json, sys
request = json.loads(sys.stdin.readline())
assert set(request) == {'protocol', 'task', 'instance', 'trial'}
print(json.dumps({
  'protocol': 1,
  'patch': 'diff --git a/a.py b/a.py\\n',
  'output': {'summary': 'attempted'},
  'trajectory': [{'tool': 'edit'}],
}))
""",
    )
    evaluator = _executable(
        tmp_path / "evaluator.py",
        """
import json, pathlib, sys
args = sys.argv[1:]
run_id = args[args.index('--run-id') + 1]
instance = args[args.index('--instance') + 1]
prediction = pathlib.Path(args[args.index('--predictions') + 1])
assert json.loads(prediction.read_text())[0]['instance_id'] == instance
root = pathlib.Path.cwd() / 'logs' / 'run_evaluation' / run_id / 'solver' / instance
root.mkdir(parents=True)
(root / 'report.json').write_text(json.dumps({instance: {'resolved': False}}))
(root / 'test_output.txt').write_text('one test failed')
""",
    )
    harness = SweBenchHarness(
        evaluator_argv=(str(evaluator),),
        package_version="5.0.2-boundary",
        tasks=(
            SweBenchTask(
                case="repair",
                dataset="verified",
                instance="owner__repo-1",
                revision="fixture-1",
            ),
        ),
        solver_argv=(str(solver),),
        solver_cwd=tmp_path,
        solver_revision="fixture-1",
    )
    campaign = Campaign(
        name="swebench-boundary",
        revision="1",
        cases=(
            Case(
                name="repair",
                task="Repair the bug",
                data={"private": "must not reach solver"},
                required_checks=("swebench.resolved",),
            ),
        ),
        variants=(Variant(name="only", harness=harness),),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    terminal = report.results[0].terminal
    assert isinstance(terminal, Completed)
    assert terminal.evidence.checks[0].passed is False
    prediction = next((tmp_path / "run").glob("trials/**/prediction.json"))
    assert json.loads(prediction.read_text())[0]["model_patch"].startswith("diff --git")
    native_report = next((tmp_path / "run").glob("trials/**/logs/**/report.json"))
    assert json.loads(native_report.read_text())["owner__repo-1"]["resolved"] is False


def _executable(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path
