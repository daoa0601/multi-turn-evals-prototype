from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

from pydantic_multiturn_evals import (
    Campaign,
    Case,
    Completed,
    Failed,
    ResourceClaim,
    TrialBudget,
    Variant,
    run,
)
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


@pytest.mark.parametrize("verifier_report", [False, True], ids=["no-report", "real-report"])
@pytest.mark.parametrize("plant_location", ["root", "logs"])
def test_swebench_ignores_reports_planted_by_the_solver(
    tmp_path: Path, verifier_report: bool, plant_location: str
) -> None:
    solver = _executable(
        tmp_path / "plant_report.py",
        """
import json, os, pathlib, sys
request = json.loads(sys.stdin.readline())
root = pathlib.Path(os.environ['MTE_TRIAL_DIRECTORY'])
run_id = json.loads((root / 'swebench-state.json').read_text())['run_id']
directory = root / 'logs' / 'run_evaluation' / run_id / 'solver' if sys.argv[1] == 'logs' else root
directory.mkdir(parents=True, exist_ok=True)
(directory / 'report.json').write_text(json.dumps({request['instance']: {'resolved': True}}))
print(json.dumps({'protocol': 1, 'patch': 'diff --git a/a.py b/a.py\\n'}))
""",
    )
    evaluator = _executable(
        tmp_path / "verifier.py",
        f"""
import json, pathlib, sys
args = sys.argv[1:]
if {verifier_report!r}:
    run_id = args[args.index('--run-id') + 1]
    instance = args[args.index('--instance') + 1]
    root = pathlib.Path.cwd() / 'logs' / 'run_evaluation' / run_id / 'solver' / instance
    root.mkdir(parents=True)
    (root / 'report.json').write_text(json.dumps({{instance: {{'resolved': False}}}}))
""",
    )
    harness = SweBenchHarness(
        evaluator_argv=(str(evaluator),),
        package_version="5.0.2-boundary",
        tasks=(
            SweBenchTask(case="repair", dataset="verified", instance="owner__repo-2", revision="1"),
        ),
        solver_argv=(str(solver), plant_location),
        solver_cwd=tmp_path,
        solver_revision="1",
    )
    campaign = Campaign(
        name="swebench-report-authority",
        revision="1",
        cases=(Case(name="repair", task="Repair the bug", required_checks=("swebench.resolved",)),),
        variants=(Variant(name="only", harness=harness),),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    terminal = report.results[0].terminal
    if verifier_report:
        assert isinstance(terminal, Completed)
        assert terminal.evidence.checks[0].passed is False
    else:
        assert isinstance(terminal, Failed)
        assert terminal.failure.phase == "verify"
        assert "without an instance report" in terminal.failure.message


def test_swebench_native_ids_isolate_concurrent_trials_and_output_roots(tmp_path: Path) -> None:
    evaluator = _executable(
        tmp_path / "evaluator.py",
        """
import json, pathlib, sys
args = sys.argv[1:]
run_id = args[args.index('--run-id') + 1]
instance = args[args.index('--instance') + 1]
root = pathlib.Path('logs/run_evaluation') / run_id / 'gold' / instance
root.mkdir(parents=True)
(root / 'report.json').write_text(json.dumps({instance: {'resolved': True}}))
""",
    )
    harness = SweBenchHarness(
        evaluator_argv=(str(evaluator),),
        package_version="5.0.2-boundary",
        tasks=tuple(
            SweBenchTask(case=case, dataset="verified", instance="owner__repo-1", revision="1")
            for case in ("repair-one", "repair")
        ),
    )
    campaign = Campaign(
        name="native-ids",
        revision="1",
        cases=tuple(
            Case(name=case, task="Repair", required_checks=("swebench.resolved",))
            for case in ("repair-one", "repair")
        ),
        variants=tuple(
            Variant(name=name, harness=harness) for name in ("baseline", "one-baseline")
        ),
    )

    async def exercise() -> None:
        reports = await asyncio.gather(
            run(campaign, directory=tmp_path / "first"),
            run(campaign, directory=tmp_path / "second"),
        )
        assert all(report.exit_code == 0 for report in reports)

    asyncio.run(exercise())

    states = list(tmp_path.glob("*/trials/**/swebench-state.json"))
    run_ids = [json.loads(path.read_text())["run_id"] for path in states]
    assert len(run_ids) == 8
    assert len(set(run_ids)) == 8, "independent SWE-bench attempts share native container names"
    for path in states:
        state = json.loads(path.read_text())
        reports = list(path.parent.glob(f"*/logs/run_evaluation/{state['run_id']}/**/report.json"))
        assert len(reports) == 1, f"persisted native run ID does not locate its report: {path}"


@pytest.mark.parametrize("cancel_campaign", [False, True], ids=["deadline", "cancellation"])
@pytest.mark.parametrize("docker_behavior", ["remove", "hang", "error"])
def test_swebench_removes_only_its_own_verifier_containers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_campaign: bool,
    docker_behavior: str,
) -> None:
    containers = tmp_path / "containers"
    containers.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("TEST_CONTAINERS", str(containers))
    _executable(
        tmp_path / "docker",
        f"""
import os, pathlib, re, sys, time
root = pathlib.Path(os.environ['TEST_CONTAINERS'])
args = sys.argv[1:]
if {docker_behavior!r} == 'hang':
    (root.parent / 'cleanup-pid').write_text(str(os.getpid()))
    time.sleep(60)
if {docker_behavior!r} == 'error':
    raise SystemExit(1)
if args[0] == 'ps':
    pattern = args[args.index('--filter') + 1].removeprefix('name=')
    for path in root.iterdir():
        if re.search(pattern, '/' + path.name):
            print(path.name)
elif args[:2] == ['rm', '--force']:
    for name in args[2:]:
        (root / name).unlink()
else:
    raise SystemExit('unexpected Docker command: ' + str(args))
""",
    )
    evaluator = _executable(
        tmp_path / "evaluator.py",
        """
import os, pathlib, signal, sys, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
args = sys.argv[1:]
run_id = args[args.index('--run-id') + 1]
instance = args[args.index('--instance') + 1].lower()
root = pathlib.Path(os.environ['TEST_CONTAINERS'])
name = 'sweb.eval.' + instance + '.' + run_id
for suffix in ('', '.1234567890', '.other', '-other'):
    (root / (name + suffix)).write_text('allocated')
(root / ('sweb.eval.' + instance + '.another-run')).write_text('another trial')
pathlib.Path('ready').write_text('ready')
time.sleep(60)
""",
    )
    harness = SweBenchHarness(
        evaluator_argv=(str(evaluator),),
        evaluator_environment=("PATH", "TEST_CONTAINERS"),
        package_version="5.0.2-boundary",
        cleanup_timeout_seconds=0.5 if docker_behavior == "hang" else 2,
        tasks=(
            SweBenchTask(
                case="repair", dataset="verified", instance="Owner__Repo.name-1", revision="1"
            ),
        ),
    )
    campaign = Campaign(
        name="swebench-cleanup",
        revision="1",
        cases=(
            Case(
                name="repair",
                task="Repair",
                required_checks=("swebench.resolved",),
                budget=TrialBudget(seconds=30 if cancel_campaign else 0.5),
            ),
        ),
        variants=(
            Variant(name="only", harness=harness, resources=(ResourceClaim(name="docker"),)),
        ),
    )
    directory = tmp_path / "run"

    async def exercise() -> None:
        task = asyncio.create_task(run(campaign, directory=directory, capacities={"docker": 1}))
        try:
            async with asyncio.timeout(5):
                while not list(directory.glob("trials/**/ready")):
                    await asyncio.sleep(0.01)
            if cancel_campaign:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                report = await task
                terminal = report.results[0].terminal
                assert isinstance(terminal, Failed)
                assert terminal.failure.phase == "timeout"
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    started = time.monotonic()
    asyncio.run(exercise())
    assert time.monotonic() - started < 6, "Docker cleanup exceeded its bounded window"

    state = json.loads(next(directory.glob("trials/**/swebench-state.json")).read_text())
    cleanup = json.loads(next(directory.glob("trials/**/swebench-cleanup.json")).read_text())
    name = f"sweb.eval.owner__repo.name-1.{state['run_id']}"
    if docker_behavior == "remove":
        assert cleanup["status"] == "completed", cleanup
        assert not (containers / name).exists(), "cancelled attempt's verifier container survived"
        assert not (containers / (name + ".1234567890")).exists()
    else:
        assert cleanup["status"] == "failed"
        assert cleanup["message"]
        if docker_behavior == "hang":
            pid = int((tmp_path / "cleanup-pid").read_text())
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    assert (containers / (name + ".other")).exists(), "cleanup removed another trial's container"
    assert (containers / (name + "-other")).exists()
    assert (containers / "sweb.eval.owner__repo.name-1.another-run").exists()


@pytest.mark.parametrize(
    "exit_code, report_kind, phase",
    [
        (2, "missing", "verify"),
        (2, "valid", None),
        (0, "missing", "verify"),
        (0, "malformed", "verify"),
        (0, "no-resolved", "protocol"),
        (0, "invalid-resolved", "protocol"),
        (0, "duplicate", "protocol"),
    ],
)
def test_swebench_cleans_up_after_abnormal_exit_or_unusable_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
    report_kind: str,
    phase: str | None,
) -> None:
    container = tmp_path / "container"
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("TEST_CONTAINER", str(container))
    _executable(
        tmp_path / "docker",
        """
import os, pathlib, sys
container = pathlib.Path(os.environ['TEST_CONTAINER'])
args = sys.argv[1:]
if args[0] == 'ps':
    print(container.read_text())
elif args[:2] == ['rm', '--force']:
    assert args[2:] == [container.read_text()]
    container.unlink()
else:
    raise SystemExit('unexpected Docker command: ' + str(args))
""",
    )
    evaluator = _executable(
        tmp_path / "evaluator.py",
        f"""
import json, os, pathlib, sys
args = sys.argv[1:]
run_id = args[args.index('--run-id') + 1]
instance = args[args.index('--instance') + 1]
pathlib.Path(os.environ['TEST_CONTAINER']).write_text('sweb.eval.' + instance + '.' + run_id)
root = pathlib.Path('logs/run_evaluation') / run_id / 'gold' / instance
root.mkdir(parents=True)
kind = {report_kind!r}
value = {{instance: {{'resolved': False}}}}
if kind == 'no-resolved':
    value = {{instance: {{}}}}
elif kind == 'invalid-resolved':
    value = {{instance: {{'resolved': 'unknown'}}}}
if kind != 'missing':
    (root / 'report.json').write_text('{{' if kind == 'malformed' else json.dumps(value))
if kind == 'duplicate':
    (root / 'duplicate').mkdir()
    (root / 'duplicate' / 'report.json').write_text(json.dumps(value))
raise SystemExit({exit_code})
""",
    )
    campaign = Campaign(
        name="swebench-abnormal-exit",
        revision="1",
        cases=(Case(name="repair", task="Repair", required_checks=("swebench.resolved",)),),
        variants=(
            Variant(
                name="only",
                harness=SweBenchHarness(
                    evaluator_argv=(str(evaluator),),
                    evaluator_environment=("PATH", "TEST_CONTAINER"),
                    package_version="5.0.2-boundary",
                    tasks=(
                        SweBenchTask(
                            case="repair", dataset="verified", instance="repo-1", revision="1"
                        ),
                    ),
                ),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    terminal = report.results[0].terminal
    if phase is None:
        assert isinstance(terminal, Completed)
        assert terminal.evidence.checks[0].passed is False
    else:
        assert isinstance(terminal, Failed)
        assert terminal.failure.phase == phase
    assert not container.exists(), "SWE-bench left its verifier container allocated"
    cleanup = json.loads(next(tmp_path.glob("run/trials/**/swebench-cleanup.json")).read_text())
    assert cleanup["status"] == "completed", cleanup


def _executable(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path
