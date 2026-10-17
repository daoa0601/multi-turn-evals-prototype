from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from pydantic_multiturn_evals import (
    Campaign,
    Case,
    Completed,
    Failed,
    Trial,
    TrialKey,
    Variant,
    run,
)
from pydantic_multiturn_evals.harnesses import OneShotCommandHarness
from pydantic_multiturn_evals.tracing import LocalTrace


def test_one_shot_job_returns_native_evidence_and_a_real_artifact(tmp_path: Path) -> None:
    code = """
import json, os, pathlib, sys
request = json.loads(sys.stdin.readline())
workspace = pathlib.Path(os.environ['MTE_TRIAL_DIRECTORY'])
(workspace / 'evidence.txt').write_text(request['case']['task'])
print(json.dumps({'protocol': 1, 'evidence': {
  'public': {'output': {'seed': request['trial']['seed']}},
  'checks': [{'name': 'job-check', 'passed': True, 'reason': 'job completed'}],
  'artifacts': [{'name': 'evidence', 'path': 'evidence.txt'}]
}}))
"""
    campaign = Campaign(
        name="job-proof",
        revision="1",
        cases=(Case(name="job", task="check me", required_checks=("job-check",)),),
        variants=(
            Variant(
                name="command",
                harness=OneShotCommandHarness(
                    argv=(sys.executable, "-c", code),
                    cwd=tmp_path,
                ),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    result = report.results[0]
    assert report.exit_code == 0
    assert result.passed is True
    artifact = next((tmp_path / "run").glob("trials/**/evidence.txt"))
    assert artifact.read_text(encoding="utf-8") == "check me"


def test_cancelling_a_one_shot_job_reaps_its_process_group(tmp_path: Path) -> None:
    code = """
import json, os, pathlib, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
pathlib.Path(os.environ['MTE_TRIAL_DIRECTORY'], 'job-pids').write_text(
    json.dumps({'parent': os.getpid(), 'child': child.pid})
)
time.sleep(60)
"""

    async def exercise() -> None:
        key = TrialKey(
            campaign="job-cancel",
            revision="1",
            case="job",
            variant="command",
            sample=1,
        )
        trace = LocalTrace(key, 1, tmp_path / "events.jsonl")
        harness = OneShotCommandHarness(argv=(sys.executable, "-c", code), cwd=tmp_path)
        task = asyncio.create_task(
            harness.run(
                Case(name="job", task="wait", required_checks=("done",)),
                Trial(key=key, attempt=1, directory=tmp_path, seed=1),
                trace,
            )
        )
        pid_path = tmp_path / "job-pids"
        for _ in range(100):
            if pid_path.exists():
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        trace.close()

    asyncio.run(exercise())
    pids = json.loads((tmp_path / "job-pids").read_text(encoding="utf-8"))
    for pid in pids.values():
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_command_job_does_not_inherit_unrelated_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-leak")
    code = """
import json, os, sys
json.loads(sys.stdin.readline())
print(json.dumps({'protocol': 1, 'evidence': {
  'public': {'output': {'secret': os.environ.get('UNRELATED_SECRET')}},
  'checks': [{'name': 'done', 'passed': True, 'reason': 'job ran'}]
}}))
"""
    campaign = Campaign(
        name="environment",
        revision="1",
        cases=(Case(name="job", task="run", required_checks=("done",)),),
        variants=(
            Variant(
                name="command",
                harness=OneShotCommandHarness(
                    argv=(sys.executable, "-c", code),
                    cwd=tmp_path,
                ),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    terminal = report.results[0].terminal
    assert isinstance(terminal, Completed)
    assert terminal.evidence.public.output == {"secret": None}


def test_command_job_stops_reading_at_the_authored_output_limit(tmp_path: Path) -> None:
    campaign = Campaign(
        name="output-limit",
        revision="1",
        cases=(Case(name="job", task="run", required_checks=("done",)),),
        variants=(
            Variant(
                name="command",
                harness=OneShotCommandHarness(
                    argv=(sys.executable, "-c", "print('x' * 10000)"),
                    cwd=tmp_path,
                    output_bytes=1000,
                ),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert terminal.failure.phase == "protocol"
    assert terminal.failure.message == "command job stdout exceeds 1000 bytes"


@pytest.mark.parametrize("inherit_pipes", [False, True])
def test_successful_job_reaps_background_descendants(tmp_path: Path, inherit_pipes: bool) -> None:
    code = """
import json, os, pathlib, subprocess, sys
child = subprocess.Popen(
    [sys.executable, '-c', 'import time; time.sleep(60)'],
    stdin=subprocess.DEVNULL,
    stdout=None if INHERIT_PIPES else subprocess.DEVNULL,
    stderr=None if INHERIT_PIPES else subprocess.DEVNULL,
)
pathlib.Path(os.environ['MTE_TRIAL_DIRECTORY'], 'child-pid').write_text(str(child.pid))
print(json.dumps({'protocol': 1, 'evidence': {
  'checks': [{'name': 'done', 'passed': True, 'reason': 'parent completed'}]
}}))
"""
    campaign = Campaign(
        name="descendant",
        revision="1",
        cases=(Case(name="job", task="run", required_checks=("done",)),),
        variants=(
            Variant(
                name="command",
                harness=OneShotCommandHarness(
                    argv=(sys.executable, "-c", f"INHERIT_PIPES = {inherit_pipes!r}\n" + code),
                    cwd=tmp_path,
                    timeout_seconds=0.5,
                ),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    assert report.exit_code == 0
    child_pid = int(next((tmp_path / "run").glob("trials/**/child-pid")).read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_job_timeout_kills_a_child_that_ignores_sigint_after_parent_exit(tmp_path: Path) -> None:
    code = """
import os, pathlib, signal, subprocess, sys, time
workspace = pathlib.Path(os.environ['MTE_TRIAL_DIRECTORY'])
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
child = subprocess.Popen(
    [sys.executable, '-c', '''
import pathlib, signal, sys, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
pathlib.Path(sys.argv[1]).write_text('ready')
time.sleep(60)
''', str(workspace / 'child-ready')],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
(workspace / 'child-pid').write_text(str(child.pid))
while not (workspace / 'child-ready').exists():
    time.sleep(0.01)
time.sleep(60)
"""
    campaign = Campaign(
        name="stubborn-descendant",
        revision="1",
        cases=(Case(name="job", task="wait", required_checks=("done",)),),
        variants=(
            Variant(
                name="command",
                harness=OneShotCommandHarness(
                    argv=(sys.executable, "-c", code), cwd=tmp_path, timeout_seconds=0.5
                ),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert "exceeded 0.5 seconds" in terminal.failure.message
    child_pid = int(next((tmp_path / "run").glob("trials/**/child-pid")).read_text())
    try:
        for _ in range(100):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            pytest.fail(f"job returned while SIGINT-ignoring child {child_pid} was still running")
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_changing_a_command_limit_rejects_old_receipts(tmp_path: Path) -> None:
    code = """
import json, sys
json.loads(sys.stdin.readline())
print(json.dumps({'protocol': 1, 'evidence': {
  'checks': [{'name': 'done', 'passed': True, 'reason': 'completed'}]
}}))
"""

    def campaign_with_limit(output_bytes: int) -> Campaign:
        return Campaign(
            name="limits",
            revision="1",
            cases=(Case(name="job", task="run", required_checks=("done",)),),
            variants=(
                Variant(
                    name="command",
                    harness=OneShotCommandHarness(
                        argv=(sys.executable, "-c", code),
                        cwd=tmp_path,
                        output_bytes=output_bytes,
                    ),
                ),
            ),
        )

    asyncio.run(run(campaign_with_limit(10_000), directory=tmp_path / "run"))

    with pytest.raises(ValueError, match="different campaign definition"):
        asyncio.run(run(campaign_with_limit(20_000), directory=tmp_path / "run"))


def test_relative_run_directory_is_absolute_inside_a_different_child_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_cwd = tmp_path / "child"
    child_cwd.mkdir()
    monkeypatch.chdir(tmp_path)
    code = """
import json, os, pathlib, sys
json.loads(sys.stdin.readline())
workspace = pathlib.Path(os.environ['MTE_TRIAL_DIRECTORY'])
(workspace / 'proof.txt').write_text(str(workspace.is_absolute()))
print(json.dumps({'protocol': 1, 'evidence': {
  'checks': [{'name': 'done', 'passed': True, 'reason': 'completed'}],
  'artifacts': [{'name': 'proof', 'path': 'proof.txt'}]
}}))
"""
    campaign = Campaign(
        name="relative-directory",
        revision="1",
        cases=(Case(name="job", task="run", required_checks=("done",)),),
        variants=(
            Variant(
                name="command",
                harness=OneShotCommandHarness(
                    argv=(sys.executable, "-c", code),
                    cwd=child_cwd,
                ),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=Path("relative-run")))

    assert report.exit_code == 0
    proof = next((tmp_path / "relative-run").glob("trials/**/proof.txt"))
    assert proof.read_text(encoding="utf-8") == "True"


def test_job_timeout_covers_backpressure_while_writing_the_request(tmp_path: Path) -> None:
    code = """
import os, pathlib, time
pathlib.Path(os.environ['MTE_TRIAL_DIRECTORY'], 'blocked-pid').write_text(str(os.getpid()))
time.sleep(60)
"""
    campaign = Campaign(
        name="write-timeout",
        revision="1",
        cases=(
            Case(
                name="job",
                task="run",
                data={"large": "x" * 2_000_000},
                required_checks=("done",),
            ),
        ),
        variants=(
            Variant(
                name="command",
                harness=OneShotCommandHarness(
                    argv=(sys.executable, "-c", code),
                    cwd=tmp_path,
                    timeout_seconds=0.1,
                ),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert terminal.failure.phase == "execute"
    assert "exceeded 0.1 seconds" in terminal.failure.message
    pid = int(next((tmp_path / "run").glob("trials/**/blocked-pid")).read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
