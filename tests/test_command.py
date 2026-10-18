from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from pydantic_multiturn_evals import Case, Message, Trial, TrialKey
from pydantic_multiturn_evals.harnesses import CommandLimits, CommandTarget
from pydantic_multiturn_evals.tracing import LocalTrace

TARGET = Path(__file__).parents[1] / "examples" / "jsonl_target.py"


def test_invalid_json_names_the_protocol_failure(tmp_path: Path) -> None:
    async def exercise() -> None:
        trace = LocalTrace(_trial(tmp_path).key, 1, tmp_path / "events.jsonl")
        target = CommandTarget(
            argv=(sys.executable, str(TARGET), "invalid"),
            cwd=TARGET.parent,
            limits=CommandLimits(response_seconds=2, shutdown_seconds=1),
        )
        session = await target.open(_trial(tmp_path), trace)
        try:
            with pytest.raises(RuntimeError, match="target returned invalid JSON"):
                await session.send(Message(role="user", content="Help"))
        finally:
            await session.close()
            trace.close()

    asyncio.run(exercise())


def test_exit_keeps_stderr_out_of_the_failure_message(tmp_path: Path) -> None:
    script = tmp_path / "crashing_target.py"
    script.write_text(
        "import json, sys\n"
        "sys.stdin.readline()\n"
        "print(json.dumps({'protocol': 1, 'type': 'ready'}), flush=True)\n"
        "sys.stdin.readline()\n"
        "print('token=sk-live-do-not-log', file=sys.stderr, flush=True)\n"
        "raise SystemExit(3)\n",
        encoding="utf-8",
    )

    async def exercise() -> str:
        trace = LocalTrace(_trial(tmp_path).key, 1, tmp_path / "events.jsonl")
        target = CommandTarget(
            argv=(sys.executable, str(script)),
            cwd=tmp_path,
            limits=CommandLimits(response_seconds=2, shutdown_seconds=1),
        )
        session = await target.open(_trial(tmp_path), trace)
        try:
            with pytest.raises(RuntimeError) as raised:
                await session.send(Message(role="user", content="Help"))
            return str(raised.value)
        finally:
            await session.close()
            trace.close()

    message = asyncio.run(exercise())
    assert message.startswith("target exited with code 3")
    assert "sk-live" not in message
    assert "target.stderr.txt" in message
    assert "sk-live-do-not-log" in (tmp_path / "target.stderr.txt").read_text()


def test_timeout_reaps_the_persistent_process(tmp_path: Path) -> None:
    async def exercise() -> None:
        trace = LocalTrace(_trial(tmp_path).key, 1, tmp_path / "events.jsonl")
        target = CommandTarget(
            argv=(sys.executable, str(TARGET), "hang"),
            cwd=TARGET.parent,
            limits=CommandLimits(response_seconds=0.1, shutdown_seconds=0.1),
        )
        session = await target.open(_trial(tmp_path), trace)
        try:
            with pytest.raises(TimeoutError, match="did not return 'reply'"):
                await session.send(Message(role="user", content="Help"))
        finally:
            await session.close()
            trace.close()

    asyncio.run(exercise())
    pid = int((tmp_path / "target-pid").read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_close_reaps_a_child_after_the_target_process_exits(tmp_path: Path) -> None:
    async def exercise() -> None:
        trace = LocalTrace(_trial(tmp_path).key, 1, tmp_path / "events.jsonl")
        target = CommandTarget(
            argv=(sys.executable, str(TARGET), "orphan"),
            cwd=TARGET.parent,
            limits=CommandLimits(response_seconds=2, shutdown_seconds=1),
        )
        session = await target.open(_trial(tmp_path), trace)
        await session.close()
        trace.close()

    asyncio.run(exercise())
    child_pid = int((tmp_path / "target-child-pid").read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_turn_timeout_covers_backpressure_while_writing_to_target(tmp_path: Path) -> None:
    code = """
import json, os, pathlib, sys, time
json.loads(sys.stdin.readline())
print(json.dumps({'protocol': 1, 'type': 'ready'}), flush=True)
pathlib.Path(os.environ['MTE_TRIAL_DIRECTORY'], 'blocked-pid').write_text(str(os.getpid()))
time.sleep(60)
"""

    async def exercise() -> None:
        trace = LocalTrace(_trial(tmp_path).key, 1, tmp_path / "events.jsonl")
        target = CommandTarget(
            argv=(sys.executable, "-c", code),
            cwd=tmp_path,
            limits=CommandLimits(response_seconds=0.1, shutdown_seconds=0.1),
        )
        session = await target.open(_trial(tmp_path), trace)
        try:
            with pytest.raises(TimeoutError, match="did not return 'reply'"):
                await session.send(Message(role="user", content="x" * 2_000_000))
        finally:
            await session.close()
            trace.close()

    asyncio.run(exercise())
    pid = int((tmp_path / "blocked-pid").read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def _case() -> Case:
    return Case(name="process", task="Help", required_checks=("transport",))


def _trial(path: Path) -> Trial:
    return Trial(
        key=TrialKey(
            campaign="test",
            revision="1",
            case="process",
            variant="command",
            sample=1,
        ),
        attempt=1,
        directory=path,
        seed=1,
    )
