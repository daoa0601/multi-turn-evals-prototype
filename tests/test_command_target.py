from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from pydantic_multiturn_evals.models import (
    CaseKey,
    CommandLimits,
    CommandTargetSpec,
    ConversationView,
    SessionContext,
    SessionOutcome,
    TargetCompletion,
    Transcript,
    TurnLimitReached,
    UserTurn,
)
from pydantic_multiturn_evals.spec import LoadedTarget
from pydantic_multiturn_evals.targets import build_target

ROOT = Path(__file__).parents[1]
HARNESS = ROOT / "tests" / "helpers" / "command_harness.py"


def command_target(
    *,
    mode: str = "ok",
    pid_file: Path | None = None,
    stdout_limit: int = 2048,
):
    environment = ["PATH", "HARNESS_MODE", "HARNESS_VARIANT"]
    if pid_file is not None:
        environment.append("HARNESS_PID_FILE")
        os.environ["HARNESS_PID_FILE"] = str(pid_file)
    os.environ["HARNESS_MODE"] = mode
    os.environ["HARNESS_VARIANT"] = "example"
    spec = CommandTargetSpec(
        version=1,
        name="example-command",
        kind="command",
        argv=(sys.executable, str(HARNESS)),
        cwd=ROOT,
        inherit_env=tuple(environment),
        limits=CommandLimits(
            startup_seconds=1,
            turn_seconds=1,
            shutdown_seconds=0.2,
            stdout_bytes_per_message=stdout_limit,
            stderr_bytes_per_session=2048,
        ),
    )
    return build_target(LoadedTarget(spec=spec, source_directory=ROOT))


def context() -> SessionContext:
    return SessionContext(
        suite_name="test-suite",
        target_name="example-command",
        target_kind="command",
        target_version=1,
        key=CaseKey(scenario_id="help", repeat_index=1),
        run_id="run-1",
    )


def view(*, exchanges=()) -> ConversationView:
    return ConversationView(
        run_id="run-1",
        scenario_id="help",
        exchanges=exchanges,
        pending_user=UserTurn(content="Please help."),
    )


def test_command_target_uses_one_jsonl_session_and_returns_evidence() -> None:
    async def exercise() -> None:
        target = command_target()
        async with target.session(context()) as session:
            first = await session.reply(view())
            second = await session.reply(view())
            completion = await session.finish(
                SessionOutcome(
                    transcript=Transcript(exchanges=()),
                    decisions=(),
                    termination=TurnLimitReached(limit=2),
                )
            )

        assert first.assistant_text == "example reply 1"
        assert second.assistant_text == "example reply 2"
        assert first.evidence["message_count"] == 1
        assert first.session_id == second.session_id
        assert completion == TargetCompletion(details={"adapter_finished": True})

    asyncio.run(exercise())


def test_command_protocol_rejects_version_one() -> None:
    async def exercise() -> None:
        target = command_target(mode="version-one")
        with pytest.raises(RuntimeError, match="invalid ready message"):
            async with target.session(context()):
                pass

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("malformed", "invalid JSON"),
        ("exit", "status 23"),
        ("oversized", "byte limit"),
        ("wrong-id", "expected 1"),
    ],
)
def test_command_protocol_failures_are_actionable(mode: str, message: str) -> None:
    async def exercise() -> None:
        target = command_target(mode=mode, stdout_limit=512)
        with pytest.raises(RuntimeError, match=message):
            async with target.session(context()) as session:
                await session.reply(view())

    asyncio.run(exercise())


def test_stderr_is_drained_and_failure_text_stays_out_of_the_exception() -> None:
    async def exercise() -> None:
        draining_target = command_target(mode="heavy-stderr")
        async with draining_target.session(context()) as session:
            assert (await session.reply(view())).assistant_text == "example reply 1"

        failing_target = command_target(mode="exit")
        with pytest.raises(RuntimeError) as failure:
            async with failing_target.session(context()) as session:
                await session.reply(view())
        assert "command harness failed" not in str(failure.value)
        assert "command harness failed" in failing_target.failure_evidence()[0].stderr

    asyncio.run(exercise())


def test_shutdown_bound_includes_descendants_holding_stderr() -> None:
    async def exercise() -> float:
        target = command_target(mode="descendant-stderr")
        started = asyncio.get_running_loop().time()
        async with target.session(context()) as session:
            await session.finish(
                SessionOutcome(
                    transcript=Transcript(exchanges=()),
                    decisions=(),
                    termination=TurnLimitReached(limit=1),
                )
            )
            await asyncio.sleep(0.05)
        return asyncio.get_running_loop().time() - started

    assert asyncio.run(exercise()) < 0.8


def test_cancelling_a_turn_reaps_the_process(tmp_path: Path) -> None:
    async def exercise() -> int:
        pid_file = tmp_path / "pid"
        target = command_target(mode="hang", pid_file=pid_file)
        pid = -1
        with pytest.raises(TimeoutError):
            async with target.session(context()) as session:
                pid = int(pid_file.read_text(encoding="utf-8"))
                await asyncio.wait_for(session.reply(view()), timeout=0.05)
        return pid

    pid = asyncio.run(exercise())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
