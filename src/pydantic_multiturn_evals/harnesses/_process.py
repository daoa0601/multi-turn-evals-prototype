"""Shared process boundary rules."""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Sequence
from pathlib import Path

TRIAL_DIRECTORY_ENV = "MTE_TRIAL_DIRECTORY"


def command_environment(names: Sequence[str], trial_directory: Path) -> dict[str, str]:
    """Copy only explicitly named variables into an external harness."""

    if TRIAL_DIRECTORY_ENV in names:
        raise ValueError(f"{TRIAL_DIRECTORY_ENV} is managed by the campaign runner")
    values = inherited_environment(names)
    values[TRIAL_DIRECTORY_ENV] = str(trial_directory)
    return values


def inherited_environment(names: Sequence[str]) -> dict[str, str]:
    if len(names) != len(set(names)):
        raise ValueError("inherited environment variable names must be unique")
    invalid = [name for name in names if not name or "=" in name]
    if invalid:
        raise ValueError(f"invalid environment variable name: {', '.join(invalid)}")
    missing = [name for name in names if name not in os.environ]
    if missing:
        raise ValueError(f"environment variable is not set: {', '.join(missing)}")
    return {name: os.environ[name] for name in names}


class ProcessOutputLimitError(RuntimeError):
    pass


async def collect_process_output(
    process: asyncio.subprocess.Process,
    request: bytes,
    *,
    stdout_bytes: int,
    stderr_bytes: int,
    cancel_grace_seconds: float = 2,
) -> tuple[bytes, bytes]:
    stdin = process.stdin
    stdout = process.stdout
    stderr = process.stderr
    if stdin is None or stdout is None or stderr is None:
        raise RuntimeError("command has no process pipe")
    stdout_task = asyncio.create_task(_read_at_most(stdout, stdout_bytes, "stdout"))
    stderr_task = asyncio.create_task(_read_at_most(stderr, stderr_bytes, "stderr"))
    try:
        stdin.write(request)
        await stdin.drain()
        stdin.close()
        output, error_output = await asyncio.gather(stdout_task, stderr_task)
        await process.wait()
        terminate_process_group(process)
        return output, error_output
    except asyncio.CancelledError:
        await asyncio.shield(stop_process_group(process, signal.SIGINT, cancel_grace_seconds))
        for task in (stdout_task, stderr_task):
            task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise
    except BaseException:
        await asyncio.shield(stop_process_group(process, signal.SIGTERM, cancel_grace_seconds))
        for task in (stdout_task, stderr_task):
            task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise


def terminate_process_group(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


async def stop_process_group(
    process: asyncio.subprocess.Process,
    first_signal: signal.Signals,
    grace_seconds: float,
) -> None:
    """Give a command a bounded cleanup window, then kill its process group."""

    if grace_seconds < 0:
        raise ValueError("process cancellation grace must not be negative")
    try:
        os.killpg(process.pid, first_signal)
    except ProcessLookupError:
        await process.wait()
        return
    try:
        async with asyncio.timeout(grace_seconds):
            await process.wait()
    except TimeoutError:
        terminate_process_group(process)
        await process.wait()


async def _read_at_most(
    stream: asyncio.StreamReader,
    limit: int,
    name: str,
) -> bytes:
    if limit < 1:
        raise ValueError(f"{name} limit must be positive")
    chunks: list[bytes] = []
    total = 0
    while chunk := await stream.read(min(65_536, limit - total + 1)):
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            raise ProcessOutputLimitError(f"command {name} exceeds {limit} bytes")
    return b"".join(chunks)
