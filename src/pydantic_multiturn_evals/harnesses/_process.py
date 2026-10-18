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
    error_signal: signal.Signals = signal.SIGTERM,
) -> tuple[bytes, bytes]:
    stdin = process.stdin
    stdout = process.stdout
    stderr = process.stderr
    if stdin is None or stdout is None or stderr is None:
        raise RuntimeError("command has no process pipe")
    stdout_task = asyncio.create_task(_read_at_most(stdout, stdout_bytes, "stdout"))
    stderr_task = asyncio.create_task(_read_at_most(stderr, stderr_bytes, "stderr"))
    write_task = asyncio.create_task(_write_request(stdin, request))
    exit_task = asyncio.create_task(_finish_process_group(process))
    try:
        output, error_output, _, _ = await asyncio.gather(
            stdout_task, stderr_task, write_task, exit_task
        )
        return output, error_output
    except BaseException as error:
        first_signal = signal.SIGINT if isinstance(error, asyncio.CancelledError) else error_signal

        async def cleanup() -> None:
            tasks = (stdout_task, stderr_task, write_task, exit_task)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await _stop_after_failure(process, stdout, stderr, first_signal, cancel_grace_seconds)

        pending = asyncio.create_task(cleanup())
        while not pending.done():
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                continue
        pending.result()
        raise


async def _write_request(stdin: asyncio.StreamWriter, request: bytes) -> None:
    try:
        stdin.write(request)
        await stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        # A command may exit without reading its request; its output and exit code still decide.
        pass
    finally:
        stdin.close()


async def _stop_after_failure(
    process: asyncio.subprocess.Process,
    stdout: asyncio.StreamReader,
    stderr: asyncio.StreamReader,
    first_signal: signal.Signals,
    grace_seconds: float,
) -> None:
    # A reader that hit its limit stops consuming. Drain without retaining output so
    # full pipes cannot block cooperative cleanup or Process.wait() after a kill.
    readers = [asyncio.create_task(_discard_output(stream)) for stream in (stdout, stderr)]
    try:
        await stop_process_group(process, first_signal, grace_seconds)
    finally:
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)


async def _discard_output(stream: asyncio.StreamReader) -> None:
    while await stream.read(65_536):
        pass


async def _finish_process_group(process: asyncio.subprocess.Process) -> None:
    await _wait_for_parent_exit(process)
    terminate_process_group(process)
    await process.wait()


async def _wait_for_parent_exit(process: asyncio.subprocess.Process) -> None:
    # Process.wait() can wait for pipe EOF even after the parent has exited.
    while process.returncode is None:
        await asyncio.sleep(0.01)


def terminate_process_group(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        # macOS reports EPERM for a group whose only member is an unreaped zombie.
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
    except (ProcessLookupError, PermissionError):
        await process.wait()
        return
    try:
        async with asyncio.timeout(grace_seconds):
            await _wait_for_parent_exit(process)
    except TimeoutError:
        pass
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
