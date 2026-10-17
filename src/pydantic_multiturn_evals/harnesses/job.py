"""A one-request command harness for external jobs and benchmark runners."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from pydantic_multiturn_evals.evaluation import HarnessError
from pydantic_multiturn_evals.harnesses._process import (
    ProcessOutputLimitError,
    collect_process_output,
    command_environment,
)
from pydantic_multiturn_evals.model import Case, ComponentDefinition, Evidence, TraceRecorder, Trial


@dataclass(frozen=True, slots=True)
class OneShotCommandHarness:
    argv: tuple[str, ...]
    cwd: Path
    timeout_seconds: float = 300
    output_bytes: int = 4_000_000
    stderr_bytes: int = 100_000
    inherit_environment: tuple[str, ...] = ()

    @property
    def component(self) -> ComponentDefinition:
        return ComponentDefinition(
            kind="command-job",
            version="1",
            config={
                "argv": list(self.argv),
                "cwd": str(self.cwd),
                "inherit_environment": list(self.inherit_environment),
                "timeout_seconds": self.timeout_seconds,
                "output_bytes": self.output_bytes,
                "stderr_bytes": self.stderr_bytes,
            },
        )

    async def run(self, case: Case, trial: Trial, trace: TraceRecorder) -> Evidence:
        try:
            environment = command_environment(self.inherit_environment, trial.directory)
            process = await asyncio.create_subprocess_exec(
                *self.argv,
                cwd=self.cwd,
                env=environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except Exception as error:
            raise HarnessError("start", str(error)) from error
        request = {
            "protocol": 1,
            "case": case.model_dump(mode="json"),
            "trial": {
                "key": trial.key.model_dump(mode="json"),
                "attempt": trial.attempt,
                "seed": trial.seed,
            },
        }
        try:
            async with asyncio.timeout(self.timeout_seconds):
                stdout, stderr = await collect_process_output(
                    process,
                    (json.dumps(request) + "\n").encode(),
                    stdout_bytes=self.output_bytes,
                    stderr_bytes=self.stderr_bytes,
                )
        except TimeoutError as error:
            raise HarnessError(
                "execute",
                f"command job exceeded {self.timeout_seconds:g} seconds",
            ) from error
        except ProcessOutputLimitError as error:
            message = (
                str(error)
                .replace("command stdout", "command job stdout")
                .replace("command stderr", "command job stderr")
            )
            raise HarnessError("protocol", message) from error
        if process.returncode != 0:
            detail = stderr.decode(errors="replace").strip()
            raise HarnessError(
                "execute",
                f"command job exited with code {process.returncode}"
                + (f": {detail}" if detail else ""),
            )
        try:
            response = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise HarnessError(
                "protocol",
                f"command job returned invalid JSON: {error.msg}",
            ) from error
        if not isinstance(response, dict) or response.get("protocol") != 1:
            raise HarnessError("protocol", "command job must return a protocol 1 JSON object")
        try:
            evidence = Evidence.model_validate(response.get("evidence"))
        except ValidationError as error:
            raise HarnessError("protocol", f"invalid command job evidence: {error}") from error
        return evidence
