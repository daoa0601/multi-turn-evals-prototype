"""One-instance adapter for the official SWE-bench evaluator."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pydantic import JsonValue

from pydantic_multiturn_evals.evaluation import HarnessError
from pydantic_multiturn_evals.harnesses._process import (
    ProcessOutputLimitError,
    collect_process_output,
    command_environment,
)
from pydantic_multiturn_evals.model import (
    Artifact,
    Case,
    ComponentDefinition,
    Evidence,
    Name,
    NativeCheck,
    PublicOutcome,
    StrictModel,
    Text,
    TraceRecorder,
    Trial,
)


class SweBenchTask(StrictModel):
    case: Name
    dataset: Text
    split: Text = "test"
    instance: Text
    revision: Text


class _SolverReply(StrictModel):
    protocol: int
    patch: str
    output: JsonValue = None
    trajectory: JsonValue = None


@dataclass(frozen=True, slots=True)
class SweBenchHarness:
    """Produce one patch, then grade it in a fresh official evaluator environment."""

    evaluator_argv: tuple[str, ...]
    package_version: str
    tasks: tuple[SweBenchTask, ...]
    solver_argv: tuple[str, ...] | None = None
    solver_cwd: Path = Path(".")
    solver_name: str = "solver"
    solver_revision: str | None = None
    check: str = "swebench.resolved"
    solver_timeout_seconds: float = 900
    verifier_timeout_seconds: float = 1800
    cleanup_timeout_seconds: float = 10
    output_bytes: int = 8_000_000
    stderr_bytes: int = 2_000_000
    solver_environment: tuple[str, ...] = ()
    evaluator_environment: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        cases = [task.case for task in self.tasks]
        if not self.evaluator_argv:
            raise ValueError("SWE-bench evaluator command must not be empty")
        if not self.tasks:
            raise ValueError("SweBenchHarness needs at least one task binding")
        if len(cases) != len(set(cases)):
            raise ValueError("SWE-bench task case bindings must be unique")
        if self.solver_argv == ():
            raise ValueError("SWE-bench solver command must not be empty")
        if self.solver_argv is not None and not self.solver_revision:
            raise ValueError("SWE-bench external solver needs a semantic revision")
        if self.solver_timeout_seconds <= 0 or self.verifier_timeout_seconds <= 0:
            raise ValueError("SWE-bench timeouts must be positive")
        if self.cleanup_timeout_seconds <= 0:
            raise ValueError("SWE-bench cleanup timeout must be positive")
        if self.output_bytes < 1 or self.stderr_bytes < 1:
            raise ValueError("SWE-bench output limits must be positive")

    @property
    def component(self) -> ComponentDefinition:
        return ComponentDefinition(
            kind="swebench",
            version="1",
            config={
                "package_version": self.package_version,
                "tasks": [task.model_dump(mode="json") for task in self.tasks],
                "solver_name": self.solver_name,
                "solver_revision": self.solver_revision,
                "check": self.check,
                "solver_timeout_seconds": self.solver_timeout_seconds,
                "verifier_timeout_seconds": self.verifier_timeout_seconds,
                "cleanup_timeout_seconds": self.cleanup_timeout_seconds,
                "output_bytes": self.output_bytes,
                "stderr_bytes": self.stderr_bytes,
                "solver_environment": list(self.solver_environment),
                "evaluator_environment": list(self.evaluator_environment),
            },
        )

    async def run(self, case: Case, trial: Trial, trace: TraceRecorder) -> Evidence:
        del trace
        task = self._task(case.name)
        run_id = f"mte-{uuid4().hex}"
        state_path = trial.directory / "swebench-state.json"
        _write_json(
            state_path,
            {
                "protocol": 1,
                "phase": "solver" if self.solver_argv is not None else "gold",
                "instance": task.instance,
                "run_id": run_id,
            },
        )
        artifacts: list[Artifact] = []
        public: JsonValue = None
        prediction_path: Path | None = None
        if self.solver_argv is not None:
            solver = await self._solve(case, task, trial)
            public = solver.output
            patch_path = trial.directory / "model.patch"
            solver_path = trial.directory / "solver-result.json"
            patch_path.write_text(solver.patch, encoding="utf-8")
            _write_json(solver_path, solver.model_dump(mode="json"))
            prediction_path = trial.directory / "prediction.json"
            _write_json(
                prediction_path,
                [
                    {
                        "instance_id": task.instance,
                        "model_patch": solver.patch,
                        "model_name_or_path": self.solver_name,
                    }
                ],
            )
            artifacts.extend(
                (
                    Artifact(name="swebench-patch", path=patch_path.name, media_type="text/x-diff"),
                    Artifact(name="swebench-solver", path=solver_path.name),
                    Artifact(name="swebench-prediction", path=prediction_path.name),
                )
            )
            if not solver.patch.strip():
                _write_json(
                    state_path,
                    {
                        "protocol": 1,
                        "phase": "finished",
                        "instance": task.instance,
                        "run_id": run_id,
                        "resolved": False,
                        "verifier": "not-run-empty-patch",
                    },
                )
                return Evidence(
                    public=PublicOutcome(output=public),
                    checks=(
                        NativeCheck(
                            name=self.check,
                            passed=False,
                            reason="solver returned no patch",
                        ),
                    ),
                    artifacts=tuple(artifacts),
                )
        verifier_directory = Path(
            tempfile.mkdtemp(prefix="swebench-verifier-", dir=trial.directory)
        )
        report = await self._verify(task, verifier_directory, run_id, prediction_path)
        report_path, resolved = report
        artifacts.extend(_verifier_artifacts(trial.directory, verifier_directory, report_path))
        _write_json(
            state_path,
            {
                "protocol": 1,
                "phase": "finished",
                "instance": task.instance,
                "run_id": run_id,
                "resolved": resolved,
                "report": str(report_path.relative_to(trial.directory)),
            },
        )
        return Evidence(
            public=PublicOutcome(output=public),
            checks=(
                NativeCheck(
                    name=self.check,
                    passed=resolved,
                    reason=f"official SWE-bench report resolved={str(resolved).lower()}",
                ),
            ),
            artifacts=tuple(artifacts),
        )

    async def _solve(self, case: Case, task: SweBenchTask, trial: Trial) -> _SolverReply:
        assert self.solver_argv is not None
        request = {
            "protocol": 1,
            "task": case.task,
            "instance": task.instance,
            "trial": {
                "key": trial.key.model_dump(mode="json"),
                "attempt": trial.attempt,
                "seed": trial.seed,
            },
        }
        stdout, stderr, returncode = await _command(
            self.solver_argv,
            cwd=self.solver_cwd,
            environment=command_environment(self.solver_environment, trial.directory),
            request=(json.dumps(request) + "\n").encode(),
            timeout_seconds=self.solver_timeout_seconds,
            output_bytes=self.output_bytes,
            stderr_bytes=self.stderr_bytes,
            label="SWE-bench solver",
        )
        (trial.directory / "solver.stderr.txt").write_bytes(stderr)
        if returncode != 0:
            detail = stderr.decode(errors="replace").strip()
            raise HarnessError(
                "execute",
                f"SWE-bench solver exited with code {returncode}"
                + (f": {detail}" if detail else ""),
            )
        try:
            reply = _SolverReply.model_validate_json(stdout)
        except Exception as error:
            raise HarnessError("protocol", f"invalid SWE-bench solver result: {error}") from error
        if reply.protocol != 1:
            raise HarnessError("protocol", "SWE-bench solver must return protocol 1")
        return reply

    async def _verify(
        self,
        task: SweBenchTask,
        directory: Path,
        run_id: str,
        prediction: Path | None,
    ) -> tuple[Path, bool]:
        summary = directory / "swebench-summary"
        command = [
            *self.evaluator_argv,
            "eval",
            task.dataset,
            "--run-id",
            run_id,
            "--instance",
            task.instance,
            "--split",
            task.split,
            "--workers",
            "1",
            "--timeout",
            str(int(self.verifier_timeout_seconds)),
            "--report-dir",
            str(summary),
        ]
        command.extend(("--gold",) if prediction is None else ("--predictions", str(prediction)))
        environment = command_environment(self.evaluator_environment, directory)
        try:
            stdout, stderr, returncode = await _command(
                tuple(command),
                cwd=directory,
                environment=environment,
                request=b"",
                timeout_seconds=self.verifier_timeout_seconds + 60,
                output_bytes=self.output_bytes,
                stderr_bytes=self.stderr_bytes,
                label="SWE-bench verifier",
            )
        except BaseException:
            await self._cleanup_verifier(task.instance, run_id, directory, environment)
            raise
        (directory / "swebench.stdout.txt").write_bytes(stdout)
        (directory / "swebench.stderr.txt").write_bytes(stderr)
        report = _find_instance_report(
            directory / "logs" / "run_evaluation" / run_id, task.instance
        )
        if report is None:
            detail = stderr.decode(errors="replace").strip()
            raise HarnessError(
                "verify",
                f"SWE-bench verifier exited with code {returncode} without an instance report"
                + (f": {detail}" if detail else ""),
            )
        try:
            value = json.loads(report.read_text(encoding="utf-8"))[task.instance]["resolved"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise HarnessError("protocol", f"invalid SWE-bench report {report}") from error
        if not isinstance(value, bool):
            raise HarnessError("protocol", f"SWE-bench report {report} has no boolean resolved")
        return report, value

    async def _cleanup_verifier(
        self,
        instance: str,
        run_id: str,
        directory: Path,
        environment: dict[str, str],
    ) -> None:
        async def cleanup() -> None:
            status: dict[str, JsonValue]
            try:
                async with asyncio.timeout(self.cleanup_timeout_seconds):
                    removed = await _remove_verifier_containers(instance, run_id, environment)
                status = {"status": "completed", "removed": list(removed)}
            except Exception as error:
                status = {
                    "status": "failed",
                    "message": str(error) or type(error).__name__,
                }
            try:
                _write_json(directory / "swebench-cleanup.json", status)
            except OSError:
                # An unwritable diagnostic must not replace the original verifier failure.
                pass

        pending = asyncio.create_task(cleanup())
        # A second cancellation must not release the trial's slot before bounded cleanup finishes.
        while not pending.done():
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                continue
        pending.result()

    def _task(self, case: str) -> SweBenchTask:
        for task in self.tasks:
            if task.case == case:
                return task
        raise HarnessError("prepare", f"no SWE-bench task is bound to case {case!r}")


async def _command(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    environment: dict[str, str],
    request: bytes,
    timeout_seconds: float,
    output_bytes: int,
    stderr_bytes: int,
    label: str,
) -> tuple[bytes, bytes, int]:
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except Exception as error:
        raise HarnessError("start", f"could not start {label}: {error}") from error
    try:
        async with asyncio.timeout(timeout_seconds):
            stdout, stderr = await collect_process_output(
                process,
                request,
                stdout_bytes=output_bytes,
                stderr_bytes=stderr_bytes,
            )
    except TimeoutError as error:
        raise HarnessError("execute", f"{label} exceeded {timeout_seconds:g} seconds") from error
    except ProcessOutputLimitError as error:
        raise HarnessError("protocol", f"{label} {error}") from error
    return stdout, stderr, process.returncode or 0


def _find_instance_report(root: Path, instance: str) -> Path | None:
    found: list[Path] = []
    for path in root.rglob("report.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and isinstance(value.get(instance), dict):
            found.append(path)
    if len(found) > 1:
        raise HarnessError("protocol", "SWE-bench wrote multiple instance reports")
    return found[0] if found else None


async def _remove_verifier_containers(
    instance: str, run_id: str, environment: dict[str, str]
) -> tuple[str, ...]:
    executable = shutil.which("docker", path=environment.get("PATH"))
    if executable is None:
        raise FileNotFoundError("Docker executable is unavailable for SWE-bench cleanup")

    async def docker(*arguments: str) -> bytes:
        process = await asyncio.create_subprocess_exec(
            executable,
            *arguments,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stdout, _ = await collect_process_output(
            process, b"", stdout_bytes=65_536, stderr_bytes=65_536, cancel_grace_seconds=0
        )
        if process.returncode != 0:
            raise RuntimeError(f"Docker cleanup exited with code {process.returncode}")
        return stdout

    # SWE-bench may append a timestamp after a Docker name conflict. Both names belong
    # to this unique attempt; similar prefixes and other run IDs do not.
    name = re.escape(f"sweb.eval.{instance.lower()}.{run_id}") + r"(\.[0-9]+)?"
    stdout = await docker("ps", "--all", "--format", "{{.Names}}", "--filter", f"name=^/{name}$")
    containers = tuple(item for item in stdout.decode().splitlines() if re.fullmatch(name, item))
    if containers:
        await docker("rm", "--force", *containers)
    return containers


def _verifier_artifacts(root: Path, directory: Path, report: Path) -> tuple[Artifact, ...]:
    paths = [
        report,
        directory / "swebench.stdout.txt",
        directory / "swebench.stderr.txt",
    ]
    paths.extend(directory.rglob("run_instance.log"))
    paths.extend(directory.rglob("test_output.txt"))
    paths.extend(directory.glob("swebench-summary/*.json"))
    unique: list[Path] = []
    for path in paths:
        if path.is_file() and path not in unique:
            unique.append(path)
    return tuple(
        Artifact(name=f"swebench-{index:02d}-{path.name}", path=str(path.relative_to(root)))
        for index, path in enumerate(unique, start=1)
    )


def _write_json(path: Path, value: JsonValue) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
