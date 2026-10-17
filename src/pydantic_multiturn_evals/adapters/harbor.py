"""One-trial adapter for the Harbor evaluation CLI."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from pydantic_multiturn_evals.adapters._harbor_trajectory import (
    AtifTrajectory,
    import_trajectory,
    parse_trajectory,
)
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
    LogFact,
    Message,
    Name,
    NativeCheck,
    PublicOutcome,
    ResourceFact,
    StrictModel,
    Text,
    TraceRecorder,
    Trial,
)

_GLOB_CHARACTERS = "*?["
# Matched against names with separators, whitespace, and case removed, so
# "API-KEY", " api key ", and "openai_api_key" are all rejected.
_CREDENTIAL_NAMES = frozenset(
    {"apikey", "accesstoken", "clientsecret", "authorization", "password", "secret", "token"}
)
_RESERVED_ARGUMENTS = {
    "logs_dir",
    "logger",
    "model_name",
    "task_dir",
    "trial_paths",
    "mcp_servers",
}
_MAX_TRAJECTORY_SEGMENTS = 32
_EXCEPTION_TYPE_LIMIT = 120
_EXCEPTION_MESSAGE_LIMIT = 1000


class HarborLocalTask(StrictModel):
    kind: Literal["local"] = "local"
    case: Name
    path: Path
    revision: Text


class HarborDatasetTask(StrictModel):
    kind: Literal["dataset"] = "dataset"
    case: Name
    dataset: Text
    task: Text | None = None
    revision: Text


HarborTask: TypeAlias = Annotated[
    HarborLocalTask | HarborDatasetTask,
    Field(discriminator="kind"),
]


class _HarborModelInfo(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    name: str
    provider: str | None = None


class _HarborAgentInfo(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    name: str
    version: str
    model_info: _HarborModelInfo | None = None


class _HarborAgentResult(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    n_input_tokens: JsonValue = None
    n_cache_tokens: JsonValue = None
    n_output_tokens: JsonValue = None
    cost_usd: JsonValue = None


class _HarborVerifier(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    rewards: JsonValue = None


class _HarborExceptionInfo(BaseModel):
    """Only the safe identity of a native failure.

    Tracebacks and arbitrary metadata stay in the local Harbor artifacts; they
    are never surfaced to judges through a failure message.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    exception_type: str | None = None
    exception_message: str | None = None


class _HarborResult(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    task_name: str
    trial_name: str
    agent_info: _HarborAgentInfo
    agent_result: _HarborAgentResult | None = None
    verifier_result: _HarborVerifier | None = None
    exception_info: _HarborExceptionInfo | None = None


@dataclass(frozen=True, slots=True)
class HarborHarness:
    """Run exactly one selected Harbor task and translate its saved result.

    Imported dialogue is operational trace data. It reaches PublicOutcome, and
    therefore evaluators, only when public_trajectory opts in.
    """

    executable: Path
    harbor_version: str
    tasks: tuple[HarborTask, ...]
    agent: str = "oracle"
    agent_import_path: str | None = None
    agent_arguments: dict[str, JsonValue] = field(default_factory=dict)
    model: str | None = None
    environment: str = "docker"
    docker_platform: str | None = None
    reward: str = "reward"
    pass_threshold: float = 1.0
    check: str = "harbor.verified"
    timeout_seconds: float = 900
    cancel_grace_seconds: float = 20
    output_bytes: int = 1_000_000
    stderr_bytes: int = 1_000_000
    trajectory_bytes: int = 8_000_000
    inherit_environment: tuple[str, ...] = ()
    public_trajectory: bool = False
    _agent_arguments_json: str = field(init=False, repr=False, compare=False, default="")

    def __post_init__(self) -> None:
        cases = [task.case for task in self.tasks]
        if not self.tasks:
            raise ValueError("HarborHarness needs at least one task binding")
        if len(cases) != len(set(cases)):
            raise ValueError("Harbor task case bindings must be unique")
        if self.timeout_seconds <= 0 or self.cancel_grace_seconds < 0:
            raise ValueError("Harbor timeout must be positive and cancellation grace non-negative")
        if min(self.output_bytes, self.stderr_bytes, self.trajectory_bytes) < 1:
            raise ValueError("Harbor output limits must be positive")
        if self.docker_platform is not None and self.environment != "docker":
            raise ValueError("docker_platform requires the Docker environment")
        if (
            self.docker_platform is not None
            and "DOCKER_DEFAULT_PLATFORM" in self.inherit_environment
        ):
            raise ValueError("set docker_platform instead of inheriting DOCKER_DEFAULT_PLATFORM")
        if self.agent_import_path is not None and ":" not in self.agent_import_path:
            raise ValueError("agent_import_path must look like 'module.path:ClassName'")
        snapshot = _agent_arguments_snapshot(self.agent_arguments)
        object.__setattr__(self, "_agent_arguments_json", snapshot)
        arguments = json.loads(snapshot)
        object.__setattr__(self, "agent_arguments", arguments)
        credentials = _credential_paths(arguments)
        if credentials:
            raise ValueError(
                "Harbor agent credentials must come from the environment via inherit_environment, "
                f"not agent_arguments: {', '.join(credentials)}"
            )
        invalid = sorted(
            name
            for name in arguments
            if name in _RESERVED_ARGUMENTS or not name.strip() or "=" in name
        )
        if invalid:
            raise ValueError(f"Invalid or reserved Harbor agent options: {', '.join(invalid)}")
        for task in self.tasks:
            if (
                isinstance(task, HarborDatasetTask)
                and task.task is not None
                and any(character in task.task for character in _GLOB_CHARACTERS)
            ):
                raise ValueError(
                    f"Harbor dataset task bindings must name one task exactly: {task.task!r}"
                )

    @property
    def component(self) -> ComponentDefinition:
        return ComponentDefinition(
            kind="harbor",
            version="1",
            config={
                "harbor_version": self.harbor_version,
                "tasks": [_task_identity(task) for task in self.tasks],
                "agent": None if self.agent_import_path is not None else self.agent,
                "agent_import_path": self.agent_import_path,
                "agent_arguments": json.loads(self._agent_arguments_json),
                "model": self.model,
                "environment": self.environment,
                "docker_platform": self.docker_platform,
                "reward": self.reward,
                "pass_threshold": self.pass_threshold,
                "check": self.check,
                "timeout_seconds": self.timeout_seconds,
                "cancel_grace_seconds": self.cancel_grace_seconds,
                "output_bytes": self.output_bytes,
                "stderr_bytes": self.stderr_bytes,
                "trajectory_bytes": self.trajectory_bytes,
                "inherit_environment": list(self.inherit_environment),
                "public_trajectory": self.public_trajectory,
            },
        )

    async def run(self, case: Case, trial: Trial, trace: TraceRecorder) -> Evidence:
        task = self._task(case.name)
        jobs = trial.directory / "harbor"
        job_name = f"attempt-{trial.attempt:04d}"
        argv = self._command(task, jobs, job_name)
        state_path = trial.directory / "harbor-state.json"
        _write_json(
            state_path,
            {
                "protocol": 1,
                "phase": "starting",
                "job_directory": str(jobs / job_name),
                "task": task.model_dump(mode="json"),
            },
        )
        trace.record(ResourceFact(name="harbor-attempt", value=1, unit="trial"))
        try:
            environment = command_environment(self.inherit_environment, trial.directory)
            if self.docker_platform is not None:
                environment["DOCKER_DEFAULT_PLATFORM"] = self.docker_platform
            process = await asyncio.create_subprocess_exec(
                *argv,
                env=environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except Exception as error:
            raise HarnessError("start", f"could not start Harbor: {error}") from error
        try:
            async with asyncio.timeout(self.timeout_seconds):
                stdout, stderr = await collect_process_output(
                    process,
                    b"",
                    stdout_bytes=self.output_bytes,
                    stderr_bytes=self.stderr_bytes,
                    cancel_grace_seconds=self.cancel_grace_seconds,
                )
        except TimeoutError as error:
            raise HarnessError(
                "execute",
                f"Harbor exceeded {self.timeout_seconds:g} seconds",
            ) from error
        except ProcessOutputLimitError as error:
            raise HarnessError("protocol", f"Harbor {error}") from error

        stdout_path = trial.directory / "harbor.stdout.txt"
        stderr_path = trial.directory / "harbor.stderr.txt"
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        job_root = jobs / job_name
        result_path = self._result_path(job_root)
        agent_dir = self._trajectory_root(job_root, result_path)
        chain = _TrajectoryChain()

        def artifacts() -> tuple[Artifact, ...]:
            return _harbor_artifacts(
                trial.directory, result_path, stdout_path, stderr_path, chain.paths
            )

        def failure(messages: Sequence[Message] = ()) -> Evidence:
            return Evidence(public=self._public_outcome(messages), artifacts=artifacts())

        if result_path is None:
            messages = self._import_optional_trajectory(agent_dir, trial.directory, trace, chain)
            # Stderr can echo prompts or credentials; it stays in its artifact, not the message.
            message = f"Harbor exited with code {process.returncode} without a trial result"
            if stderr:
                message += f"; {len(stderr)} bytes of stderr in {stderr_path.name}"
            raise HarnessError("execute", message, evidence=failure(messages))

        # The saved result is authoritative: read it before the optional
        # trajectory so a native failure is never hidden by a damaged transcript.
        try:
            result = _HarborResult.model_validate_json(
                _read_native_file(result_path, trial.directory, self.output_bytes)
            )
        except (OSError, ValueError) as error:
            raise HarnessError(
                "protocol",
                f"invalid Harbor trial result {result_path}: {error}",
                evidence=failure(),
            ) from error

        starting = failure()
        self._require_task(task, result, starting)
        self._require_agent(result, starting)
        self._require_model(result, starting)
        if result.exception_info is not None:
            messages = self._import_optional_trajectory(agent_dir, trial.directory, trace, chain)
            raise HarnessError(
                "execute", _native_exception_detail(result), evidence=failure(messages)
            )

        if agent_dir is not None:
            try:
                _read_trajectory_chain(agent_dir, trial.directory, self.trajectory_bytes, chain)
            except (OSError, ValueError) as error:
                raise HarnessError(
                    "protocol",
                    f"invalid Harbor trajectory {error}",
                    evidence=failure(),
                ) from error
        messages = self._import_recorded_trajectory(chain, trial.directory, trace)
        reward = _selected_reward(result.verifier_result, self.reward, failure(messages))
        _record_reward_facts(trace, result.verifier_result)
        _record_usage_facts(trace, result.agent_result)
        check = NativeCheck(
            name=self.check,
            passed=reward >= self.pass_threshold,
            reason=(
                f"Harbor reward {self.reward}={reward:g}; pass threshold is {self.pass_threshold:g}"
            ),
        )
        _write_json(
            state_path,
            {
                "protocol": 1,
                "phase": "finished",
                "job_directory": str(jobs / job_name),
                "trial_result": str(result_path.relative_to(trial.directory)),
                "trajectory_segments": [
                    str(path.relative_to(trial.directory)) for path in chain.paths
                ],
                "reward": reward,
            },
        )
        return Evidence(
            public=PublicOutcome(
                messages=tuple(messages) if self.public_trajectory else (),
                output={"task_name": result.task_name, "trial_name": result.trial_name},
            ),
            checks=(check,),
            artifacts=artifacts(),
        )

    def _public_outcome(self, messages: Sequence[Message]) -> PublicOutcome:
        return PublicOutcome(messages=tuple(messages) if self.public_trajectory else ())

    def _import_optional_trajectory(
        self,
        agent_dir: Path | None,
        trial_root: Path,
        trace: TraceRecorder,
        chain: _TrajectoryChain,
    ) -> list[Message]:
        """Import a transcript that only supplements an already-decided failure."""

        if agent_dir is None:
            return []
        try:
            _read_trajectory_chain(agent_dir, trial_root, self.trajectory_bytes, chain)
        except (OSError, ValueError) as error:
            trace.record(
                LogFact(level="warning", message=f"Harbor trajectory import failed: {error}")
            )
        return self._import_recorded_trajectory(chain, trial_root, trace)

    def _import_recorded_trajectory(
        self,
        chain: _TrajectoryChain,
        trial_root: Path,
        trace: TraceRecorder,
    ) -> list[Message]:
        messages: list[Message] = []
        for path, trajectory in zip(chain.paths, chain.trajectories, strict=True):
            if trajectory is None:
                continue
            receipt = import_trajectory(
                trajectory,
                receipt=str(path.relative_to(trial_root)),
            )
            messages.extend(receipt.messages)
            for fact in receipt.facts:
                trace.record(fact)
        return messages

    def _task(self, case: str) -> HarborTask:
        for task in self.tasks:
            if task.case == case:
                return task
        raise HarnessError("prepare", f"no Harbor task is bound to case {case!r}")

    def _require_task(self, task: HarborTask, result: _HarborResult, evidence: Evidence) -> None:
        if (
            isinstance(task, HarborDatasetTask)
            and task.task is not None
            and result.task_name != task.task
        ):
            raise HarnessError(
                "protocol",
                f"Harbor returned task {result.task_name!r}, expected {task.task!r}",
                evidence=evidence,
            )

    def _require_agent(self, result: _HarborResult, evidence: Evidence) -> None:
        if self.agent_import_path is None and result.agent_info.name != self.agent:
            raise HarnessError(
                "protocol",
                f"Harbor returned agent {result.agent_info.name!r}, expected {self.agent!r}",
                evidence=evidence,
            )

    def _require_model(self, result: _HarborResult, evidence: Evidence) -> None:
        if self.model is None:
            return
        model = result.agent_info.model_info
        if model is None:
            raise HarnessError(
                "protocol",
                f"Harbor result has no model_info, expected model {self.model!r}",
                evidence=evidence,
            )
        names = {model.name}
        if model.provider is not None:
            names.add(f"{model.provider}/{model.name}")
        if self.model not in names:
            raise HarnessError(
                "protocol",
                f"Harbor returned model {model.name!r} from provider {model.provider!r}, "
                f"expected {self.model!r}",
                evidence=evidence,
            )

    def _command(self, task: HarborTask, jobs: Path, job_name: str) -> tuple[str, ...]:
        arguments = json.loads(self._agent_arguments_json)
        command = [
            str(self.executable),
            "run",
            "--jobs-dir",
            str(jobs),
            "--job-name",
            job_name,
            "--n-attempts",
            "1",
            "--n-concurrent",
            "1",
            "--max-retries",
            "0",
            "--quiet",
            "--env",
            self.environment,
        ]
        if self.agent_import_path is None:
            command.extend(("--agent", self.agent))
        else:
            command.extend(("--agent-import-path", self.agent_import_path))
        for name in sorted(arguments):
            command.extend(("--agent-kwarg", f"{name}={json.dumps(arguments[name])}"))
        if self.model is not None:
            command.extend(("--model", self.model))
        if isinstance(task, HarborLocalTask):
            command.extend(("--path", str(task.path)))
        else:
            command.extend(("--dataset", task.dataset))
            if task.task is None:
                command.extend(("--n-tasks", "1"))
            else:
                command.extend(("--task-name", task.task))
        return tuple(command)

    @staticmethod
    def _result_path(job_root: Path) -> Path | None:
        candidates = sorted(job_root.glob("*/result.json"))
        if len(candidates) > 1:
            values = ", ".join(str(path) for path in candidates)
            raise HarnessError("protocol", f"Harbor wrote multiple trial results: {values}")
        return candidates[0] if candidates else None

    @staticmethod
    def _trajectory_root(job_root: Path, result_path: Path | None) -> Path | None:
        if result_path is not None:
            candidate = result_path.parent / "agent"
            return candidate if (candidate / "trajectory.json").is_file() else None
        candidates = sorted(job_root.glob("*/agent/trajectory.json"))
        return candidates[0].parent if len(candidates) == 1 else None


@dataclass
class _TrajectoryChain:
    """Trajectory segments read from one attempt, in continuation order."""

    paths: list[Path] = field(default_factory=list)
    trajectories: list[AtifTrajectory | None] = field(default_factory=list)


def _read_trajectory_chain(
    agent_dir: Path,
    trial_root: Path,
    limit_bytes: int,
    chain: _TrajectoryChain,
) -> None:
    """Read agent/trajectory.json and every continued_trajectory_ref it names.

    Harbor resolves each reference against the agent directory of the trial
    (harbor/utils/traces_utils.py) and writes continuation segments next to
    trajectory.json, so the agent directory is also every segment's parent.
    Cycles, escapes, missing targets, and oversized chains are rejected rather
    than silently truncated; segments read before a failure stay in chain.
    """

    seen: set[Path] = set()
    total = 0
    current = agent_dir / "trajectory.json"
    while True:
        relative = current.relative_to(trial_root)
        resolved = current.resolve()
        if resolved in seen:
            raise ValueError(f"{relative}: trajectory continuation chain is cyclic")
        seen.add(resolved)
        if len(chain.paths) >= _MAX_TRAJECTORY_SEGMENTS:
            raise ValueError(
                f"{relative}: trajectory continuation chain exceeds "
                f"{_MAX_TRAJECTORY_SEGMENTS} segments"
            )
        _require_native_file(current, trial_root)
        remaining = limit_bytes - total
        with current.open("rb") as stream:
            data = stream.read(remaining + 1)
        if len(data) > remaining:
            raise ValueError(
                f"{relative}: trajectory continuation chain exceeds the {limit_bytes} byte limit"
            )
        total += len(data)
        chain.paths.append(current)
        try:
            trajectory = parse_trajectory(data)
        except ValueError as error:
            chain.trajectories.append(None)
            raise ValueError(f"{relative}: {error}") from error
        chain.trajectories.append(trajectory)
        reference = trajectory.continued_trajectory_ref
        if reference is None:
            return
        current = _continuation_path(agent_dir, reference)


def _continuation_path(agent_dir: Path, reference: str) -> Path:
    target = Path(reference)
    if target.is_absolute() or ".." in target.parts:
        raise ValueError(
            f"continued_trajectory_ref must stay inside the agent directory: {reference!r}"
        )
    return agent_dir / target


def _require_native_file(path: Path, root: Path) -> None:
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{relative} contains a symbolic link")
    if not path.is_file():
        raise ValueError(f"{relative} is not a regular file")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{relative} is outside the attempt directory")


def _read_native_file(path: Path, root: Path, limit_bytes: int) -> bytes:
    _require_native_file(path, root)
    with path.open("rb") as stream:
        data = stream.read(limit_bytes + 1)
    if len(data) > limit_bytes:
        raise ValueError(f"{path.name} exceeds the {limit_bytes} byte limit")
    return data


def _agent_arguments_snapshot(arguments: dict[str, JsonValue]) -> str:
    """Freeze agent arguments as JSON so later mutation cannot change a run."""

    try:
        return json.dumps(arguments, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Harbor agent_arguments must be plain JSON data") from error


def _is_credential_name(name: str) -> bool:
    normalized = re.sub(r"[\W_]+", "", name).casefold()
    return any(normalized.endswith(value) for value in _CREDENTIAL_NAMES)


def _credential_paths(value: JsonValue, path: str = "agent_arguments") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for name, child in value.items():
            child_path = f"{path}.{name}"
            # A name/value pair such as {"name": "access_token", "value": ...}
            # carries the credential in its value, so flag the naming field.
            if _is_credential_name(str(name)) or (
                name == "name" and isinstance(child, str) and _is_credential_name(child)
            ):
                paths.append(child_path)
            else:
                paths.extend(_credential_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_credential_paths(child, f"{path}[{index}]"))
    return paths


def _native_exception_detail(result: _HarborResult) -> str:
    info = result.exception_info
    detail = ""
    if info is not None:
        kind = _bounded_text(info.exception_type or "", _EXCEPTION_TYPE_LIMIT)
        message = _bounded_text(info.exception_message or "", _EXCEPTION_MESSAGE_LIMIT)
        detail = ": ".join(part for part in (kind, message) if part)
    summary = f"Harbor trial {result.trial_name!r} recorded an exception"
    return f"{summary}: {detail}" if detail else summary


def _bounded_text(value: str, limit: int) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def _selected_reward(
    verifier: _HarborVerifier | None,
    reward: str,
    evidence: Evidence,
) -> float:
    values = verifier.rewards if verifier is not None else None
    if not isinstance(values, dict):
        raise HarnessError("verify", "Harbor result has no verifier rewards", evidence=evidence)
    value = values.get(reward)
    number = _reward_number(value)
    if number is not None:
        return number
    raise HarnessError(
        "verify",
        f"Harbor result has no finite numeric reward {reward!r}",
        evidence=evidence,
    )


def _record_reward_facts(trace: TraceRecorder, verifier: _HarborVerifier | None) -> None:
    values = verifier.rewards if verifier is not None else None
    if not isinstance(values, dict):
        return
    for key, value in values.items():
        number = _reward_number(value)
        name = _fact_name(f"harbor-reward-{key}") if isinstance(key, str) else None
        if number is None or name is None:
            continue
        trace.record(ResourceFact(name=name, value=number, unit="reward"))


def _record_usage_facts(trace: TraceRecorder, agent_result: _HarborAgentResult | None) -> None:
    if agent_result is None:
        return
    fields = (
        ("n_input_tokens", "harbor-agent-input-tokens", "tokens"),
        ("n_cache_tokens", "harbor-agent-cache-tokens", "tokens"),
        ("n_output_tokens", "harbor-agent-output-tokens", "tokens"),
        ("cost_usd", "harbor-agent-cost", "usd"),
    )
    for attribute, name, unit in fields:
        number = _number(getattr(agent_result, attribute))
        if number is not None:
            trace.record(ResourceFact(name=name, value=number, unit=unit))


def _reward_number(value: JsonValue) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _number(value: JsonValue) -> float | None:
    number = _reward_number(value)
    return number if number is not None and number >= 0 else None


def _fact_name(value: str) -> str | None:
    name = re.sub(r"[^A-Za-z0-9._-]", "-", value)
    return name if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) else None


def _harbor_artifacts(
    root: Path,
    result: Path | None,
    stdout: Path,
    stderr: Path,
    segments: list[Path],
) -> tuple[Artifact, ...]:
    """Collect the known Harbor layout without walking the attempt tree."""

    candidates = [path for path in (result, stdout, stderr, *segments) if path is not None]
    trial_dirs = {path.parent for path in segments}
    if result is not None:
        trial_dirs.add(result.parent)
    for trial_dir in sorted(trial_dirs):
        candidates.extend(trial_dir / name for name in ("trial.log", "exception.txt"))
        candidates.extend(
            trial_dir / "verifier" / name
            for name in ("reward.json", "reward.txt", "test-stdout.txt", "test-stderr.txt")
        )
    unique: list[Path] = []
    for path in candidates:
        if path in unique:
            continue
        try:
            _require_native_file(path, root)
        except (OSError, ValueError):
            continue
        unique.append(path)
    return tuple(
        Artifact(name=f"harbor-{index:02d}-{path.name}", path=str(path.relative_to(root)))
        for index, path in enumerate(unique, start=1)
    )


def _task_identity(task: HarborTask) -> JsonValue:
    if isinstance(task, HarborLocalTask):
        return {
            "kind": task.kind,
            "case": task.case,
            "revision": task.revision,
        }
    return task.model_dump(mode="json")


def _write_json(path: Path, value: JsonValue) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
