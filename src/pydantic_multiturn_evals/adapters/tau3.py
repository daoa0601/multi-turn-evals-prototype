"""One-task adapter for the text runner in tau3-bench."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator

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
    Message,
    MessageFact,
    ModelCall,
    ModelCallFact,
    ModelUsage,
    Name,
    NativeCheck,
    PublicOutcome,
    StrictModel,
    Text,
    ToolFact,
    TraceRecorder,
    Trial,
)


class Tau3Task(StrictModel):
    case: Name
    domain: Name
    task: Text
    task_set: Text | None = None
    split: Text = "base"
    revision: Text


class Tau3Participant(StrictModel):
    implementation: Text
    model: Text
    provider: Name = "litellm"
    interface: Name = "chat"
    arguments: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_inline_credentials(self) -> Self:
        invalid = list(_credential_paths(self.arguments))
        if invalid:
            raise ValueError(
                f"Tau3 model credentials must come from the environment: {', '.join(invalid)}"
            )
        return self


def _credential_paths(value: JsonValue, path: str = "") -> Iterator[str]:
    if isinstance(value, dict):
        for name, item in value.items():
            child_path = f"{path}.{name}" if path else name
            normalized = name.casefold()
            if normalized in {"api_key", "authorization", "password", "secret", "token"} or (
                normalized.endswith("_api_key")
            ):
                yield child_path
            yield from _credential_paths(item, child_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _credential_paths(item, f"{path}[{index}]")


class _WorkerReply(StrictModel):
    protocol: Literal[1]
    package_version: Text
    simulations: Annotated[int, Field(ge=0)]


class _TauToolCall(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    id: str = ""
    name: str
    arguments: JsonValue


class _TauMessage(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    role: str
    content: str | None = None
    tool_calls: tuple[_TauToolCall, ...] | None = None
    id: str | None = None
    cost: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    usage: JsonValue = None
    raw_data: JsonValue = None
    generation_time_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None


class _TauReward(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    reward: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    reward_basis: JsonValue = None
    reward_breakdown: JsonValue = None


class _TauParticipantInfo(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    implementation: str
    llm: str | None = None
    llm_args: JsonValue = None


class _TauInfo(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    git_commit: str
    agent_info: _TauParticipantInfo
    user_info: _TauParticipantInfo


class _TauSimulation(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    id: str
    task_id: str
    duration: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    termination_reason: str
    agent_cost: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    user_cost: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    reward_info: _TauReward | None = None
    messages: tuple[_TauMessage, ...] = ()


class _TauResults(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    info: _TauInfo
    simulations: tuple[_TauSimulation, ...]


@dataclass(frozen=True, slots=True)
class Tau3TextHarness:
    """Let tau3-bench own one text simulation and import its native result."""

    worker_argv: tuple[str, ...]
    worker_cwd: Path
    package_version: str
    checkout_revision: str
    tasks: tuple[Tau3Task, ...]
    agent: Tau3Participant
    user: Tau3Participant
    check: str = "tau3.reward"
    pass_threshold: float = 1.0
    max_steps: int = 200
    max_errors: int = 10
    enforce_communication_protocol: bool = False
    simulation_timeout_seconds: float = 900
    cancel_grace_seconds: float = 10
    output_bytes: int = 100_000
    stderr_bytes: int = 2_000_000
    inherit_environment: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        cases = [task.case for task in self.tasks]
        if not self.worker_argv:
            raise ValueError("Tau3 worker command must not be empty")
        if not self.tasks:
            raise ValueError("Tau3TextHarness needs at least one task binding")
        if len(cases) != len(set(cases)):
            raise ValueError("Tau3 task case bindings must be unique")
        if not 0 <= self.pass_threshold <= 1:
            raise ValueError("Tau3 pass threshold must be between zero and one")
        if self.max_steps < 1 or self.max_errors < 1:
            raise ValueError("Tau3 step and error limits must be positive")
        if self.simulation_timeout_seconds <= 0 or self.cancel_grace_seconds < 0:
            raise ValueError("Tau3 timeout must be positive and cancellation grace non-negative")
        if self.output_bytes < 1 or self.stderr_bytes < 1:
            raise ValueError("Tau3 output limits must be positive")

    @property
    def component(self) -> ComponentDefinition:
        return ComponentDefinition(
            kind="tau3-text",
            version="1",
            config={
                "package_version": self.package_version,
                "checkout_revision": self.checkout_revision,
                "tasks": [task.model_dump(mode="json") for task in self.tasks],
                "agent": self.agent.model_dump(mode="json"),
                "user": self.user.model_dump(mode="json"),
                "check": self.check,
                "pass_threshold": self.pass_threshold,
                "max_steps": self.max_steps,
                "max_errors": self.max_errors,
                "enforce_communication_protocol": self.enforce_communication_protocol,
                "simulation_timeout_seconds": self.simulation_timeout_seconds,
                "cancel_grace_seconds": self.cancel_grace_seconds,
                "output_bytes": self.output_bytes,
                "stderr_bytes": self.stderr_bytes,
                "inherit_environment": list(self.inherit_environment),
            },
        )

    async def run(self, case: Case, trial: Trial, trace: TraceRecorder) -> Evidence:
        task = self._task(case.name)
        tau_directory = trial.directory / "tau3"
        tau_directory.mkdir()
        results_path = tau_directory / "results.json"
        reply_path = trial.directory / "tau3-worker-reply.json"
        request_path = trial.directory / "tau3-request.json"
        state_path = trial.directory / "tau3-state.json"
        request = self._request(task, trial, results_path, reply_path, tau_directory)
        _write_json(request_path, request)
        _write_json(
            state_path,
            {
                "protocol": 1,
                "phase": "starting",
                "domain": task.domain,
                "task": task.task,
            },
        )
        environment = command_environment(self.inherit_environment, trial.directory)
        try:
            process = await asyncio.create_subprocess_exec(
                *self.worker_argv,
                cwd=self.worker_cwd,
                env=environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except Exception as error:
            raise HarnessError("start", f"could not start Tau3 worker: {error}") from error
        try:
            stdout, stderr = await collect_process_output(
                process,
                (json.dumps(request) + "\n").encode(),
                stdout_bytes=self.output_bytes,
                stderr_bytes=self.stderr_bytes,
                cancel_grace_seconds=self.cancel_grace_seconds,
            )
        except ProcessOutputLimitError as error:
            raise HarnessError("protocol", f"Tau3 worker {error}") from error

        stdout_path = trial.directory / "tau3.stdout.txt"
        stderr_path = trial.directory / "tau3.stderr.txt"
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        artifacts = _artifacts(
            trial.directory,
            request_path,
            reply_path,
            results_path,
            state_path,
            stdout_path,
            stderr_path,
        )
        partial = Evidence(artifacts=artifacts)
        if process.returncode != 0:
            detail = stderr.decode(errors="replace").strip()
            raise HarnessError(
                "execute",
                f"Tau3 worker exited with code {process.returncode}"
                + (f": {detail[-2000:]}" if detail else ""),
                evidence=partial,
            )
        try:
            reply = _read_reply(reply_path)
            results = _read_results(results_path)
        except HarnessError as error:
            raise HarnessError(error.phase, str(error), evidence=partial) from error
        if reply.package_version != self.package_version:
            raise HarnessError(
                "protocol",
                f"Tau3 worker reported version {reply.package_version!r}, expected "
                f"{self.package_version!r}",
                evidence=partial,
            )
        if results.info.git_commit != self.checkout_revision:
            raise HarnessError(
                "protocol",
                f"Tau3 result came from revision {results.info.git_commit!r}, expected "
                f"{self.checkout_revision!r}",
                evidence=partial,
            )
        if reply.simulations != 1 or len(results.simulations) != 1:
            raise HarnessError(
                "protocol", "Tau3 worker must return one simulation", evidence=partial
            )
        simulation = results.simulations[0]
        if simulation.task_id != task.task:
            raise HarnessError(
                "protocol",
                f"Tau3 returned task {simulation.task_id!r}, expected {task.task!r}",
                evidence=partial,
            )
        if simulation.reward_info is None:
            raise HarnessError("verify", "Tau3 result has no native reward", evidence=partial)

        _require_participant("agent", self.agent, results.info.agent_info, partial)
        _require_participant("user", self.user, results.info.user_info, partial)
        public_messages = _import_trace(
            results.info,
            simulation,
            self.agent,
            self.user,
            trace,
        )
        reward = simulation.reward_info.reward
        check = NativeCheck(
            name=self.check,
            passed=reward >= self.pass_threshold,
            reason=(
                f"Tau3 reward={reward:g}; pass threshold is {self.pass_threshold:g}; "
                f"termination={simulation.termination_reason}"
            ),
        )
        _write_json(
            state_path,
            {
                "protocol": 1,
                "phase": "finished",
                "domain": task.domain,
                "task": task.task,
                "simulation": simulation.id,
                "reward": reward,
            },
        )
        return Evidence(
            public=PublicOutcome(
                messages=public_messages,
                output={
                    "task_id": simulation.task_id,
                    "simulation_id": simulation.id,
                    "termination_reason": simulation.termination_reason,
                    "duration_seconds": simulation.duration,
                    "agent_cost_usd": simulation.agent_cost,
                    "user_cost_usd": simulation.user_cost,
                },
            ),
            checks=(check,),
            artifacts=artifacts,
        )

    def _request(
        self,
        task: Tau3Task,
        trial: Trial,
        results_path: Path,
        reply_path: Path,
        output_directory: Path,
    ) -> JsonValue:
        return {
            "protocol": 1,
            "package_version": self.package_version,
            "checkout_revision": self.checkout_revision,
            "task": task.model_dump(mode="json"),
            "agent": self.agent.model_dump(mode="json"),
            "user": self.user.model_dump(mode="json"),
            "seed": trial.seed,
            "max_steps": self.max_steps,
            "max_errors": self.max_errors,
            "enforce_communication_protocol": self.enforce_communication_protocol,
            "simulation_timeout_seconds": self.simulation_timeout_seconds,
            "results_path": str(results_path.resolve()),
            "reply_path": str(reply_path.resolve()),
            "output_directory": str(output_directory.resolve()),
        }

    def _task(self, case: str) -> Tau3Task:
        for task in self.tasks:
            if task.case == case:
                return task
        raise HarnessError("prepare", f"no Tau3 task is bound to case {case!r}")


def _read_reply(path: Path) -> _WorkerReply:
    try:
        return _WorkerReply.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise HarnessError("protocol", f"invalid Tau3 worker reply: {error}") from error


def _read_results(path: Path) -> _TauResults:
    try:
        return _TauResults.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise HarnessError("protocol", f"invalid Tau3 result {path}: {error}") from error


def _import_trace(
    info: _TauInfo,
    simulation: _TauSimulation,
    agent: Tau3Participant,
    user: Tau3Participant,
    trace: TraceRecorder,
) -> tuple[Message, ...]:
    public_messages: list[Message] = []
    tool_names: dict[str, str] = {}
    generations = {"assistant": 0, "user": 0}
    participants = {"assistant": info.agent_info, "user": info.user_info}
    profiles = {"assistant": agent, "user": user}
    roles = {"assistant": "target", "user": "actor"}
    actors = {"assistant": "tau3-agent", "user": "tau3-user"}
    for message in simulation.messages:
        if message.role in participants:
            participant = participants[message.role]
            if (
                participant.llm is not None
                and participant.llm != "dummy"
                and _is_model_generation(message)
            ):
                generations[message.role] += 1
                trace.record(
                    ModelCallFact(
                        call=_model_call(
                            message,
                            participant,
                            profiles[message.role],
                            roles[message.role],
                            generations[message.role],
                        )
                    )
                )
        if message.role in actors and message.content and message.content.strip():
            public = Message(
                role=message.role,
                content=message.content,
                actor=actors[message.role],
            )
            public_messages.append(public)
            trace.record(MessageFact(message=public))
        for call in message.tool_calls or ():
            tool_names[call.id] = call.name
            trace.record(ToolFact(name=call.name, input=call.arguments))
        if message.role == "tool" and message.content:
            trace.record(
                ToolFact(
                    name=tool_names.get(message.id or "", "tau3-tool-result"),
                    output=_json_or_text(message.content),
                )
            )
    return tuple(public_messages)


def _is_model_generation(message: _TauMessage) -> bool:
    return (
        message.usage is not None
        or message.raw_data is not None
        or message.generation_time_seconds is not None
    )


def _model_call(
    message: _TauMessage,
    participant: _TauParticipantInfo,
    profile: Tau3Participant,
    role: str,
    sequence: int,
) -> ModelCall:
    arguments = participant.llm_args if isinstance(participant.llm_args, dict) else {}
    return ModelCall(
        role=role,
        operation=f"message-{sequence}",
        requested=ComponentDefinition(
            kind="tau3-participant-model",
            version="1",
            config={
                "implementation": participant.implementation,
                "model": participant.llm,
                "arguments": arguments,
            },
        ),
        reported_model=participant.llm,
        provider=profile.provider,
        interface=profile.interface,
        sampling=arguments,
        duration_seconds=message.generation_time_seconds,
        status="completed",
        usage=_model_usage(message),
    )


def _require_participant(
    role: str,
    expected: Tau3Participant,
    actual: _TauParticipantInfo,
    evidence: Evidence,
) -> None:
    if actual.implementation != expected.implementation or actual.llm != expected.model:
        raise HarnessError(
            "protocol",
            f"Tau3 {role} identity changed from {expected.implementation}/{expected.model} "
            f"to {actual.implementation}/{actual.llm}",
            evidence=evidence,
        )


def _model_usage(message: _TauMessage) -> ModelUsage:
    usage = message.usage if isinstance(message.usage, dict) else {}
    known = {
        "prompt_tokens",
        "input_tokens",
        "completion_tokens",
        "output_tokens",
        "cached_tokens",
        "cache_read_tokens",
    }
    details = {
        name: value
        for name, value in usage.items()
        if name not in known
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    }
    return ModelUsage(
        requests=1,
        tool_calls=len(message.tool_calls or ()),
        input_tokens=_non_negative_int(usage, "input_tokens", "prompt_tokens"),
        cache_read_tokens=_non_negative_int(usage, "cache_read_tokens", "cached_tokens"),
        output_tokens=_non_negative_int(usage, "output_tokens", "completion_tokens"),
        cost_usd=max(message.cost, 0) if message.cost is not None else None,
        details=details,
    )


def _non_negative_int(values: dict[str, JsonValue], *names: str) -> int | None:
    for name in names:
        value = values.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _json_or_text(value: str) -> JsonValue:
    try:
        return TypeAdapter(JsonValue).validate_python(json.loads(value))
    except (json.JSONDecodeError, ValueError):
        return value


def _artifacts(
    root: Path,
    request: Path,
    reply: Path,
    results: Path,
    state: Path,
    stdout: Path,
    stderr: Path,
) -> tuple[Artifact, ...]:
    candidates = (
        ("tau3-request", request),
        ("tau3-worker-reply", reply),
        ("tau3-results", results),
        ("tau3-state", state),
        ("tau3-stdout", stdout),
        ("tau3-stderr", stderr),
    )
    return tuple(
        Artifact(name=name, path=str(path.relative_to(root)))
        for name, path in candidates
        if path.is_file()
    )


def _write_json(path: Path, value: JsonValue) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
