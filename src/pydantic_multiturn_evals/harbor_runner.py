"""Whole-arm Harbor execution, deterministic task compilation, and normalization."""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import signal
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai.models import Model

from pydantic_multiturn_evals.comparison import (
    ComparisonResult,
    CompletedArm,
    pair_results,
)
from pydantic_multiturn_evals.evaluation import (
    SuiteResult,
    build_dataset,
    derive_gate,
    plan_cases,
)
from pydantic_multiturn_evals.models import (
    CaseKey,
    EnvironmentEvidence,
    EnvironmentName,
    GateResult,
    Identifier,
    PlannedCase,
    Scenario,
    ScenarioResult,
    StrictModel,
    SuiteSpec,
    TargetCompletion,
    Text,
)
from pydantic_multiturn_evals.observability import NO_TRACE, TraceFields, TraceRuntime
from pydantic_multiturn_evals.providers import build_model
from pydantic_multiturn_evals.spec import load_suite

HARBOR_USER_AGENT = "pydantic_multiturn_harbor_agent:PydanticAdaptiveUserAgent"
CASE_START = "<pydantic-multiturn-eval-case>"
CASE_END = "</pydantic-multiturn-eval-case>"


class HarborLimits(StrictModel):
    job_seconds: float = Field(default=7200, gt=0, le=86400)
    stderr_bytes: int = Field(default=64 * 1024, gt=0, le=16 * 1024 * 1024)


class HarborArmSpec(StrictModel):
    version: Literal[1]
    name: Identifier
    kind: Literal["harbor"]
    base_config: Path
    task_template: Path
    command: tuple[Text, ...] = Field(min_length=1)
    actor_command: tuple[Text, ...] = Field(min_length=1)
    inherit_env: tuple[EnvironmentName, ...] = ()
    reward_key: Identifier = "reward"
    minimum_environment_reward: float = Field(default=1, ge=0, le=1)
    limits: HarborLimits = HarborLimits()


@dataclass(frozen=True, slots=True)
class LoadedHarborArm:
    spec: HarborArmSpec
    source_directory: Path


@dataclass(frozen=True, slots=True)
class CompiledHarborCase:
    key: CaseKey
    case_name: str
    task_name: str
    task_path: Path
    scenario: Scenario


@dataclass(frozen=True, slots=True)
class HarborJobPlan:
    comparison_id: str
    role: Literal["baseline", "candidate"]
    arm_name: str
    config_path: Path
    job_dir: Path
    cases: tuple[CompiledHarborCase, ...]
    max_concurrency: int


@dataclass(frozen=True, slots=True)
class HarborTrialReceipt:
    result_path: Path
    conversation_path: Path


@dataclass(frozen=True, slots=True)
class HarborJobReceipt:
    job_id: str
    job_dir: Path
    trials: tuple[HarborTrialReceipt, ...]


class HarborBackend(Protocol):
    async def run_job(self, plan: HarborJobPlan) -> HarborJobReceipt: ...


class _HarborVerifierResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rewards: dict[str, float | int] | None = None


class _HarborTrialResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    task_name: str
    trial_name: str
    verifier_result: _HarborVerifierResult | None = None
    exception_info: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class HarborSuiteResult:
    normalized: SuiteResult
    receipt: HarborJobReceipt

    @property
    def gate(self) -> GateResult:
        return self.normalized.gate

    @property
    def report(self):
        return self.normalized.report

    def write_artifacts(self, directory: str | Path) -> None:
        output = Path(directory)
        self.normalized.write_artifacts(output)
        destination = output / "harbor"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(self.receipt.job_dir, destination)


class HarborArmRunner:
    def __init__(self, loaded: LoadedHarborArm, backend: HarborBackend) -> None:
        self._loaded = loaded
        self._backend = backend

    async def run(
        self,
        suite: SuiteSpec,
        *,
        role: Literal["baseline", "candidate"],
        comparison_id: str,
        repeat: int,
        max_concurrency: int,
        progress: bool,
        judge_model: Model,
        work_directory: Path,
        trace: TraceRuntime,
    ) -> CompletedArm:
        plan = prepare_harbor_job(
            suite,
            loaded=self._loaded,
            role=role,
            comparison_id=comparison_id,
            repeat=repeat,
            max_concurrency=max_concurrency,
            work_directory=work_directory,
        )
        receipt = await self._backend.run_job(plan)
        normalized = await _normalize_harbor_job(
            suite,
            plan=plan,
            receipt=receipt,
            spec=self._loaded.spec,
            judge_model=judge_model,
            progress=progress,
            trace=trace,
        )
        return CompletedArm(
            role=role,
            target_name=self._loaded.spec.name,
            target_kind="harbor",
            result=HarborSuiteResult(normalized=normalized, receipt=receipt),
        )


BackendFactory = Callable[[LoadedHarborArm], HarborBackend]
JudgeModelFactory = Callable[[SuiteSpec], Model]


async def compare_harbor_suite(
    source: str | Path | SuiteSpec,
    *,
    baseline: LoadedHarborArm,
    candidate: LoadedHarborArm,
    work_directory: Path,
    backend_factory: BackendFactory | None = None,
    judge_model_factory: JudgeModelFactory | None = None,
    max_concurrency: int = 1,
    repeat: int = 1,
    progress: bool = True,
    trace: TraceRuntime = NO_TRACE,
) -> ComparisonResult:
    """Run two isolated Harbor jobs and pair their normalized case results."""

    suite = source if isinstance(source, SuiteSpec) else load_suite(source)
    if baseline.spec.name == candidate.spec.name:
        raise ValueError("baseline and candidate need distinct Harbor arm names")
    if repeat < 1:
        raise ValueError("repeat must be positive")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    make_backend = backend_factory or (
        lambda loaded: HarborCliBackend(loaded.spec, loaded.source_directory)
    )
    make_judge = judge_model_factory or (lambda value: build_model(value.judge.model))
    comparison_id = uuid4().hex
    fields = TraceFields(suite=suite.name, comparison_id=comparison_id)

    with trace.span("multiturn-evals.comparison", fields, as_type="evaluator") as span:
        baseline_arm = await HarborArmRunner(baseline, make_backend(baseline)).run(
            suite,
            role="baseline",
            comparison_id=comparison_id,
            repeat=repeat,
            max_concurrency=max_concurrency,
            progress=progress,
            judge_model=make_judge(suite),
            work_directory=work_directory / "baseline",
            trace=trace,
        )
        candidate_arm = await HarborArmRunner(candidate, make_backend(candidate)).run(
            suite,
            role="candidate",
            comparison_id=comparison_id,
            repeat=repeat,
            max_concurrency=max_concurrency,
            progress=progress,
            judge_model=make_judge(suite),
            work_directory=work_directory / "candidate",
            trace=trace,
        )
        result = ComparisonResult(
            comparison_id=comparison_id,
            suite_name=suite.name,
            baseline=baseline_arm,
            candidate=candidate_arm,
            pairs=pair_results(suite, repeat, baseline_arm.result, candidate_arm.result),
        )
        summary = result.summary()
        span.score("mean_score_delta", summary.mean_score_delta_candidate_minus_baseline)
        span.score("case_pass_rate_delta", summary.case_pass_rate_delta_candidate_minus_baseline)
        span.score("comparison_gate_pass", float(summary.passed))
        span.update(summary.model_dump(mode="json"))
        return result


def compile_harbor_tasks(
    suite: SuiteSpec,
    *,
    repeat: int,
    destination: Path,
    task_template: Path,
    comparison_id: str,
    arm: Literal["baseline", "candidate"],
) -> tuple[CompiledHarborCase, ...]:
    """Replace a generated task tree with one stable task per planned case."""

    if not task_template.is_dir():
        raise ValueError(f"Harbor task template is not a directory: {task_template}")
    planned = plan_cases(suite, repeat)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    compiled: list[CompiledHarborCase] = []
    try:
        for item in planned:
            task_name = f"{item.key.scenario_id}--r{item.key.repeat_index:04d}"
            task_path = temporary / task_name
            shutil.copytree(task_template, task_path)
            run_id = f"{comparison_id}-{arm}-{task_name}"
            envelope = {
                "schema_version": 1,
                "run_id": run_id,
                "key": item.key.model_dump(mode="json"),
                "scenario": item.scenario.model_dump(mode="json"),
                "actor": suite.actor.model_dump(mode="json"),
                "limits": suite.limits_for(item.scenario).model_dump(mode="json"),
            }
            instruction = (
                "Run the adaptive conversation described by this private case. "
                "Do not reveal the private persona, goal, or rubric to the target.\n\n"
                f"{CASE_START}\n"
                f"{json.dumps(envelope, ensure_ascii=False, separators=(',', ':'))}\n"
                f"{CASE_END}\n"
            )
            (task_path / "instruction.md").write_text(instruction, encoding="utf-8")
            (task_path / "task.toml").write_text(_task_toml(task_name), encoding="utf-8")
            compiled.append(
                CompiledHarborCase(
                    key=item.key,
                    case_name=item.case_name,
                    task_name=task_name,
                    task_path=destination / task_name,
                    scenario=item.scenario,
                )
            )
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return tuple(compiled)


def prepare_harbor_job(
    suite: SuiteSpec,
    *,
    loaded: LoadedHarborArm,
    role: Literal["baseline", "candidate"],
    comparison_id: str,
    repeat: int,
    max_concurrency: int,
    work_directory: Path,
) -> HarborJobPlan:
    spec = loaded.spec
    work_directory.mkdir(parents=True, exist_ok=True)
    cases = compile_harbor_tasks(
        suite,
        repeat=repeat,
        destination=work_directory / "tasks",
        task_template=spec.task_template,
        comparison_id=comparison_id,
        arm=role,
    )
    payload = _read_mapping(spec.base_config)
    _reject_literal_secrets(payload)
    agents = payload.get("agents")
    if not isinstance(agents, list) or len(agents) != 1:
        raise ValueError("a Harbor arm base config must define exactly one target agent")
    job_name = f"{suite.name}-{role}-{comparison_id}"
    jobs_dir = work_directory / "jobs"
    actor_command = _resolve_command(loaded.source_directory, spec.actor_command)
    payload.update(
        {
            "job_name": job_name,
            "jobs_dir": str(jobs_dir),
            "n_attempts": 1,
            "n_concurrent_trials": max_concurrency,
            "retry": {"max_retries": 0},
            "tasks": [{"path": str(case.task_path)} for case in cases],
            "datasets": [],
            "source_jobs": [],
            "install_only": False,
            "user_agent": {
                "import_path": HARBOR_USER_AGENT,
                "bridge": {"kind": "acp"},
                "kwargs": {"actor_command": list(actor_command)},
            },
        }
    )
    config_path = work_directory / "job.yaml"
    _write_yaml_atomic(config_path, payload)
    return HarborJobPlan(
        comparison_id=comparison_id,
        role=role,
        arm_name=spec.name,
        config_path=config_path,
        job_dir=jobs_dir / job_name,
        cases=cases,
        max_concurrency=max_concurrency,
    )


class HarborCliBackend:  # pragma: no cover - exercised by the opt-in Harbor smoke
    """Run pinned Harbor as one subprocess and inspect only its saved results."""

    def __init__(self, spec: HarborArmSpec, cwd: Path) -> None:
        self._spec = spec
        self._cwd = cwd

    async def run_job(self, plan: HarborJobPlan) -> HarborJobReceipt:
        environment = {
            name: value
            for name in self._spec.inherit_env
            if (value := os.environ.get(name)) is not None
        }
        missing = [name for name in self._spec.inherit_env if name not in environment]
        if missing:
            raise ValueError(f"Harbor requires environment variable(s): {', '.join(missing)}")
        process = await asyncio.create_subprocess_exec(
            *self._spec.command,
            "run",
            "--config",
            str(plan.config_path),
            "--yes",
            cwd=self._cwd,
            env=environment,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stderr_task = asyncio.create_task(
            _drain_bounded(process.stderr, self._spec.limits.stderr_bytes)
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=self._spec.limits.job_seconds)
        except TimeoutError as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            await stderr_task
            raise RuntimeError(
                f"Harbor job exceeded {self._spec.limits.job_seconds:g} seconds"
            ) from error
        stderr, truncated = await stderr_task
        if process.returncode != 0:
            suffix = " (truncated)" if truncated else ""
            raise RuntimeError(
                f"Harbor exited with status {process.returncode}; captured "
                f"{len(stderr)} stderr byte(s){suffix}"
            )
        root_result = _read_mapping(plan.job_dir / "result.json")
        job_id = root_result.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("Harbor job result has no id")
        trials = tuple(
            HarborTrialReceipt(
                result_path=path,
                conversation_path=path.parent / "user-agent" / "multiturn-result.json",
            )
            for path in sorted(plan.job_dir.glob("*/result.json"))
        )
        return HarborJobReceipt(job_id=job_id, job_dir=plan.job_dir, trials=trials)


async def _normalize_harbor_job(
    suite: SuiteSpec,
    *,
    plan: HarborJobPlan,
    receipt: HarborJobReceipt,
    spec: HarborArmSpec,
    judge_model: Model,
    progress: bool,
    trace: TraceRuntime,
) -> SuiteResult:
    by_task: dict[str, ScenarioResult | RuntimeError] = {}
    trial_ids: dict[str, str] = {}
    expected = {case.task_name: case for case in plan.cases}
    for item in receipt.trials:
        try:
            trial = _HarborTrialResult.model_validate_json(item.result_path.read_text())
            if trial.task_name not in expected:
                raise RuntimeError(f"Harbor returned unknown task {trial.task_name!r}")
            if trial.task_name in by_task:
                raise RuntimeError(f"Harbor returned duplicate task {trial.task_name!r}")
            if trial.exception_info is not None:
                raise RuntimeError(f"Harbor trial {trial.trial_name!r} failed")
            if trial.verifier_result is None or trial.verifier_result.rewards is None:
                raise RuntimeError(f"Harbor trial {trial.trial_name!r} has no verifier rewards")
            reward_value = trial.verifier_result.rewards.get(spec.reward_key)
            reward = float(reward_value) if reward_value is not None else math.nan
            if not math.isfinite(reward) or not 0 <= reward <= 1:
                raise RuntimeError(
                    f"Harbor trial {trial.trial_name!r} has no finite {spec.reward_key!r} reward"
                )
            conversation = ScenarioResult.model_validate_json(
                item.conversation_path.read_text(encoding="utf-8")
            )
            case = expected[trial.task_name]
            if (
                conversation.scenario_id != case.key.scenario_id
                or conversation.repeat_index != case.key.repeat_index
            ):
                raise RuntimeError(f"Harbor trial {trial.trial_name!r} returned the wrong case")
            reason = (
                f"Harbor reward {spec.reward_key}={reward:g} met the threshold."
                if reward >= spec.minimum_environment_reward
                else f"Harbor reward {spec.reward_key}={reward:g} was below "
                f"{spec.minimum_environment_reward:g}."
            )
            by_task[trial.task_name] = conversation.model_copy(
                update={
                    "completion": TargetCompletion(
                        environment=EnvironmentEvidence(
                            provider="harbor",
                            trial_id=trial.id,
                            verifier=spec.reward_key,
                            passed=reward >= spec.minimum_environment_reward,
                            reward=reward,
                            reason=reason,
                            details={"job_id": receipt.job_id, "trial_name": trial.trial_name},
                        )
                    )
                }
            )
            trial_ids[trial.task_name] = trial.id
        except (OSError, ValidationError, ValueError, RuntimeError) as error:
            task_name = _best_effort_task_name(item.result_path)
            if task_name in expected and task_name not in by_task:
                by_task[task_name] = RuntimeError(str(error))
                continue
            raise RuntimeError(
                f"could not associate Harbor trial result {item.result_path} with one planned case"
            ) from error

    planned_by_key = {case.key: case for case in plan.cases}

    async def task(planned: PlannedCase) -> ScenarioResult:
        compiled = planned_by_key.get(planned.key)
        if compiled is None:
            raise RuntimeError(f"Harbor did not plan case {planned.case_name!r}")
        value = by_task.get(compiled.task_name)
        if value is None:
            raise RuntimeError(f"Harbor returned no trial for {compiled.task_name!r}")
        if isinstance(value, RuntimeError):
            raise value
        environment = value.completion.environment
        case_fields = TraceFields(
            suite=suite.name,
            target=plan.arm_name,
            harness="harbor",
            comparison_id=plan.comparison_id,
            arm=plan.role,
            scenario_id=planned.key.scenario_id,
            repeat_index=planned.key.repeat_index,
            run_id=value.run_id,
            harbor_job_id=receipt.job_id,
            harbor_trial_id=trial_ids[compiled.task_name],
            version=str(spec.version),
        )
        with trace.span(
            "multiturn-evals.scenario",
            case_fields,
            input={"first_prompt": planned.scenario.first_prompt},
            as_type="agent",
        ) as scenario_span:
            scenario_span.update(
                {
                    "termination": value.termination.kind,
                    "target_turns": len(value.transcript.exchanges),
                    "environment_provider": "harbor",
                    "environment_passed": environment.passed if environment else None,
                    "environment_reward": environment.reward if environment else None,
                }
            )
            if environment is not None:
                scenario_span.score(
                    "environment_pass", float(environment.passed), environment.reason
                )
                if environment.reward is not None:
                    scenario_span.score(
                        "environment_reward", environment.reward, environment.reason
                    )
            return value

    dataset = build_dataset(suite, judge_model=judge_model, repeat=_repeat_count(plan.cases))
    fields = TraceFields(
        suite=suite.name,
        target=plan.arm_name,
        harness="harbor",
        comparison_id=plan.comparison_id,
        arm=plan.role,
        version=str(spec.version),
    )
    with trace.span("multiturn-evals.arm", fields, as_type="evaluator") as arm_span:
        report = await dataset.evaluate(
            task,
            name=suite.name,
            task_name="harbor_adaptive_conversation",
            max_concurrency=plan.max_concurrency,
            repeat=1,
            progress=progress,
            metadata={
                "suite": suite.name,
                "schema_version": suite.version,
                "comparison_id": plan.comparison_id,
                "arm": plan.role,
                "target": plan.arm_name,
                "harness": "harbor",
                "harbor_job_id": receipt.job_id,
            },
        )
        gate = derive_gate(report, suite.gate)
        arm_span.score("case_pass_rate", gate.case_pass_rate)
        arm_span.score("mean_score", gate.mean_score)
        arm_span.score("gate_pass", float(gate.passed))
        arm_span.update(gate.model_dump(mode="json"))
    gates = {case.case_name: case for case in gate.cases}
    for case in report.cases:
        outcome = gates[case.name]
        trace.score_run(
            case.output.run_id,
            case_name=case.name,
            score=outcome.score,
            assertion=outcome.assertion,
            passed=outcome.passed,
            reason=outcome.reason,
        )
    return SuiteResult(report=report, gate=gate)


def load_harbor_arm(path: str | Path) -> LoadedHarborArm:
    source = Path(path).resolve()
    try:
        spec = HarborArmSpec.model_validate(_read_mapping(source))
    except ValidationError as error:
        raise ValueError(f"invalid Harbor arm in {source}:\n{error}") from error
    spec = spec.model_copy(
        update={
            "base_config": _resolve_from(source.parent, spec.base_config),
            "task_template": _resolve_from(source.parent, spec.task_template),
        }
    )
    return LoadedHarborArm(spec=spec, source_directory=source.parent)


async def _drain_bounded(stream: asyncio.StreamReader | None, limit: int) -> tuple[bytes, bool]:
    if stream is None:
        return b"", False
    captured = bytearray()
    truncated = False
    while chunk := await stream.read(8192):
        remaining = limit - len(captured)
        if remaining > 0:
            captured.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated = True
    return bytes(captured), truncated


def _best_effort_task_name(result_path: Path) -> str | None:
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("task_name") if isinstance(payload, dict) else None
    return value if isinstance(value, str) else None


def _repeat_count(cases: tuple[CompiledHarborCase, ...]) -> int:
    return max((case.key.repeat_index for case in cases), default=1)


def _read_mapping(path: Path) -> dict[str, object]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"could not read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping in {path}")
    return cast(dict[str, object], payload)


def _reject_literal_secrets(payload: object, path: tuple[str, ...] = ()) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            name = str(key)
            location = (*path, name)
            lowered = name.lower()
            sensitive = any(word in lowered for word in ("api_key", "secret", "token", "password"))
            if sensitive and isinstance(value, str) and not _is_environment_reference(value):
                raise ValueError(
                    f"Harbor base config must not store a literal secret at {'.'.join(location)}"
                )
            _reject_literal_secrets(value, location)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _reject_literal_secrets(value, (*path, str(index)))


def _is_environment_reference(value: str) -> bool:
    return value.startswith("${") and value.endswith("}") and value[2:-1].isidentifier()


def _write_yaml_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _resolve_from(directory: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (directory / path).resolve()


def _resolve_command(directory: Path, command: tuple[str, ...]) -> tuple[str, ...]:
    executable = Path(command[0])
    if executable.is_absolute() or executable.parent == Path("."):
        return command
    return (str((directory / executable).resolve()), *command[1:])


def _task_toml(task_name: str) -> str:
    return (
        'schema_version = "1.4"\n\n'
        "[task]\n"
        f'name = "{task_name}"\n'
        'version = "1.0.0"\n'
        "authors = []\n"
        "keywords = []\n\n"
        "[verifier]\n"
        "timeout_sec = 300.0\n\n"
        "[agent]\n"
        "timeout_sec = 1800.0\n\n"
        "[environment]\n"
        "build_timeout_sec = 900.0\n"
        "cpus = 2\n"
        "memory_mb = 4096\n"
        "storage_mb = 10240\n"
        "gpus = 0\n"
        "mcp_servers = []\n\n"
        "[verifier.env]\n\n"
        "[solution.env]\n"
    )
