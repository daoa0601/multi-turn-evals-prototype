"""Durable execution for one frozen experiment plan."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import shutil
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import Field, ValidationError

from pydantic_multiturn_evals.evaluation import evaluate_suite
from pydantic_multiturn_evals.experiment import (
    AgentEnvExecution,
    ArmCaseKey,
    ComparisonPlan,
    ContainerExecution,
    CorpusCaseKey,
    ExperimentPlan,
    HostExecution,
    PlannedExperimentCase,
    PydanticAIHarness,
    SessionArmPlan,
)
from pydantic_multiturn_evals.model_bindings import bind_model, required_environment
from pydantic_multiturn_evals.models import (
    AgentEnvTargetSpec,
    CaseKey,
    CommandTargetSpec,
    FixtureEntry,
    ModelSpec,
    PlannedCase,
    PydanticAITargetSpec,
    StrictModel,
)
from pydantic_multiturn_evals.providers import PydanticActor, PydanticAITarget
from pydantic_multiturn_evals.spec import LoadedTarget
from pydantic_multiturn_evals.targets import Target, build_target
from pydantic_multiturn_evals.trajectory import PydanticTrajectoryAssessor


class RequestedModel(StrictModel):
    role: Literal["actor", "target", "judge", "observer"]
    provider: str
    model: str
    reported_model: str | None = None
    usage_status: Literal["reported", "partial", "unavailable"] = "unavailable"
    requests: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class ExperimentCaseResult(StrictModel):
    key: ArmCaseKey
    passed: bool
    score: float | None = None
    run_id: str | None = None
    duration_seconds: float = Field(ge=0)
    task_failed: bool = False
    errors: tuple[str, ...] = ()
    trajectory_status: Literal["complete", "partial", "skipped"] | None = None
    requested_models: tuple[RequestedModel, ...] = ()


class ArmRunSummary(StrictModel):
    arm: str
    passed: bool
    planned: int
    completed: int
    quality_passed: int
    task_failed: int
    execution_failed: int
    interrupted: int
    running: int
    pending: int
    case_pass_rate: float
    mean_score: float


class ExperimentPairResult(StrictModel):
    corpus: CorpusCaseKey
    baseline: ExperimentCaseResult
    candidate: ExperimentCaseResult
    score_delta_candidate_minus_baseline: float | None = None


class ExperimentComparisonSummary(StrictModel):
    name: str
    baseline: str
    candidate: str
    complete: bool
    mean_score_delta_candidate_minus_baseline: float | None = None
    pairs: tuple[ExperimentPairResult, ...] = ()


class ExperimentSummary(StrictModel):
    schema_version: Literal[1] = 1
    experiment: str
    passed: bool
    planned: int
    completed: int
    quality_passed: int
    task_failed: int
    execution_failed: int
    interrupted: int
    running: int
    pending: int
    arms: tuple[ArmRunSummary, ...]
    comparisons: tuple[ExperimentComparisonSummary, ...]


CaseExecutor = Callable[
    [ExperimentPlan, SessionArmPlan, PlannedExperimentCase, Path],
    Awaitable[ExperimentCaseResult],
]


def save_new_experiment(plan: ExperimentPlan, directory: Path) -> None:
    """Publish a new immutable plan without overwriting prior output."""

    if directory.exists():
        existing = [path for path in directory.iterdir() if path.name != ".lock"]
        if existing:
            raise ValueError(f"experiment output directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    frozen_plan = _freeze_command_workspaces(plan, directory)
    _write_json_atomic(directory / "plan.json", frozen_plan.model_dump(mode="json"))
    _write_summary(frozen_plan, directory)


def load_saved_experiment(directory: Path) -> ExperimentPlan:
    path = directory / "plan.json"
    try:
        plan = ExperimentPlan.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"could not read saved experiment plan {path}: {error}") from error
    except ValidationError as error:
        raise ValueError(f"invalid saved experiment plan {path}:\n{error}") from error
    return _resolve_saved_workspaces(plan, directory)


@contextmanager
def experiment_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another experiment process owns {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def preflight_experiment(
    plan: ExperimentPlan,
    *,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Check every arm before starting any case."""

    values = os.environ if environment is None else environment
    issues: list[tuple[str, str, str]] = []
    for arm in plan.arms:
        models = (
            ("actor", arm.suite.actor.model),
            ("target", arm.target_model),
            ("judge", arm.suite.judge.model),
        )
        if plan.diagnostics.trajectory.enabled:
            models = (*models, ("observer", arm.observer_model))
        for role, model in models:
            for name in required_environment(model):
                if not values.get(name):
                    issues.append(("PREFLIGHT_ENV", f"arms.{arm.name}.{role}", f"missing {name}"))
        _preflight_target(arm, values, issues)
    if issues:
        rendered = "\n".join(
            f"[{code}] {path}: {message}"
            for code, path, message in sorted(set(issues), key=lambda item: (item[1], item[0]))
        )
        raise ValueError(f"experiment preflight failed:\n{rendered}")


async def execute_experiment(
    plan: ExperimentPlan,
    directory: Path,
    *,
    executor: CaseExecutor | None = None,
) -> ExperimentSummary:
    """Run pending cases with bounded concurrency and durable terminal receipts."""

    run_case = executor or _execute_case
    units = _units(plan)
    semaphore = asyncio.Semaphore(plan.bounds.max_concurrency)

    async def bounded(
        ordinal: int,
        arm: SessionArmPlan,
        case: PlannedExperimentCase,
    ) -> None:
        async with semaphore:
            await _run_case(plan, ordinal, arm, case, directory, run_case)
            _write_summary(plan, directory)

    try:
        async with asyncio.timeout(plan.bounds.run_seconds):
            await asyncio.gather(*(bounded(ordinal, arm, case) for ordinal, arm, case in units))
    except TimeoutError:
        _write_json_atomic(
            directory / "timeout.json",
            {"experiment": plan.experiment_name, "reason": "run deadline reached"},
        )
    return _write_summary(plan, directory)


def summarize_experiment(plan: ExperimentPlan, directory: Path) -> ExperimentSummary:
    completed: dict[str, ExperimentCaseResult] = {}
    state_by_key: dict[str, str] = {}
    arm_by_name = {arm.name: arm for arm in plan.arms}
    for ordinal, _arm, case in _units(plan):
        case_directory = _case_directory(directory, ordinal, case.key)
        key = case.key.model_dump_json()
        complete = case_directory / "complete.json"
        if complete.exists():
            completed[key] = ExperimentCaseResult.model_validate_json(
                complete.read_text(encoding="utf-8")
            )
            state_by_key[key] = "complete"
        elif (case_directory / "failure.json").exists():
            state_by_key[key] = "failure"
        elif (case_directory / "interrupted.json").exists():
            state_by_key[key] = "interrupted"
        elif (case_directory / "started.json").exists():
            state_by_key[key] = "running"
        else:
            state_by_key[key] = "pending"

    arms = tuple(
        _summarize_arm(arm_by_name[arm.name], completed, state_by_key) for arm in plan.arms
    )
    comparisons = tuple(
        _summarize_comparison(comparison, completed) for comparison in plan.comparisons
    )
    return ExperimentSummary(
        experiment=plan.experiment_name,
        passed=bool(arms) and all(arm.passed for arm in arms),
        planned=len(state_by_key),
        completed=sum(state == "complete" for state in state_by_key.values()),
        quality_passed=sum(result.passed for result in completed.values()),
        task_failed=sum(_is_task_failure(result) for result in completed.values()),
        execution_failed=sum(state == "failure" for state in state_by_key.values()),
        interrupted=sum(state == "interrupted" for state in state_by_key.values()),
        running=sum(state == "running" for state in state_by_key.values()),
        pending=sum(state == "pending" for state in state_by_key.values()),
        arms=arms,
        comparisons=comparisons,
    )


async def _run_case(
    plan: ExperimentPlan,
    ordinal: int,
    arm: SessionArmPlan,
    case: PlannedExperimentCase,
    directory: Path,
    executor: CaseExecutor,
) -> None:
    case_directory = _case_directory(directory, ordinal, case.key)
    if any(
        (case_directory / name).exists()
        for name in ("complete.json", "failure.json", "interrupted.json")
    ):
        return
    started = case_directory / "started.json"
    if started.exists():
        _write_json_atomic(
            case_directory / "interrupted.json",
            {
                "key": case.key.model_dump(mode="json"),
                "reason": "a previous process started this case without a terminal receipt",
            },
        )
        return

    case_directory.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(
        started,
        {"key": case.key.model_dump(mode="json"), "attempt_id": uuid4().hex},
    )
    try:
        result = await executor(plan, arm, case, case_directory)
        if result.key != case.key:
            raise RuntimeError("case executor returned the wrong arm or corpus key")
    except asyncio.CancelledError:
        _write_json_atomic(
            case_directory / "interrupted.json",
            {"key": case.key.model_dump(mode="json"), "reason": "execution was cancelled"},
        )
        raise
    except Exception as error:
        _write_json_atomic(
            case_directory / "failure.json",
            {
                "key": case.key.model_dump(mode="json"),
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        return
    _write_json_atomic(case_directory / "complete.json", result.model_dump(mode="json"))


async def _execute_case(
    plan: ExperimentPlan,
    arm: SessionArmPlan,
    case: PlannedExperimentCase,
    directory: Path,
) -> ExperimentCaseResult:
    actor_binding = bind_model(arm.suite.actor.model)
    judge_binding = bind_model(arm.suite.judge.model)
    observer_binding = (
        bind_model(arm.observer_model) if plan.diagnostics.trajectory.enabled else None
    )
    actor = PydanticActor(arm.suite.actor, model_binding=actor_binding)
    target = _build_target(arm, directory)
    trajectory_prompt = arm.prompts.trajectory
    assessor = (
        PydanticTrajectoryAssessor(
            observer_binding,
            instructions=trajectory_prompt.text,
        )
        if observer_binding is not None and trajectory_prompt is not None
        else None
    )
    fixture = tuple(
        FixtureEntry(name=item.name, value=json.loads(item.json_value))
        for item in arm.fixture.values
    )
    planned = PlannedCase(
        key=CaseKey(
            scenario_id=case.key.corpus.scenario_id,
            repeat_index=case.key.corpus.repeat_index,
        ),
        case_name=(
            case.scenario.id
            if case.key.corpus.repeat_index == 1
            else f"{case.scenario.id} [{case.key.corpus.repeat_index}]"
        ),
        scenario=case.scenario,
    )
    suite = arm.suite.model_copy(update={"scenarios": (case.scenario,)})
    started = perf_counter()
    result = await evaluate_suite(
        suite,
        target=target,
        actor=actor,
        judge_binding=judge_binding,
        planned_cases=(planned,),
        progress=False,
        arm=arm.name,
        target_model=arm.target_model,
        target_instructions=arm.prompts.target.text,
        fixture=fixture,
        capture_target_evidence=(
            not isinstance(arm.harness, PydanticAIHarness) or arm.harness.capture_target_evidence
        ),
        trajectory_assessor=assessor,
        trajectory_rubric=(
            trajectory_prompt.text
            if assessor is not None and trajectory_prompt is not None
            else None
        ),
        max_trajectory_prefixes=plan.diagnostics.trajectory.max_prefixes_per_case,
    )
    result.write_artifacts(directory)
    gate = result.gate.cases[0]
    run_id = result.report.cases[0].output.run_id if result.report.cases else None
    return ExperimentCaseResult(
        key=case.key,
        passed=gate.passed,
        score=gate.score,
        run_id=run_id,
        duration_seconds=perf_counter() - started,
        task_failed=bool(result.report.failures),
        errors=gate.errors,
        trajectory_status=(
            result.trajectories[0].assessment.status if result.trajectories else None
        ),
        requested_models=_requested_models(plan, arm),
    )


def _build_target(arm: SessionArmPlan, directory: Path) -> Target:
    spec = _materialize_target_spec(arm, directory)
    if isinstance(spec, PydanticAITargetSpec):
        return PydanticAITarget(spec, model_binding=bind_model(arm.target_model))
    source_directory = spec.cwd.parent if isinstance(spec, CommandTargetSpec) else directory
    return build_target(LoadedTarget(spec=spec, source_directory=source_directory))


def _materialize_target_spec(
    arm: SessionArmPlan,
    directory: Path,
) -> PydanticAITargetSpec | CommandTargetSpec | AgentEnvTargetSpec:
    spec = arm.target
    execution = arm.execution
    if isinstance(execution, ContainerExecution):
        if not isinstance(spec, CommandTargetSpec):
            raise ValueError("container execution requires a command harness")
        arguments = ["docker", "run", "--rm", "--init", "--interactive"]
        for name in spec.inherit_env:
            if name != "PATH":
                arguments.extend(("--env", name))
        arguments.append(execution.image)
        return spec.model_copy(
            update={
                "argv": tuple(arguments),
                "inherit_env": tuple(dict.fromkeys(("PATH", *spec.inherit_env))),
            }
        )
    if isinstance(execution, HostExecution) and execution.workspace_mode == "fresh":
        if not isinstance(spec, CommandTargetSpec):
            raise ValueError("fresh workspace execution requires a command harness")
        workspace = directory / "workspace"
        shutil.copytree(
            spec.cwd,
            workspace,
            ignore=shutil.ignore_patterns(
                ".git",
                ".venv",
                ".audit",
                ".references",
                "__pycache__",
                "dist",
                "outputs",
            ),
        )
        executable = Path(spec.argv[0])
        argv = spec.argv
        if not executable.is_absolute() and executable.parent != Path("."):
            argv = (str((spec.cwd / executable).resolve()), *spec.argv[1:])
        return spec.model_copy(update={"cwd": workspace, "argv": argv})
    return spec


def _freeze_command_workspaces(plan: ExperimentPlan, directory: Path) -> ExperimentPlan:
    frozen_root = directory / "frozen-workspaces"
    destinations: dict[Path, Path] = {}
    frozen_arms: list[SessionArmPlan] = []
    for arm in plan.arms:
        target = arm.target
        if not isinstance(target, CommandTargetSpec):
            frozen_arms.append(arm)
            continue
        source = target.cwd.resolve()
        destination = destinations.get(source)
        if destination is None:
            destination = frozen_root / f"{len(destinations) + 1:04d}"
            shutil.copytree(
                source,
                destination,
                ignore=shutil.ignore_patterns(
                    ".git",
                    ".venv",
                    ".audit",
                    ".references",
                    "__pycache__",
                    "dist",
                    "outputs",
                ),
            )
            destinations[source] = destination
        executable = Path(target.argv[0])
        argv = target.argv
        if not executable.is_absolute() and executable.parent != Path("."):
            argv = (str((source / executable).resolve()), *target.argv[1:])
        frozen_arms.append(
            arm.model_copy(
                update={
                    "target": target.model_copy(
                        update={
                            "cwd": destination.relative_to(directory),
                            "argv": argv,
                        }
                    )
                }
            )
        )
    return plan.model_copy(update={"arms": tuple(frozen_arms)})


def _resolve_saved_workspaces(plan: ExperimentPlan, directory: Path) -> ExperimentPlan:
    arms: list[SessionArmPlan] = []
    for arm in plan.arms:
        target = arm.target
        if isinstance(target, CommandTargetSpec) and not target.cwd.is_absolute():
            target = target.model_copy(update={"cwd": (directory / target.cwd).resolve()})
            arm = arm.model_copy(update={"target": target})
        arms.append(arm)
    return plan.model_copy(update={"arms": tuple(arms)})


def _preflight_target(
    arm: SessionArmPlan,
    environment: Mapping[str, str],
    issues: list[tuple[str, str, str]],
) -> None:
    spec = arm.target
    if isinstance(spec, CommandTargetSpec):
        for name in spec.inherit_env:
            if not environment.get(name):
                issues.append(("PREFLIGHT_ENV", f"arms.{arm.name}.harness", f"missing {name}"))
        executable = "docker" if isinstance(arm.execution, ContainerExecution) else spec.argv[0]
        if not _command_available(executable, spec.cwd, environment):
            issues.append(
                (
                    "PREFLIGHT_COMMAND",
                    f"arms.{arm.name}.execution",
                    f"command is unavailable: {executable}",
                )
            )
    elif isinstance(spec, AgentEnvTargetSpec):
        names = (
            spec.credentials.api_url_env,
            spec.credentials.sandbox_url_env,
            spec.credentials.api_key_env,
            *spec.guest_env,
        )
        for name in names:
            if not environment.get(name):
                issues.append(("PREFLIGHT_ENV", f"arms.{arm.name}.execution", f"missing {name}"))
    if isinstance(arm.execution, AgentEnvExecution) and not isinstance(spec, AgentEnvTargetSpec):
        issues.append(
            (
                "PREFLIGHT_EXECUTION",
                f"arms.{arm.name}.execution",
                "AgentENV execution requires an AgentENV target",
            )
        )


def _command_available(
    executable: str,
    cwd: Path,
    environment: Mapping[str, str],
) -> bool:
    path = Path(executable)
    if path.is_absolute() or path.parent != Path("."):
        candidate = path if path.is_absolute() else cwd / path
        return candidate.is_file() and os.access(candidate, os.X_OK)
    return shutil.which(executable, path=environment.get("PATH")) is not None


def _requested_models(
    plan: ExperimentPlan,
    arm: SessionArmPlan,
) -> tuple[RequestedModel, ...]:
    values = [
        _requested_model("actor", arm.suite.actor.model),
        _requested_model("target", arm.target_model),
        _requested_model("judge", arm.suite.judge.model),
    ]
    if plan.diagnostics.trajectory.enabled:
        values.append(_requested_model("observer", arm.observer_model))
    return tuple(values)


def _requested_model(
    role: Literal["actor", "target", "judge", "observer"],
    spec: ModelSpec,
) -> RequestedModel:
    return RequestedModel(
        role=role,
        provider=spec.provider.kind,
        model=spec.name,
    )


def _summarize_arm(
    arm: SessionArmPlan,
    completed: dict[str, ExperimentCaseResult],
    states: dict[str, str],
) -> ArmRunSummary:
    keys = [case.key.model_dump_json() for case in arm.cases]
    results = [completed[key] for key in keys if key in completed]
    scores = [result.score for result in results if result.score is not None]
    quality_passed = sum(result.passed for result in results)
    task_failed = sum(_is_task_failure(result) for result in results)
    case_pass_rate = quality_passed / len(results) if results else 0.0
    mean_score = fmean(scores) if scores else 0.0
    counts = {state: sum(states[key] == state for key in keys) for state in set(states.values())}
    gate = arm.suite.gate
    passed = (
        len(results) == len(keys)
        and task_failed == 0
        and counts.get("failure", 0) == 0
        and counts.get("interrupted", 0) == 0
        and case_pass_rate >= gate.minimum_case_pass_rate
        and mean_score >= gate.minimum_mean_score
    )
    return ArmRunSummary(
        arm=arm.name,
        passed=passed,
        planned=len(keys),
        completed=len(results),
        quality_passed=quality_passed,
        task_failed=task_failed,
        execution_failed=counts.get("failure", 0),
        interrupted=counts.get("interrupted", 0),
        running=counts.get("running", 0),
        pending=counts.get("pending", 0),
        case_pass_rate=case_pass_rate,
        mean_score=mean_score,
    )


def _is_task_failure(result: ExperimentCaseResult) -> bool:
    if result.task_failed:
        return True
    return not result.passed and result.run_id is None and result.score is None


def _summarize_comparison(
    comparison: ComparisonPlan,
    completed: dict[str, ExperimentCaseResult],
) -> ExperimentComparisonSummary:
    pairs: list[ExperimentPairResult] = []
    deltas: list[float] = []
    for expected in comparison.pairs:
        baseline = completed.get(expected.baseline.model_dump_json())
        candidate = completed.get(expected.candidate.model_dump_json())
        if baseline is None or candidate is None:
            continue
        delta = None
        if baseline.score is not None and candidate.score is not None:
            delta = candidate.score - baseline.score
            deltas.append(delta)
        pairs.append(
            ExperimentPairResult(
                corpus=expected.corpus,
                baseline=baseline,
                candidate=candidate,
                score_delta_candidate_minus_baseline=delta,
            )
        )
    return ExperimentComparisonSummary(
        name=comparison.name,
        baseline=comparison.baseline,
        candidate=comparison.candidate,
        complete=len(pairs) == len(comparison.pairs),
        mean_score_delta_candidate_minus_baseline=(fmean(deltas) if deltas else None),
        pairs=tuple(pairs),
    )


def _units(
    plan: ExperimentPlan,
) -> tuple[tuple[int, SessionArmPlan, PlannedExperimentCase], ...]:
    units: list[tuple[int, SessionArmPlan, PlannedExperimentCase]] = []
    for arm in plan.arms:
        for case in arm.cases:
            units.append((len(units) + 1, arm, case))
    return tuple(units)


def _case_directory(directory: Path, ordinal: int, key: ArmCaseKey) -> Path:
    corpus = key.corpus
    name = f"{ordinal:06d}-{key.arm}-{corpus.scenario_id}-r{corpus.repeat_index:04d}"
    return directory / "cases" / name


def _write_summary(plan: ExperimentPlan, directory: Path) -> ExperimentSummary:
    summary = summarize_experiment(plan, directory)
    _write_json_atomic(directory / "summary.json", summary.model_dump(mode="json"))
    return summary


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_json_exclusive(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as destination:
        json.dump(payload, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")
