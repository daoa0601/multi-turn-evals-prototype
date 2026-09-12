"""A bounded, resumable runner for one concrete multi-environment campaign."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import sys
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

import yaml
from pydantic import Field, ValidationError, model_validator

from pydantic_multiturn_evals.evaluation import evaluate_suite
from pydantic_multiturn_evals.models import (
    EnvironmentName,
    Identifier,
    Scenario,
    StrictModel,
    SuiteSpec,
    TargetSpec,
    Text,
)
from pydantic_multiturn_evals.spec import LoadedTarget, load_suite, load_target
from pydantic_multiturn_evals.targets import build_target


class CampaignBounds(StrictModel):
    max_cases: int = Field(ge=1, le=10_000)
    max_concurrency: int = Field(ge=1, le=32)
    run_seconds: float = Field(gt=0, le=86_400)
    max_model_requests: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    judge_requests_per_case: int = Field(default=2, ge=1, le=5)


class CampaignLaneSpec(StrictModel):
    name: Identifier
    revision: Identifier
    execution_environment: Identifier
    target: Path
    interaction: Literal["real", "simulated"]
    required_env: tuple[EnvironmentName, ...] = ()
    target_requests_per_turn: int = Field(default=1, ge=0, le=10)
    target_max_output_tokens: int = Field(default=2048, ge=1, le=131_072)


class CampaignExcludedLaneSpec(StrictModel):
    name: Identifier
    execution_environment: Identifier
    reason: Text


class CampaignSpec(StrictModel):
    version: Literal[1]
    name: Identifier
    suite: Path
    repeat: Literal[1] = 1
    bounds: CampaignBounds
    lanes: tuple[CampaignLaneSpec, ...] = Field(min_length=1)
    excluded_lanes: tuple[CampaignExcludedLaneSpec, ...] = ()

    @model_validator(mode="after")
    def reject_duplicate_lanes(self) -> CampaignSpec:
        names = [lane.name for lane in self.lanes]
        names.extend(lane.name for lane in self.excluded_lanes)
        if len(names) != len(set(names)):
            raise ValueError("campaign lane names must be unique")
        return self


@dataclass(frozen=True, slots=True)
class LoadedCampaign:
    spec: CampaignSpec
    source_directory: Path


class CampaignLanePlan(StrictModel):
    name: Identifier
    revision: Identifier
    execution_environment: Identifier
    interaction: Literal["real", "simulated"]
    target: TargetSpec
    source_directory: Path
    target_requests_per_turn: int
    target_max_output_tokens: int


class CampaignUnitKey(StrictModel):
    lane: Identifier
    lane_revision: Identifier
    target: Identifier
    target_kind: Identifier
    execution_environment: Identifier
    channel: Identifier
    task: Identifier
    scenario_environment: Identifier
    scenario_revision: Identifier
    scenario_id: Identifier
    repeat_index: Literal[1] = 1


class CampaignUnitPlan(StrictModel):
    ordinal: int = Field(ge=1)
    key: CampaignUnitKey
    scenario: Scenario


class CampaignExclusion(StrictModel):
    lane: Identifier
    execution_environment: Identifier
    reason: Text


class CampaignPlan(StrictModel):
    version: Literal[1]
    name: Identifier
    suite: SuiteSpec
    lanes: tuple[CampaignLanePlan, ...]
    units: tuple[CampaignUnitPlan, ...]
    exclusions: tuple[CampaignExclusion, ...]
    bounds: CampaignBounds
    estimated_model_requests: int
    estimated_max_output_tokens: int

    @model_validator(mode="after")
    def reject_duplicate_units(self) -> CampaignPlan:
        keys = [unit.key.model_dump_json() for unit in self.units]
        if len(keys) != len(set(keys)):
            raise ValueError("campaign unit keys must be unique")
        return self


class CampaignUnitResult(StrictModel):
    key: CampaignUnitKey
    passed: bool
    score: float | None = None
    run_id: str | None = None
    duration_seconds: float = Field(ge=0)


class CampaignSummary(StrictModel):
    schema_version: Literal[1] = 1
    campaign: Identifier
    passed: bool
    planned: int
    completed: int
    quality_passed: int
    execution_failed: int
    running: int
    interrupted: int
    pending: int
    excluded_lanes: int
    quality_pass_rate: float
    completed_by_channel: dict[str, int]
    completed_by_task: dict[str, int]
    completed_by_scenario_environment: dict[str, int]
    completed_by_lane: dict[str, int]
    completed_by_execution_environment: dict[str, int]


UnitExecutor = Callable[[CampaignPlan, CampaignUnitPlan, Path], Awaitable[CampaignUnitResult]]


def load_campaign(path: str | Path) -> LoadedCampaign:
    source = Path(path).resolve()
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        spec = CampaignSpec.model_validate(payload)
    except OSError as error:
        raise ValueError(f"could not read campaign {source}: {error}") from error
    except (yaml.YAMLError, ValidationError) as error:
        raise ValueError(f"invalid campaign in {source}:\n{error}") from error
    directory = source.parent
    resolved_lanes = tuple(
        lane.model_copy(
            update={"target": _resolve(directory, lane.target)},
        )
        for lane in spec.lanes
    )
    return LoadedCampaign(
        spec=spec.model_copy(
            update={
                "suite": _resolve(directory, spec.suite),
                "lanes": resolved_lanes,
            }
        ),
        source_directory=directory,
    )


def compile_campaign(
    loaded: LoadedCampaign,
    *,
    environment: Mapping[str, str] | None = None,
) -> CampaignPlan:
    spec = loaded.spec
    suite = load_suite(spec.suite)
    values = os.environ if environment is None else environment
    lanes: list[CampaignLanePlan] = []
    exclusions = [
        CampaignExclusion(
            lane=item.name,
            execution_environment=item.execution_environment,
            reason=item.reason,
        )
        for item in spec.excluded_lanes
    ]

    for lane in spec.lanes:
        target = load_target(lane.target)
        missing = [name for name in lane.required_env if not values.get(name)]
        if missing:
            exclusions.append(
                CampaignExclusion(
                    lane=lane.name,
                    execution_environment=lane.execution_environment,
                    reason=f"missing environment variable(s): {', '.join(missing)}",
                )
            )
            continue
        lanes.append(
            CampaignLanePlan(
                name=lane.name,
                revision=lane.revision,
                execution_environment=lane.execution_environment,
                interaction=lane.interaction,
                target=target.spec,
                source_directory=target.source_directory,
                target_requests_per_turn=lane.target_requests_per_turn,
                target_max_output_tokens=lane.target_max_output_tokens,
            )
        )

    if not lanes:
        excluded = ", ".join(item.lane for item in exclusions)
        raise ValueError(f"campaign has no runnable lanes; excluded: {excluded}")

    units: list[CampaignUnitPlan] = []
    estimated_requests = 0
    estimated_tokens = 0
    for lane in lanes:
        for scenario in suite.scenarios:
            profile = _scenario_profile(scenario)
            key = CampaignUnitKey(
                lane=lane.name,
                lane_revision=lane.revision,
                target=lane.target.name,
                target_kind=lane.target.kind,
                execution_environment=lane.execution_environment,
                channel=profile["channel"],
                task=profile["task"],
                scenario_environment=profile["environment"],
                scenario_revision=profile["revision"],
                scenario_id=scenario.id,
            )
            units.append(
                CampaignUnitPlan(
                    ordinal=len(units) + 1,
                    key=key,
                    scenario=scenario,
                )
            )
            turns = suite.limits_for(scenario).max_target_turns
            actor_requests = turns * suite.actor.model_request_limit
            target_requests = turns * lane.target_requests_per_turn
            judge_requests = spec.bounds.judge_requests_per_case
            estimated_requests += actor_requests + target_requests + judge_requests
            estimated_tokens += (
                actor_requests * suite.actor.model.options.max_tokens
                + target_requests * lane.target_max_output_tokens
                + judge_requests * suite.judge.model.options.max_tokens
            )

    if len(units) > spec.bounds.max_cases:
        raise ValueError(
            f"campaign plans {len(units)} cases, above max_cases={spec.bounds.max_cases}"
        )
    if estimated_requests > spec.bounds.max_model_requests:
        raise ValueError(
            f"campaign reserves {estimated_requests} model requests, above "
            f"max_model_requests={spec.bounds.max_model_requests}"
        )
    if estimated_tokens > spec.bounds.max_output_tokens:
        raise ValueError(
            f"campaign reserves {estimated_tokens} output tokens, above "
            f"max_output_tokens={spec.bounds.max_output_tokens}"
        )

    return CampaignPlan(
        version=1,
        name=spec.name,
        suite=suite,
        lanes=tuple(lanes),
        units=tuple(units),
        exclusions=tuple(exclusions),
        bounds=spec.bounds,
        estimated_model_requests=estimated_requests,
        estimated_max_output_tokens=estimated_tokens,
    )


def save_new_plan(plan: CampaignPlan, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "plan.json"
    if path.exists():
        raise ValueError(f"campaign output already has a plan; use resume: {directory}")
    _write_json_atomic(path, plan.model_dump(mode="json"))


def load_saved_plan(directory: Path) -> CampaignPlan:
    path = directory / "plan.json"
    try:
        return CampaignPlan.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"could not read saved campaign plan {path}: {error}") from error
    except ValidationError as error:
        raise ValueError(f"invalid saved campaign plan {path}:\n{error}") from error


@contextmanager
def campaign_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another campaign process owns {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


async def execute_campaign(
    plan: CampaignPlan,
    directory: Path,
    *,
    executor: UnitExecutor | None = None,
) -> CampaignSummary:
    run_unit = executor or _execute_unit
    try:
        async with asyncio.timeout(plan.bounds.run_seconds):
            for lane in plan.lanes:
                semaphore = asyncio.Semaphore(plan.bounds.max_concurrency)

                async def bounded(
                    unit: CampaignUnitPlan,
                    limit: asyncio.Semaphore = semaphore,
                ) -> None:
                    async with limit:
                        await _run_unit(plan, unit, directory, run_unit)
                        _write_summary(plan, directory)

                await asyncio.gather(
                    *(bounded(unit) for unit in plan.units if unit.key.lane == lane.name)
                )
    except TimeoutError:
        _write_json_atomic(
            directory / "timeout.json",
            {"campaign": plan.name, "reason": "run deadline reached"},
        )
    return _write_summary(plan, directory)


async def _run_unit(
    plan: CampaignPlan,
    unit: CampaignUnitPlan,
    directory: Path,
    executor: UnitExecutor,
) -> None:
    unit_directory = _unit_directory(directory, unit)
    complete = unit_directory / "complete.json"
    failure = unit_directory / "failure.json"
    interrupted = unit_directory / "interrupted.json"
    if complete.exists() or failure.exists() or interrupted.exists():
        return
    started = unit_directory / "started.json"
    if started.exists():
        _write_json_atomic(
            interrupted,
            {
                "key": unit.key.model_dump(mode="json"),
                "reason": "a previous process started this unit without a terminal receipt",
            },
        )
        return

    unit_directory.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(
        started,
        {
            "key": unit.key.model_dump(mode="json"),
            "attempt_id": uuid4().hex,
            "started_at": _now(),
        },
    )
    try:
        result = await executor(plan, unit, unit_directory)
    except asyncio.CancelledError:
        _write_json_atomic(
            interrupted,
            {
                "key": unit.key.model_dump(mode="json"),
                "reason": "campaign execution was cancelled",
            },
        )
        raise
    except Exception as error:
        _write_json_atomic(
            failure,
            {
                "key": unit.key.model_dump(mode="json"),
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        return
    _write_json_atomic(complete, result.model_dump(mode="json"))


async def _execute_unit(
    plan: CampaignPlan,
    unit: CampaignUnitPlan,
    directory: Path,
) -> CampaignUnitResult:
    lane = next(item for item in plan.lanes if item.name == unit.key.lane)
    target = build_target(LoadedTarget(spec=lane.target, source_directory=lane.source_directory))
    suite = plan.suite.model_copy(update={"scenarios": (unit.scenario,)})
    started = perf_counter()
    result = await evaluate_suite(
        suite,
        target=target,
        max_concurrency=1,
        repeat=1,
        progress=False,
    )
    result.write_artifacts(directory)
    run_id = result.report.cases[0].output.run_id if result.report.cases else None
    gate = result.gate.cases[0]
    return CampaignUnitResult(
        key=unit.key,
        passed=gate.passed,
        score=gate.score,
        run_id=run_id,
        duration_seconds=perf_counter() - started,
    )


def summarize_campaign(plan: CampaignPlan, directory: Path) -> CampaignSummary:
    completed_results: list[CampaignUnitResult] = []
    execution_failed = 0
    running = 0
    interrupted = 0
    pending = 0
    for unit in plan.units:
        unit_directory = _unit_directory(directory, unit)
        complete = unit_directory / "complete.json"
        if complete.exists():
            completed_results.append(
                CampaignUnitResult.model_validate_json(complete.read_text(encoding="utf-8"))
            )
        elif (unit_directory / "failure.json").exists():
            execution_failed += 1
        elif (unit_directory / "interrupted.json").exists():
            interrupted += 1
        elif (unit_directory / "started.json").exists():
            running += 1
        else:
            pending += 1
    completed = len(completed_results)
    quality_passed = sum(result.passed for result in completed_results)
    quality_pass_rate = quality_passed / completed if completed else 0.0
    return CampaignSummary(
        campaign=plan.name,
        passed=(
            completed == len(plan.units)
            and execution_failed == 0
            and running == 0
            and interrupted == 0
            and quality_passed == completed
        ),
        planned=len(plan.units),
        completed=completed,
        quality_passed=quality_passed,
        execution_failed=execution_failed,
        running=running,
        interrupted=interrupted,
        pending=pending,
        excluded_lanes=len(plan.exclusions),
        quality_pass_rate=quality_pass_rate,
        completed_by_channel=_counts(completed_results, "channel"),
        completed_by_task=_counts(completed_results, "task"),
        completed_by_scenario_environment=_counts(completed_results, "scenario_environment"),
        completed_by_lane=_counts(completed_results, "lane"),
        completed_by_execution_environment=_counts(completed_results, "execution_environment"),
    )


def _write_summary(plan: CampaignPlan, directory: Path) -> CampaignSummary:
    summary = summarize_campaign(plan, directory)
    _write_json_atomic(directory / "summary.json", summary.model_dump(mode="json"))
    return summary


def _counts(results: list[CampaignUnitResult], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        value = getattr(result.key, field)
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _scenario_profile(scenario: Scenario) -> dict[str, str]:
    return {
        field: _one_prefixed_tag(scenario, f"{field}.")
        for field in ("channel", "task", "environment", "revision")
    }


def _one_prefixed_tag(scenario: Scenario, prefix: str) -> str:
    values = sorted(tag.removeprefix(prefix) for tag in scenario.tags if tag.startswith(prefix))
    if len(values) != 1:
        raise ValueError(f"scenario {scenario.id!r} needs exactly one {prefix.rstrip('.')} tag")
    return values[0]


def _unit_directory(directory: Path, unit: CampaignUnitPlan) -> Path:
    return directory / "cases" / (f"{unit.ordinal:04d}-{unit.key.lane}-{unit.key.scenario_id}")


def _resolve(directory: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (directory / path).resolve()


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


def _now() -> str:
    return datetime.now(UTC).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-scale-campaign")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("campaign", type=Path)
        command.add_argument("--out", type=Path, required=True)
    resume = commands.add_parser("resume")
    resume.add_argument("directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "resume":
            directory = args.directory.resolve()
            with campaign_lock(directory):
                plan = load_saved_plan(directory)
                summary = asyncio.run(execute_campaign(plan, directory))
        else:
            directory = args.out.resolve()
            plan = compile_campaign(load_campaign(args.campaign))
            with campaign_lock(directory):
                save_new_plan(plan, directory)
                summary = (
                    _write_summary(plan, directory)
                    if args.command == "plan"
                    else asyncio.run(execute_campaign(plan, directory))
                )
        print(summary.model_dump_json(indent=2))
        return 0 if args.command == "plan" or summary.passed else 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
