"""Bounded campaign execution with durable per-attempt receipts."""

from __future__ import annotations

import asyncio
import fcntl
import json
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import TypeAdapter, computed_field, model_validator

from pydantic_multiturn_evals.evaluation import EvaluationError, HarnessError, evaluate
from pydantic_multiturn_evals.model import (
    Artifact,
    ArtifactFact,
    ArtifactLimits,
    Campaign,
    Case,
    CheckFact,
    Completed,
    ComponentDefinition,
    Evidence,
    Failed,
    Failure,
    Interrupted,
    LifecycleFact,
    ModelCallFact,
    ResourceClaim,
    Shard,
    StrictModel,
    TraceEvent,
    Trial,
    TrialKey,
    TrialResult,
    Variant,
)
from pydantic_multiturn_evals.tracing import (
    LocalTrace,
    TraceExporter,
    write_export_errors,
)


class ManifestVariant(StrictModel):
    name: str
    harness: ComponentDefinition
    evaluators: tuple[ComponentDefinition, ...]
    resources: tuple[ResourceClaim, ...]


class CampaignManifest(StrictModel):
    protocol: Literal[1] = 1
    name: str
    revision: str
    samples: int
    cases: tuple[Case, ...]
    variants: tuple[ManifestVariant, ...]
    artifact_limits: ArtifactLimits
    shard: Shard | None = None


class TraceExportState(StrictModel):
    export_id: str
    succeeded: tuple[ComponentDefinition, ...] = ()


class PairResult(StrictModel):
    case: str
    sample: int
    left_passed: bool
    right_passed: bool


class Comparison(StrictModel):
    left: str
    right: str
    pairs: tuple[PairResult, ...]

    @computed_field
    @property
    def only_left_passed(self) -> int:
        return sum(pair.left_passed and not pair.right_passed for pair in self.pairs)

    @computed_field
    @property
    def only_right_passed(self) -> int:
        return sum(pair.right_passed and not pair.left_passed for pair in self.pairs)


class Report(StrictModel):
    campaign: str
    revision: str
    executed_trials: int
    results: tuple[TrialResult, ...]
    exit_code: Literal[0, 1, 2]

    @model_validator(mode="after")
    def match_exit_code_to_results(self) -> Report:
        expected = _exit_code(self.results)
        if self.exit_code != expected:
            raise ValueError(f"exit code {self.exit_code} does not match results ({expected})")
        return self

    def compare(self, left: str, right: str) -> Comparison:
        by_identity = {
            (result.key.case, result.key.sample, result.key.variant): result
            for result in self.results
        }
        identities = sorted(
            {
                (result.key.case, result.key.sample)
                for result in self.results
                if result.key.variant in (left, right)
            }
        )
        pairs: list[PairResult] = []
        for case, sample in identities:
            try:
                left_result = by_identity[(case, sample, left)]
                right_result = by_identity[(case, sample, right)]
            except KeyError as error:
                raise ValueError(
                    f"cannot compare {left!r} and {right!r}: missing exact pair "
                    f"for case {case!r}, sample {sample}"
                ) from error
            pairs.append(
                PairResult(
                    case=case,
                    sample=sample,
                    left_passed=left_result.passed,
                    right_passed=right_result.passed,
                )
            )
        if not pairs:
            raise ValueError(f"no results for variants {left!r} and {right!r}")
        return Comparison(left=left, right=right, pairs=tuple(pairs))


async def run(
    campaign: Campaign,
    *,
    directory: Path,
    workers: int = 4,
    retry_interrupted: bool = False,
    exporters: Sequence[TraceExporter] = (),
    export_timeout_seconds: float = 10,
    capacities: Mapping[str, int] | None = None,
    artifact_limits: ArtifactLimits | None = None,
    shard: Shard | None = None,
) -> Report:
    """Run missing trials and reuse terminal receipts from earlier calls."""

    if workers < 1:
        raise ValueError("workers must be positive")
    if export_timeout_seconds <= 0:
        raise ValueError("export timeout must be positive")
    _validate_exporters(exporters)
    capacity = _CapacityPool(campaign, capacities or {})
    artifact_limits = artifact_limits or ArtifactLimits()
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(directory / ".lock"):
        _ensure_manifest(directory, _manifest(campaign, artifact_limits, shard))
        results: list[TrialResult] = []
        pending: list[tuple[TrialKey, Case, Variant, int]] = []
        for key, case, variant in _trials(campaign, shard):
            prior, attempt = _prior_result(directory, key, retry_interrupted)
            if prior is not None:
                results.append(prior)
            else:
                pending.append((key, case, variant, attempt))
        executed = len(pending)
        results.extend(
            await _run_pending(
                pending,
                root=directory,
                workers=workers,
                capacity=capacity,
                artifact_limits=artifact_limits,
            )
        )

        ordered = tuple(sorted(results, key=_result_order))
        report = Report(
            campaign=campaign.name,
            revision=campaign.revision,
            executed_trials=executed,
            results=ordered,
            exit_code=_exit_code(ordered),
        )
        _write_json(directory / "report.json", report.model_dump(mode="json"))
        await _export_traces(
            tuple(
                (result, _attempt_directory(directory, result.key, result.attempt))
                for result in ordered
            ),
            exporters,
            workers=workers,
            timeout_seconds=export_timeout_seconds,
        )
        return report


def read_report(directory: Path) -> Report:
    return Report.model_validate_json((directory / "report.json").read_text(encoding="utf-8"))


class _CapacityPool:
    """Single-writer capacity accounting for the campaign scheduler."""

    def __init__(self, campaign: Campaign, capacities: Mapping[str, int]) -> None:
        invalid = {name: slots for name, slots in capacities.items() if slots < 1}
        if invalid:
            name, slots = next(iter(invalid.items()))
            raise ValueError(f"capacity {name!r} must be positive, got {slots}")
        claimed = {
            claim.name
            for variant in campaign.variants
            for claim in variant.resources
        }
        missing = claimed - capacities.keys()
        if missing:
            raise ValueError(f"missing capacities for resources: {', '.join(sorted(missing))}")
        for variant in campaign.variants:
            for claim in variant.resources:
                if claim.slots > capacities[claim.name]:
                    raise ValueError(
                        f"variant {variant.name!r} claims {claim.slots} slots from "
                        f"{claim.name!r}, whose capacity is {capacities[claim.name]}"
                    )
        self._available = dict(capacities)

    def fits(self, claims: Sequence[ResourceClaim]) -> bool:
        return all(self._available[claim.name] >= claim.slots for claim in claims)

    def acquire(self, claims: Sequence[ResourceClaim]) -> None:
        if not self.fits(claims):
            raise RuntimeError("resource claims were acquired before they fit")
        for claim in claims:
            self._available[claim.name] -= claim.slots

    def release(self, claims: Sequence[ResourceClaim]) -> None:
        for claim in claims:
            self._available[claim.name] += claim.slots


async def _run_pending(
    pending: list[tuple[TrialKey, Case, Variant, int]],
    *,
    root: Path,
    workers: int,
    capacity: _CapacityPool,
    artifact_limits: ArtifactLimits,
) -> list[TrialResult]:
    """Run the oldest feasible trials without parking workers on a full resource."""

    active: set[asyncio.Task[TrialResult]] = set()
    results: list[TrialResult] = []

    async def execute(
        key: TrialKey,
        case: Case,
        variant: Variant,
        attempt: int,
    ) -> TrialResult:
        try:
            return await _run_trial(key, case, variant, attempt, root, artifact_limits)
        finally:
            capacity.release(variant.resources)

    try:
        while pending or active:
            while pending and len(active) < workers:
                index = next(
                    (
                        item_index
                        for item_index, (_, _, variant, _) in enumerate(pending)
                        if capacity.fits(variant.resources)
                    ),
                    None,
                )
                if index is None:
                    break
                key, case, variant, attempt = pending.pop(index)
                capacity.acquire(variant.resources)
                active.add(asyncio.create_task(execute(key, case, variant, attempt)))
            if not active:
                raise RuntimeError("no pending trial fits the configured resource capacities")
            done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            results.extend(task.result() for task in done)
    finally:
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
    return results


async def _run_trial(
    key: TrialKey,
    case: Case,
    variant: Variant,
    attempt: int,
    root: Path,
    artifact_limits: ArtifactLimits,
) -> TrialResult:
    directory = _attempt_directory(root, key, attempt)
    directory.mkdir(parents=True, exist_ok=False)
    _write_json(
        directory / "started.json",
        {"key": key.model_dump(mode="json"), "attempt": attempt},
    )
    trial = Trial(key=key, attempt=attempt, directory=directory, seed=key.sample)
    trace = LocalTrace(key, attempt, directory / "events.jsonl")
    started = time.monotonic()
    trace.record(LifecycleFact(phase="started"))
    terminal: Completed | Failed | Interrupted
    evidence: Evidence | None = None
    cancelled = False
    try:
        async with asyncio.timeout(case.budget.seconds):
            evidence = await _execute_harness(variant, case, trial, trace)
            for check in evidence.checks:
                trace.record(CheckFact(check=check))
            for artifact in evidence.artifacts:
                trace.record(ArtifactFact(artifact=artifact))
            _validate_artifacts(evidence.artifacts, directory, evidence, artifact_limits)
            judgments = await evaluate(case, evidence, variant.evaluators, directory, trace)
            terminal = Completed(evidence=evidence, judgments=judgments)
    except asyncio.CancelledError:
        cancelled = True
        terminal = Interrupted(reason="campaign task was cancelled")
    except TimeoutError as error:
        terminal = Failed(
            failure=Failure(phase="timeout", message=str(error) or "trial exceeded its budget")
        )
    except HarnessError as error:
        terminal = Failed(
            failure=Failure(
                phase=error.phase,
                message=str(error),
                exception_type=type(error.__cause__ or error).__name__,
            ),
            evidence=error.evidence,
        )
    except EvaluationError as error:
        terminal = Failed(
            failure=Failure(
                phase="evaluate",
                message=str(error),
                exception_type=type(error).__name__,
            ),
            evidence=evidence,
        )
    except Exception as error:
        terminal = Failed(
            failure=Failure(
                phase="internal",
                message=str(error) or repr(error),
                exception_type=type(error).__name__,
            ),
            evidence=evidence,
        )

    result = TrialResult(
        key=key,
        attempt=attempt,
        duration_seconds=time.monotonic() - started,
        terminal=terminal,
        model_calls=tuple(
            event.fact.call for event in trace.events if isinstance(event.fact, ModelCallFact)
        ),
    )
    try:
        trace.record(LifecycleFact(phase="terminal", detail=terminal.kind))
        _write_json(directory / "result.json", result.model_dump(mode="json"))
    finally:
        trace.close()
    if cancelled:
        raise asyncio.CancelledError
    return result


async def _execute_harness(
    variant: Variant,
    case: Case,
    trial: Trial,
    trace: LocalTrace,
) -> Evidence:
    try:
        output = await variant.harness.run(case, trial, trace)
        return TypeAdapter(Evidence).validate_python(output)
    except HarnessError:
        raise
    except Exception as error:
        component = variant.harness.component
        identity = f"{component.kind}@{component.version}"
        raise HarnessError("execute", f"harness {identity} failed: {error}") from error


async def _export_traces(
    trials: Sequence[tuple[TrialResult, Path]],
    exporters: Sequence[TraceExporter],
    *,
    workers: int,
    timeout_seconds: float,
) -> None:
    if not exporters or not trials:
        return
    queue: asyncio.Queue[tuple[TrialResult, Path] | None] = asyncio.Queue(maxsize=workers * 2)

    async def worker() -> None:
        while True:
            item = await queue.get()
            if item is None:
                return
            result, directory = item
            await _export_trace(result, directory, exporters, timeout_seconds)

    worker_count = min(workers, len(trials))
    async with asyncio.TaskGroup() as tasks:
        for _ in range(worker_count):
            tasks.create_task(worker())
        for item in trials:
            await queue.put(item)
        for _ in range(worker_count):
            await queue.put(None)


async def _export_trace(
    result: TrialResult,
    directory: Path,
    exporters: Sequence[TraceExporter],
    timeout_seconds: float,
) -> None:
    errors: list[dict[str, str]] = []
    state_path = directory / "trace-exports.json"
    try:
        events = tuple(
            TraceEvent.model_validate_json(line)
            for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
        )
        state = (
            TraceExportState.model_validate_json(state_path.read_text(encoding="utf-8"))
            if state_path.exists()
            else TraceExportState(export_id=uuid4().hex)
        )
    except Exception as error:
        errors.append({"exporter": "local-trace-reader", "error": str(error) or repr(error)})
        write_export_errors(directory / "trace-export-errors.json", errors)
        return
    _write_json(state_path, state.model_dump(mode="json"))
    succeeded = list(state.succeeded)
    for exporter in exporters:
        if exporter.component in succeeded:
            continue
        try:
            async with asyncio.timeout(timeout_seconds):
                await exporter.export(result.key, events, result, state.export_id)
            succeeded.append(exporter.component)
            _write_json(
                state_path,
                TraceExportState(
                    export_id=state.export_id,
                    succeeded=tuple(succeeded),
                ).model_dump(mode="json"),
            )
        except Exception as error:
            errors.append(
                {
                    "exporter": f"{type(exporter).__module__}.{type(exporter).__qualname__}",
                    "error": str(error) or f"timed out after {timeout_seconds:g} seconds",
                }
            )
    write_export_errors(directory / "trace-export-errors.json", errors)


def _validate_exporters(exporters: Sequence[TraceExporter]) -> None:
    definitions = [exporter.component for exporter in exporters]
    for index, definition in enumerate(definitions):
        if definition in definitions[:index]:
            raise ValueError(
                f"trace exporter definitions must be unique: {definition.kind}@{definition.version}"
            )


def _validate_artifacts(
    artifacts: Sequence[Artifact],
    directory: Path,
    evidence: Evidence,
    limits: ArtifactLimits,
) -> None:
    if len(artifacts) > limits.files:
        raise HarnessError(
            "evidence",
            f"harness returned {len(artifacts)} artifacts; limit is {limits.files}",
            evidence=evidence,
        )
    root = directory.resolve()
    total_bytes = 0
    for value in artifacts:
        path = Path(value.path)
        if path.is_absolute():
            raise HarnessError(
                "evidence",
                f"artifact path must be relative: {path}",
                evidence=evidence,
            )
        candidate = directory / path
        resolved = candidate.resolve()
        if (
            _contains_symlink(candidate, directory)
            or root not in resolved.parents
            or not resolved.is_file()
        ):
            raise HarnessError(
                "evidence",
                f"artifact must name a regular non-symlink file inside the trial directory: {path}",
                evidence=evidence,
            )
        size = resolved.stat().st_size
        if size > limits.file_bytes:
            raise HarnessError(
                "evidence",
                f"artifact {value.name!r} is {size} bytes; file limit is {limits.file_bytes}",
                evidence=evidence,
            )
        total_bytes += size
    if total_bytes > limits.total_bytes:
        raise HarnessError(
            "evidence",
            f"artifacts total {total_bytes} bytes; limit is {limits.total_bytes}",
            evidence=evidence,
        )


def _manifest(
    campaign: Campaign,
    artifact_limits: ArtifactLimits,
    shard: Shard | None,
) -> CampaignManifest:
    return CampaignManifest(
        name=campaign.name,
        revision=campaign.revision,
        samples=campaign.samples,
        cases=campaign.cases,
        variants=tuple(
            ManifestVariant(
                name=variant.name,
                harness=variant.harness.component,
                evaluators=tuple(evaluator.component for evaluator in variant.evaluators),
                resources=variant.resources,
            )
            for variant in campaign.variants
        ),
        artifact_limits=artifact_limits,
        shard=shard,
    )


def _ensure_manifest(directory: Path, manifest: CampaignManifest) -> None:
    path = directory / "campaign.json"
    authored = manifest.model_dump(mode="json")
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != authored:
            raise ValueError("run directory contains a different campaign definition")
        return
    _write_json(path, authored)


def _trials(
    campaign: Campaign,
    shard: Shard | None = None,
) -> Iterator[tuple[TrialKey, Case, Variant]]:
    for case_index, case in enumerate(campaign.cases):
        for sample in range(1, campaign.samples + 1):
            pair_index = case_index * campaign.samples + sample - 1
            if shard is not None and pair_index % shard.count != shard.index:
                continue
            for variant in campaign.variants:
                yield (
                    TrialKey(
                        campaign=campaign.name,
                        revision=campaign.revision,
                        case=case.name,
                        variant=variant.name,
                        sample=sample,
                    ),
                    case,
                    variant,
                )


def _prior_result(
    root: Path,
    key: TrialKey,
    retry_interrupted: bool,
) -> tuple[TrialResult | None, int]:
    trial_root = _trial_directory(root, key)
    attempts = list(trial_root.glob("attempt-*")) if trial_root.exists() else []
    if not attempts:
        return None, 1
    latest = max(attempts, key=_attempt_number)
    attempt = _attempt_number(latest)
    result_path = latest / "result.json"
    if result_path.exists():
        result = TrialResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        if result.key != key or result.attempt != attempt:
            raise ValueError(
                f"receipt identity does not match its attempt directory: {result_path}"
            )
    elif (latest / "started.json").exists():
        result = TrialResult(
            key=key,
            attempt=attempt,
            duration_seconds=0,
            terminal=Interrupted(reason="a previous process stopped without a terminal receipt"),
        )
        _write_json(result_path, result.model_dump(mode="json"))
    else:
        raise ValueError(f"incomplete attempt directory: {latest}")
    if retry_interrupted and isinstance(result.terminal, Interrupted):
        return None, attempt + 1
    return result, attempt


def _trial_directory(root: Path, key: TrialKey) -> Path:
    return root / "trials" / key.case / key.variant / f"sample-{key.sample:04d}"


def _attempt_directory(root: Path, key: TrialKey, attempt: int) -> Path:
    return _trial_directory(root, key) / f"attempt-{attempt:04d}"


def _attempt_number(path: Path) -> int:
    try:
        return int(path.name.removeprefix("attempt-"))
    except ValueError as error:
        raise ValueError(f"invalid attempt directory name: {path}") from error


def _result_order(result: TrialResult) -> tuple[str, int, str]:
    return result.key.case, result.key.sample, result.key.variant


def _contains_symlink(path: Path, root: Path) -> bool:
    current = path
    while current != root:
        if current.is_symlink():
            return True
        if root not in current.parents:
            return False
        current = current.parent
    return False


def _exit_code(results: Sequence[TrialResult]) -> Literal[0, 1, 2]:
    if any(isinstance(result.terminal, (Failed, Interrupted)) for result in results):
        return 2
    if any(not result.passed for result in results):
        return 1
    return 0


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    stream = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"campaign is already running in {path.parent}") from error
        yield
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()
