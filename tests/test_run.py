from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from examples.offline_campaign import campaign
from pydantic_multiturn_evals import (
    Campaign,
    Case,
    ComponentDefinition,
    Criterion,
    EvaluationInput,
    Evidence,
    Failed,
    FunctionEvaluator,
    Interrupted,
    Judgment,
    NativeCheck,
    PublicOutcome,
    TraceEvent,
    TrialResult,
    Variant,
    read_report,
    run,
)
from pydantic_multiturn_evals.harnesses import FunctionHarness
from pydantic_multiturn_evals.model import TraceRecorder, Trial, TrialKey


def test_offline_campaign_runs_real_targets_compares_exact_pairs_and_resumes(
    tmp_path: Path,
) -> None:
    first = asyncio.run(run(campaign, directory=tmp_path, workers=3))

    assert first.executed_trials == 8
    assert first.exit_code == 1
    comparison = first.compare("baseline", "candidate")
    assert comparison.only_right_passed == 2
    assert [(pair.case, pair.sample) for pair in comparison.pairs] == [
        ("account-access", 1),
        ("account-access", 2),
    ]
    target_starts = tuple(tmp_path.glob("trials/**/target-started.json"))
    assert len(target_starts) == 4
    assert all(
        "case" not in json.loads(path.read_text(encoding="utf-8"))["start_keys"]
        for path in target_starts
    )
    assert next(tmp_path.glob("trials/**/job-output.txt")).read_text(encoding="utf-8") == (
        "I cannot access my account.\n"
    )

    second = asyncio.run(run(campaign, directory=tmp_path, workers=3))

    assert second.executed_trials == 0
    assert second.results == first.results
    assert read_report(tmp_path) == second
    assert len(tuple(tmp_path.glob("trials/**/target-started.json"))) == 4


@dataclass(frozen=True, slots=True)
class BrokenExporter:
    component = ComponentDefinition(kind="test-exporter", version="1")

    async def export(
        self,
        key: TrialKey,
        events: Sequence[TraceEvent],
        result: TrialResult,
        export_id: str,
    ) -> None:
        del key, events, result, export_id
        raise RuntimeError("collector is offline")


@dataclass(slots=True)
class RecordingExporter:
    component = ComponentDefinition(kind="test-exporter", version="1")
    export_ids: list[str]

    async def export(
        self,
        key: TrialKey,
        events: Sequence[TraceEvent],
        result: TrialResult,
        export_id: str,
    ) -> None:
        del key, events, result
        self.export_ids.append(export_id)


@dataclass(slots=True)
class HangingExporter:
    report_path: Path
    component = ComponentDefinition(kind="hanging-exporter", version="1")
    saw_durable_report: bool = False

    async def export(
        self,
        key: TrialKey,
        events: Sequence[TraceEvent],
        result: TrialResult,
        export_id: str,
    ) -> None:
        del key, events, result, export_id
        self.saw_durable_report = self.report_path.is_file()
        await asyncio.Event().wait()


def test_trace_export_failure_cannot_change_a_durable_result(tmp_path: Path) -> None:
    async def direct(case: Case, trial: Trial, trace: TraceRecorder) -> Evidence:
        del case, trial, trace
        return Evidence(
            public=PublicOutcome(output={"answer": 42}),
            checks=(NativeCheck(name="answer", passed=True, reason="answer is 42"),),
        )

    direct_campaign = Campaign(
        name="trace-proof",
        revision="1",
        cases=(Case(name="answer", task="Compute", required_checks=("answer",)),),
        variants=(
            Variant(
                name="direct",
                harness=FunctionHarness(
                    component=ComponentDefinition(kind="direct", version="1"),
                    function=direct,
                ),
            ),
        ),
    )

    report = asyncio.run(run(direct_campaign, directory=tmp_path, exporters=(BrokenExporter(),)))

    assert report.exit_code == 0
    result_path = next(tmp_path.glob("trials/**/result.json"))
    assert json.loads(result_path.read_text(encoding="utf-8"))["terminal"]["kind"] == "completed"
    error_path = next(tmp_path.glob("trials/**/trace-export-errors.json"))
    assert "collector is offline" in error_path.read_text(encoding="utf-8")

    retry = RecordingExporter(export_ids=[])
    resumed = asyncio.run(run(direct_campaign, directory=tmp_path, exporters=(retry,)))

    assert resumed.executed_trials == 0
    assert len(retry.export_ids) == 1
    state_path = next(tmp_path.glob("trials/**/trace-exports.json"))
    assert json.loads(state_path.read_text(encoding="utf-8"))["export_id"] == retry.export_ids[0]
    assert not error_path.exists()


def test_hung_trace_export_is_bounded_after_the_report_is_durable(tmp_path: Path) -> None:
    async def direct(_case: Case, _trial: Trial, _trace: TraceRecorder) -> Evidence:
        return Evidence(checks=(NativeCheck(name="done", passed=True, reason="done"),))

    direct_campaign = Campaign(
        name="trace-timeout",
        revision="1",
        cases=(Case(name="case", task="Run", required_checks=("done",)),),
        variants=(
            Variant(
                name="direct",
                harness=FunctionHarness(
                    component=ComponentDefinition(kind="direct", version="1"),
                    function=direct,
                ),
            ),
        ),
    )
    exporter = HangingExporter(report_path=tmp_path / "report.json")

    report = asyncio.run(
        run(
            direct_campaign,
            directory=tmp_path,
            exporters=(exporter,),
            export_timeout_seconds=0.05,
        )
    )

    assert report.exit_code == 0
    assert exporter.saw_durable_report is True
    error_path = next(tmp_path.glob("trials/**/trace-export-errors.json"))
    assert "timed out after 0.05 seconds" in error_path.read_text(encoding="utf-8")


def test_evaluator_failure_is_reported_with_harness_evidence(tmp_path: Path) -> None:
    async def direct(_case: Case, _trial: Trial, _trace: TraceRecorder) -> Evidence:
        return Evidence(public=PublicOutcome(output={"answer": 42}))

    async def broken(_value: EvaluationInput) -> tuple[Judgment, ...]:
        raise RuntimeError("judge unavailable")

    failed_campaign = Campaign(
        name="evaluator-failure",
        revision="1",
        cases=(
            Case(
                name="case",
                task="Run",
                criteria=(Criterion(name="correct", instruction="The answer is correct."),),
            ),
        ),
        variants=(
            Variant(
                name="direct",
                harness=FunctionHarness(
                    component=ComponentDefinition(kind="direct", version="1"),
                    function=direct,
                ),
                evaluators=(
                    FunctionEvaluator(
                        component=ComponentDefinition(kind="broken-judge", version="3"),
                        function=broken,
                    ),
                ),
            ),
        ),
    )

    report = asyncio.run(run(failed_campaign, directory=tmp_path))

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert terminal.failure.phase == "evaluate"
    assert terminal.failure.message == "evaluator broken-judge@3 failed: judge unavailable"
    assert terminal.evidence is not None
    assert terminal.evidence.public.output == {"answer": 42}


def test_interrupted_trial_requires_an_explicit_retry(tmp_path: Path) -> None:
    started = asyncio.Event()

    async def interrupt_once(
        _case: Case,
        trial: Trial,
        _trace: TraceRecorder,
    ) -> Evidence:
        if trial.attempt == 1:
            started.set()
            await asyncio.Event().wait()
        return Evidence(checks=(NativeCheck(name="done", passed=True, reason="retry completed"),))

    interrupted_campaign = Campaign(
        name="interruption",
        revision="1",
        cases=(Case(name="case", task="Run", required_checks=("done",)),),
        variants=(
            Variant(
                name="only",
                harness=FunctionHarness(
                    component=ComponentDefinition(kind="interrupt-once", version="1"),
                    function=interrupt_once,
                ),
            ),
        ),
    )

    async def exercise() -> None:
        task = asyncio.create_task(run(interrupted_campaign, directory=tmp_path))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        skipped = await run(interrupted_campaign, directory=tmp_path)
        assert skipped.executed_trials == 0
        assert isinstance(skipped.results[0].terminal, Interrupted)

        retried = await run(
            interrupted_campaign,
            directory=tmp_path,
            retry_interrupted=True,
        )
        assert retried.executed_trials == 1
        assert retried.results[0].attempt == 2
        assert retried.exit_code == 0

    asyncio.run(exercise())


def test_run_directory_rejects_a_changed_campaign(tmp_path: Path) -> None:
    asyncio.run(run(campaign, directory=tmp_path))
    changed = Campaign(
        name=campaign.name,
        revision="2",
        cases=campaign.cases,
        variants=campaign.variants,
        samples=campaign.samples,
    )

    with pytest.raises(ValueError, match="different campaign definition"):
        asyncio.run(run(changed, directory=tmp_path))
