from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from pydantic_multiturn_evals import (
    Artifact,
    ArtifactLimits,
    Campaign,
    Case,
    ComponentDefinition,
    Evidence,
    Failed,
    NativeCheck,
    ResourceClaim,
    Shard,
    TrialBudget,
    Variant,
    merge_reports,
    run,
)
from pydantic_multiturn_evals.harnesses import FunctionHarness
from pydantic_multiturn_evals.model import TraceRecorder, Trial


def test_shared_capacity_covers_execution_and_cleanup(tmp_path: Path) -> None:
    active = 0
    maximum = 0

    async def measured(
        _case: Case,
        _trial: Trial,
        _trace: TraceRecorder,
    ) -> Evidence:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep(0.01)
        finally:
            await asyncio.sleep(0.01)
            active -= 1
        return Evidence(checks=(NativeCheck(name="done", passed=True, reason="finished"),))

    harness = FunctionHarness(
        component=ComponentDefinition(kind="measured", version="1"),
        function=measured,
    )
    campaign = Campaign(
        name="shared-capacity",
        revision="1",
        cases=(Case(name="work", task="Run", required_checks=("done",)),),
        variants=(
            Variant(
                name="left",
                harness=harness,
                resources=(ResourceClaim(name="sandbox"),),
            ),
            Variant(
                name="right",
                harness=harness,
                resources=(ResourceClaim(name="sandbox"),),
            ),
        ),
        samples=3,
    )

    report = asyncio.run(run(campaign, directory=tmp_path, workers=6, capacities={"sandbox": 1}))

    assert report.exit_code == 0
    assert maximum == 1


def test_capacity_wait_does_not_spend_the_case_budget(tmp_path: Path) -> None:
    async def timed(case: Case, _trial: Trial, _trace: TraceRecorder) -> Evidence:
        await asyncio.sleep(0.08 if case.name == "occupy" else 0.01)
        return Evidence(checks=(NativeCheck(name="done", passed=True, reason="finished"),))

    harness = FunctionHarness(
        component=ComponentDefinition(kind="timed", version="1"),
        function=timed,
    )
    campaign = Campaign(
        name="admission-budget",
        revision="1",
        cases=(
            Case(
                name="occupy",
                task="Hold capacity",
                required_checks=("done",),
                budget=TrialBudget(seconds=0.2),
            ),
            Case(
                name="short",
                task="Run after waiting",
                required_checks=("done",),
                budget=TrialBudget(seconds=0.04),
            ),
        ),
        variants=(
            Variant(
                name="only",
                harness=harness,
                resources=(ResourceClaim(name="sandbox"),),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path, workers=2, capacities={"sandbox": 1}))

    assert report.exit_code == 0
    assert all(result.passed for result in report.results)


def test_full_resource_does_not_idle_an_unrelated_resource(tmp_path: Path) -> None:
    unrelated_started = asyncio.Event()

    async def coordinated(
        _case: Case,
        trial: Trial,
        _trace: TraceRecorder,
    ) -> Evidence:
        if trial.key.variant == "scarce-first":
            await asyncio.wait_for(unrelated_started.wait(), timeout=0.2)
        elif trial.key.variant == "unrelated":
            unrelated_started.set()
        return Evidence(checks=(NativeCheck(name="done", passed=True, reason="finished"),))

    harness = FunctionHarness(
        component=ComponentDefinition(kind="coordinated", version="1"),
        function=coordinated,
    )
    campaign = Campaign(
        name="work-conserving",
        revision="1",
        cases=(Case(name="work", task="Run", required_checks=("done",)),),
        variants=(
            Variant(
                name="scarce-first",
                harness=harness,
                resources=(ResourceClaim(name="scarce"),),
            ),
            Variant(
                name="scarce-second",
                harness=harness,
                resources=(ResourceClaim(name="scarce"),),
            ),
            Variant(
                name="unrelated",
                harness=harness,
                resources=(ResourceClaim(name="other"),),
            ),
        ),
    )

    report = asyncio.run(
        run(
            campaign,
            directory=tmp_path,
            workers=2,
            capacities={"scarce": 1, "other": 1},
        )
    )

    assert report.exit_code == 0


def test_static_shards_keep_pairs_together_and_merge_exactly(tmp_path: Path) -> None:
    async def complete(_case: Case, _trial: Trial, _trace: TraceRecorder) -> Evidence:
        return Evidence(checks=(NativeCheck(name="done", passed=True, reason="finished"),))

    harness = FunctionHarness(
        component=ComponentDefinition(kind="shard-proof", version="1"),
        function=complete,
    )
    campaign = Campaign(
        name="shards",
        revision="1",
        cases=tuple(
            Case(name=f"case-{index}", task="Run", required_checks=("done",)) for index in range(3)
        ),
        variants=(
            Variant(name="left", harness=harness),
            Variant(name="right", harness=harness),
        ),
        samples=2,
    )
    roots = (tmp_path / "shard-0", tmp_path / "shard-1")

    reports = tuple(
        asyncio.run(run(campaign, directory=root, shard=Shard(index=index, count=2)))
        for index, root in enumerate(roots)
    )

    for report in reports:
        owners: dict[tuple[str, int], set[str]] = {}
        for result in report.results:
            owners.setdefault((result.key.case, result.key.sample), set()).add(result.key.variant)
        assert all(variants == {"left", "right"} for variants in owners.values())
    merged = merge_reports(campaign, roots)
    assert len(merged.results) == 12
    assert len(merged.compare("left", "right").pairs) == 6


def test_merge_rejects_a_missing_trial_receipt(tmp_path: Path) -> None:
    async def complete(_case: Case, _trial: Trial, _trace: TraceRecorder) -> Evidence:
        return Evidence(checks=(NativeCheck(name="done", passed=True, reason="finished"),))

    campaign = Campaign(
        name="shard-gap",
        revision="1",
        cases=(Case(name="case", task="Run", required_checks=("done",)),),
        variants=(
            Variant(
                name="only",
                harness=FunctionHarness(
                    component=ComponentDefinition(kind="gap-proof", version="1"),
                    function=complete,
                ),
            ),
        ),
    )
    roots = (tmp_path / "shard-0", tmp_path / "shard-1")
    for index, root in enumerate(roots):
        asyncio.run(run(campaign, directory=root, shard=Shard(index=index, count=2)))
    report_path = roots[0] / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["results"] = []
    report["exit_code"] = 0
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="missing trial result"):
        merge_reports(campaign, roots)


def test_artifact_limits_reject_evidence_before_evaluation(tmp_path: Path) -> None:
    async def oversized(_case: Case, trial: Trial, _trace: TraceRecorder) -> Evidence:
        (trial.directory / "large.txt").write_text("too large", encoding="utf-8")
        return Evidence(
            checks=(NativeCheck(name="done", passed=True, reason="finished"),),
            artifacts=(Artifact(name="large", path="large.txt"),),
        )

    campaign = Campaign(
        name="artifact-limit",
        revision="1",
        cases=(Case(name="case", task="Run", required_checks=("done",)),),
        variants=(
            Variant(
                name="only",
                harness=FunctionHarness(
                    component=ComponentDefinition(kind="large-artifact", version="1"),
                    function=oversized,
                ),
            ),
        ),
    )

    report = asyncio.run(
        run(
            campaign,
            directory=tmp_path,
            artifact_limits=ArtifactLimits(files=1, file_bytes=4, total_bytes=4),
        )
    )

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert terminal.failure.phase == "evidence"
    assert "file limit is 4" in terminal.failure.message
