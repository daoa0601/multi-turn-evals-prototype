from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from pydantic_multiturn_evals import campaign
from pydantic_multiturn_evals.campaign import (
    CampaignBounds,
    CampaignLaneSpec,
    CampaignSpec,
    CampaignUnitResult,
    LoadedCampaign,
    compile_campaign,
    execute_campaign,
    load_saved_plan,
    main,
    save_new_plan,
)
from pydantic_multiturn_evals.models import (
    ActorBrief,
    PydanticAITargetSpec,
    Scenario,
    SuiteSpec,
)
from pydantic_multiturn_evals.spec import load_suite

ROOT = Path(__file__).parents[1]


def scenario(identifier: str = "example") -> Scenario:
    return Scenario(
        id=identifier,
        first_prompt="Help me.",
        actor=ActorBrief(persona="A user", goal="Get help"),
        judge_rubric="The answer helps.",
        tags=frozenset(
            {
                "channel.web-chat",
                "task.customer-support",
                "environment.consumer",
                "revision.v1",
            }
        ),
    )


def loaded_campaign(tmp_path: Path, *, max_requests: int = 100) -> LoadedCampaign:
    suite = SuiteSpec(version=1, name="wide", scenarios=(scenario(),))
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite.model_dump(mode="json")), encoding="utf-8")
    target = PydanticAITargetSpec(
        version=1,
        name="glm",
        kind="pydantic_ai",
        instructions="Give a useful answer.",
    )
    target_path = tmp_path / "target.yaml"
    target_path.write_text(yaml.safe_dump(target.model_dump(mode="json")), encoding="utf-8")
    spec = CampaignSpec(
        version=1,
        name="wide",
        suite=suite_path,
        bounds=CampaignBounds(
            max_cases=10,
            max_concurrency=2,
            run_seconds=30,
            max_model_requests=max_requests,
            max_output_tokens=100_000,
        ),
        lanes=(
            CampaignLaneSpec(
                name="direct",
                revision="v1",
                execution_environment="in-process",
                target=target_path,
                interaction="simulated",
                required_env=("ZAI_API_KEY",),
                target_max_output_tokens=512,
            ),
            CampaignLaneSpec(
                name="remote",
                revision="v1",
                execution_environment="agentenv",
                target=target_path,
                interaction="real",
                required_env=("E2B_API_KEY",),
            ),
        ),
    )
    return LoadedCampaign(spec=spec, source_directory=tmp_path)


def test_campaign_compiles_typed_dimensions_and_explicit_exclusions(tmp_path: Path) -> None:
    plan = compile_campaign(loaded_campaign(tmp_path), environment={"ZAI_API_KEY": "set"})

    assert len(plan.units) == 1
    assert plan.units[0].key.channel == "web-chat"
    assert plan.units[0].key.task == "customer-support"
    assert plan.units[0].key.scenario_environment == "consumer"
    assert plan.units[0].key.target_kind == "pydantic_ai"
    assert plan.units[0].key.execution_environment == "in-process"
    assert plan.exclusions[0].lane == "remote"
    assert "E2B_API_KEY" in plan.exclusions[0].reason


def test_campaign_rejects_work_above_its_request_budget(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="above max_model_requests=1"):
        compile_campaign(
            loaded_campaign(tmp_path, max_requests=1),
            environment={"ZAI_API_KEY": "set"},
        )


def test_campaign_refuses_to_plan_without_a_runnable_lane(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no runnable lanes"):
        compile_campaign(loaded_campaign(tmp_path), environment={})


def test_completed_campaign_units_are_not_run_twice(tmp_path: Path) -> None:
    plan = compile_campaign(loaded_campaign(tmp_path), environment={"ZAI_API_KEY": "set"})
    output = tmp_path / "output"
    save_new_plan(plan, output)
    calls: list[str] = []

    async def fake_executor(plan, unit, directory):
        calls.append(unit.key.scenario_id)
        return CampaignUnitResult(
            key=unit.key,
            passed=True,
            score=1,
            run_id="run-1",
            duration_seconds=0.01,
        )

    first = asyncio.run(execute_campaign(plan, output, executor=fake_executor))
    restored = load_saved_plan(output)
    second = asyncio.run(execute_campaign(restored, output, executor=fake_executor))

    assert first.completed == 1
    assert second.completed == 1
    assert calls == ["example"]


def test_started_unit_is_marked_interrupted_instead_of_replayed(tmp_path: Path) -> None:
    plan = compile_campaign(loaded_campaign(tmp_path), environment={"ZAI_API_KEY": "set"})
    output = tmp_path / "output"
    save_new_plan(plan, output)
    unit_directory = output / "cases" / "0001-direct-example"
    unit_directory.mkdir(parents=True)
    (unit_directory / "started.json").write_text(json.dumps({"started": True}), encoding="utf-8")
    calls = 0

    async def fake_executor(plan, unit, directory):
        nonlocal calls
        calls += 1
        raise AssertionError("interrupted work must not be replayed")

    summary = asyncio.run(execute_campaign(plan, output, executor=fake_executor))

    assert summary.interrupted == 1
    assert summary.completed == 0
    assert calls == 0
    assert (unit_directory / "interrupted.json").exists()


def test_started_unit_is_running_until_resume_classifies_it(tmp_path: Path) -> None:
    plan = compile_campaign(loaded_campaign(tmp_path), environment={"ZAI_API_KEY": "set"})
    output = tmp_path / "output"
    save_new_plan(plan, output)
    unit_directory = output / "cases" / "0001-direct-example"
    unit_directory.mkdir(parents=True)
    (unit_directory / "started.json").write_text(json.dumps({"started": True}), encoding="utf-8")

    summary = campaign.summarize_campaign(plan, output)

    assert summary.running == 1
    assert summary.interrupted == 0


def test_execution_failure_is_terminal_and_survives_resume(tmp_path: Path) -> None:
    plan = compile_campaign(loaded_campaign(tmp_path), environment={"ZAI_API_KEY": "set"})
    output = tmp_path / "output"
    save_new_plan(plan, output)
    calls = 0

    async def broken_executor(plan, unit, directory):
        nonlocal calls
        calls += 1
        raise RuntimeError("provider unavailable")

    first = asyncio.run(execute_campaign(plan, output, executor=broken_executor))
    second = asyncio.run(execute_campaign(plan, output, executor=broken_executor))

    assert first.execution_failed == 1
    assert second.execution_failed == 1
    assert calls == 1
    failure = json.loads((output / "cases" / "0001-direct-example" / "failure.json").read_text())
    assert failure["error_message"] == "provider unavailable"


def test_run_deadline_marks_active_work_interrupted(tmp_path: Path) -> None:
    plan = compile_campaign(loaded_campaign(tmp_path), environment={"ZAI_API_KEY": "set"})
    plan = plan.model_copy(update={"bounds": plan.bounds.model_copy(update={"run_seconds": 0.01})})
    output = tmp_path / "output"
    save_new_plan(plan, output)

    async def slow_executor(plan, unit, directory):
        await asyncio.sleep(1)
        raise AssertionError("deadline did not cancel the unit")

    summary = asyncio.run(execute_campaign(plan, output, executor=slow_executor))

    assert summary.interrupted == 1
    assert (output / "timeout.json").exists()


def test_default_executor_writes_case_artifacts(tmp_path: Path, monkeypatch: Any) -> None:
    plan = compile_campaign(loaded_campaign(tmp_path), environment={"ZAI_API_KEY": "set"})
    output = tmp_path / "output"
    save_new_plan(plan, output)
    monkeypatch.setenv("ZAI_API_KEY", "set")

    class FakeResult:
        report = SimpleNamespace(cases=[SimpleNamespace(output=SimpleNamespace(run_id="run-1"))])
        gate = SimpleNamespace(cases=[SimpleNamespace(passed=True, score=0.9)])

        def write_artifacts(self, directory: Path) -> None:
            (directory / "gate.json").write_text('{"passed":true}\n', encoding="utf-8")

    async def fake_evaluate(*args: object, **kwargs: object) -> FakeResult:
        return FakeResult()

    monkeypatch.setattr(campaign, "evaluate_suite", fake_evaluate)

    summary = asyncio.run(execute_campaign(plan, output))

    assert summary.passed is True
    assert (output / "cases" / "0001-direct-example" / "gate.json").exists()


def test_campaign_cli_loads_relative_paths_and_writes_a_plan(
    tmp_path: Path, monkeypatch: Any
) -> None:
    loaded = loaded_campaign(tmp_path)
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(
        yaml.safe_dump(loaded.spec.model_dump(mode="json")),
        encoding="utf-8",
    )
    output = tmp_path / "planned"
    monkeypatch.setenv("ZAI_API_KEY", "set")

    assert main(["plan", str(campaign_path), "--out", str(output)]) == 0
    assert load_saved_plan(output).units[0].key.channel == "web-chat"
    assert main(["plan", str(campaign_path), "--out", str(output)]) == 2


def test_campaign_requires_all_four_scenario_dimensions(tmp_path: Path) -> None:
    loaded = loaded_campaign(tmp_path)
    suite = load_suite(loaded.spec.suite)
    invalid = suite.model_copy(
        update={
            "scenarios": (
                suite.scenarios[0].model_copy(
                    update={"tags": frozenset({"channel.web-chat", "task.help"})}
                ),
            )
        }
    )
    Path(loaded.spec.suite).write_text(
        yaml.safe_dump(invalid.model_dump(mode="json")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one environment tag"):
        compile_campaign(loaded, environment={"ZAI_API_KEY": "set"})


def test_checked_in_wide_campaign_has_the_declared_coverage() -> None:
    plan = compile_campaign(
        campaign.load_campaign(ROOT / "campaigns" / "glm-wide.yaml"),
        environment={"ZAI_API_KEY": "configured"},
    )

    assert len(plan.units) == 60
    assert {unit.key.channel for unit in plan.units} == {
        "api",
        "cli",
        "email",
        "ticket",
        "web-chat",
    }
    assert {unit.key.target_kind for unit in plan.units} == {
        "command",
        "pydantic_ai",
    }
    assert {unit.key.execution_environment for unit in plan.units} == {
        "docker-container",
        "in-process",
        "local-process",
    }
    assert {item.execution_environment for item in plan.exclusions} == {
        "agentenv",
        "harbor",
    }
