from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic_ai.models.test import TestModel

from pydantic_multiturn_evals.harbor_runner import (
    HarborArmSpec,
    HarborBackend,
    HarborCliBackend,
    HarborJobPlan,
    HarborJobReceipt,
    HarborLimits,
    HarborTrialReceipt,
    LoadedHarborArm,
    compare_harbor_suite,
    compile_harbor_tasks,
)
from pydantic_multiturn_evals.models import (
    AcceptDecision,
    ActorAccepted,
    ActorBrief,
    AssistantTurn,
    Exchange,
    GatePolicy,
    Scenario,
    ScenarioResult,
    SuiteSpec,
    Transcript,
    UserTurn,
)
from tests.helpers.model_binding import fake_model_binding


def suite() -> SuiteSpec:
    return SuiteSpec(
        version=1,
        name="harbor-suite",
        gate=GatePolicy(minimum_case_pass_rate=1, minimum_mean_score=0.7),
        scenarios=(
            Scenario(
                id="help",
                first_prompt="Help me.",
                actor=ActorBrief(persona="A user", goal="Get useful help"),
                judge_rubric="The reply helps.",
            ),
        ),
    )


def template(root: Path) -> Path:
    source = root / "template"
    (source / "environment").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (source / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
    return source


def arm(root: Path, name: str) -> LoadedHarborArm:
    base_config = root / f"{name}-job.yaml"
    base_config.write_text(
        yaml.safe_dump(
            {
                "agents": [{"name": "claude-code"}],
                "environment": {"type": "e2b"},
            }
        )
    )
    return LoadedHarborArm(
        spec=HarborArmSpec(
            version=1,
            name=name,
            kind="harbor",
            base_config=base_config,
            task_template=template(root / name),
            command=("harbor",),
            actor_command=("python", "-m", "pydantic_multiturn_evals.harbor_actor_worker"),
            inherit_env=(),
        ),
        source_directory=root,
    )


class FakeBackend(HarborBackend):
    def __init__(self, reward: float, calls: list[HarborJobPlan]) -> None:
        self.reward = reward
        self.calls = calls

    async def run_job(self, plan: HarborJobPlan) -> HarborJobReceipt:
        self.calls.append(plan)
        plan.job_dir.mkdir(parents=True)
        (plan.job_dir / "result.json").write_text(json.dumps({"id": f"job-{plan.role}"}))
        trials: list[HarborTrialReceipt] = []
        for case in plan.cases:
            trial_dir = plan.job_dir / f"trial-{case.task_name}"
            (trial_dir / "user-agent").mkdir(parents=True)
            result_path = trial_dir / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "id": f"trial-{plan.role}-{case.task_name}",
                        "task_name": case.task_name,
                        "trial_name": trial_dir.name,
                        "verifier_result": {"rewards": {"reward": self.reward}},
                    }
                )
            )
            decision = AcceptDecision(reason="Done.")
            conversation = ScenarioResult(
                run_id=f"run-{plan.role}-{case.task_name}",
                scenario_id=case.key.scenario_id,
                repeat_index=case.key.repeat_index,
                transcript=Transcript(
                    exchanges=(
                        Exchange(
                            user=UserTurn(content=case.scenario.first_prompt),
                            assistant=AssistantTurn(content=f"{plan.role} helpful reply"),
                        ),
                    )
                ),
                decisions=(decision,),
                termination=ActorAccepted(reason=decision.reason),
            )
            conversation_path = trial_dir / "user-agent" / "multiturn-result.json"
            conversation_path.write_text(conversation.model_dump_json())
            trials.append(
                HarborTrialReceipt(
                    result_path=result_path,
                    conversation_path=conversation_path,
                )
            )
        return HarborJobReceipt(
            job_id=f"job-{plan.role}",
            job_dir=plan.job_dir,
            trials=tuple(trials),
        )


def test_harbor_task_compiler_is_stable_and_removes_stale_tasks(tmp_path: Path) -> None:
    destination = tmp_path / "compiled"
    source = template(tmp_path)

    cases = compile_harbor_tasks(
        suite(),
        repeat=2,
        destination=destination,
        task_template=source,
        comparison_id="comparison-1",
        arm="baseline",
    )
    (destination / "stale").mkdir()
    rerun = compile_harbor_tasks(
        suite(),
        repeat=2,
        destination=destination,
        task_template=source,
        comparison_id="comparison-1",
        arm="baseline",
    )

    assert [case.task_name for case in cases] == [
        "pydantic-multiturn-evals/help--r0001",
        "pydantic-multiturn-evals/help--r0002",
    ]
    assert [case.task_name for case in rerun] == [
        "pydantic-multiturn-evals/help--r0001",
        "pydantic-multiturn-evals/help--r0002",
    ]
    assert not (destination / "stale").exists()
    instruction = (destination / "help--r0001" / "instruction.md").read_text()
    assert '"first_prompt":"Help me."' in instruction
    assert (destination / "help--r0001" / "environment" / "Dockerfile").exists()
    assert (destination / "help--r0001" / "tests" / "test.sh").exists()


def test_harbor_runs_two_jobs_and_combines_verifier_with_judge(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("SHOULD_NOT_BE_SERIALIZED", "secret-value")
    calls: list[HarborJobPlan] = []
    baseline = arm(tmp_path, "baseline")
    candidate = arm(tmp_path, "candidate")
    rewards = {"baseline": 0.0, "candidate": 1.0}

    result = asyncio.run(
        compare_harbor_suite(
            suite(),
            baseline=baseline,
            candidate=candidate,
            work_directory=tmp_path / "work",
            backend_factory=lambda loaded: FakeBackend(rewards[loaded.spec.name], calls),
            judge_binding_factory=lambda _suite: fake_model_binding(
                TestModel(custom_output_args={"reason": "Helpful.", "pass": True, "score": 0.9})
            ),
            repeat=2,
            max_concurrency=3,
            progress=False,
        )
    )
    result.write_artifacts(tmp_path / "output")

    assert [call.role for call in calls] == ["baseline", "candidate"]
    assert all(call.max_concurrency == 3 for call in calls)
    assert result.baseline.result.gate.passed is False
    assert result.candidate.result.gate.passed is True
    assert [pair.key.repeat_index for pair in result.pairs] == [1, 2]
    baseline_job = yaml.safe_load(calls[0].config_path.read_text())
    assert baseline_job["n_attempts"] == 1
    assert baseline_job["n_concurrent_trials"] == 3
    assert baseline_job["user_agent"]["import_path"].endswith("PydanticAdaptiveUserAgent")
    assert "secret-value" not in calls[0].config_path.read_text()
    evidence = json.loads(
        (tmp_path / "output" / "candidate" / "evidence.jsonl").read_text().splitlines()[0]
    )
    assert evidence["completion"]["environment"]["provider"] == "harbor"
    assert (tmp_path / "output" / "candidate" / "harbor" / "result.json").exists()


def test_cancelling_harbor_reaps_its_process_group(tmp_path: Path, monkeypatch: Any) -> None:
    pid_file = tmp_path / "harbor.pid"
    monkeypatch.setenv("HARBOR_TEST_PID_FILE", str(pid_file))
    sleeper = (
        "import os,time; from pathlib import Path; "
        "Path(os.environ['HARBOR_TEST_PID_FILE']).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    spec = HarborArmSpec(
        version=1,
        name="cancel-test",
        kind="harbor",
        base_config=tmp_path / "unused-job.yaml",
        task_template=tmp_path / "unused-template",
        command=(sys.executable, "-c", sleeper),
        actor_command=(sys.executable, "-c", "pass"),
        inherit_env=("HARBOR_TEST_PID_FILE",),
        limits=HarborLimits(job_seconds=60),
    )
    plan = HarborJobPlan(
        comparison_id="comparison",
        role="baseline",
        arm_name="cancel-test",
        config_path=tmp_path / "job.yaml",
        job_dir=tmp_path / "jobs" / "cancel-test",
        cases=(),
        max_concurrency=1,
    )

    async def exercise() -> bool:
        task = asyncio.create_task(HarborCliBackend(spec, tmp_path).run_job(plan))
        for _ in range(100):
            if pid_file.exists():
                break
            await asyncio.sleep(0.01)
        assert pid_file.exists()
        pid = int(pid_file.read_text(encoding="utf-8"))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        os.killpg(pid, signal.SIGKILL)
        return True

    assert asyncio.run(exercise()) is False
