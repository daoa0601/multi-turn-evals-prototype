from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from pydantic_multiturn_evals import (
    Campaign,
    Case,
    Completed,
    ComponentDefinition,
    Criterion,
    Failed,
    FunctionEvaluator,
    LogFact,
    MessageFact,
    ModelCallFact,
    ResourceFact,
    ToolFact,
    Variant,
    run,
)
from pydantic_multiturn_evals.adapters import (
    HarborDatasetTask,
    HarborHarness,
    HarborLocalTask,
)
from pydantic_multiturn_evals.model import TraceEvent


def _cli(tmp_path: Path, name: str, body: str) -> Path:
    executable = tmp_path / name
    executable.write_text(f"#!{sys.executable}\n" + body, encoding="utf-8")
    executable.chmod(0o755)
    return executable


def _receipt(run_directory: Path) -> dict[str, Any]:
    path = next((run_directory / "trials").glob("*/**/attempt-*/result.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def _events(run_directory: Path) -> list[TraceEvent]:
    path = next((run_directory / "trials").glob("*/**/attempt-*/events.jsonl"))
    return [
        TraceEvent.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_harbor_adapter_uses_the_saved_verifier_reward(tmp_path: Path) -> None:
    fake_cli = _cli(
        tmp_path,
        "harbor_reward_cli.py",
        """
import json, pathlib, sys
arguments = sys.argv[1:]
trial = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
trial = trial / arguments[arguments.index('--job-name') + 1] / 'case__trial'
trial.mkdir(parents=True)
(trial / 'verifier').mkdir()
(trial / 'verifier' / 'reward.txt').write_text('0\\n')
(trial / 'result.json').write_text(json.dumps({
  'task_name': 'case',
  'trial_name': 'case__trial',
  'agent_info': {'name': 'oracle', 'version': '1'},
  'agent_result': None,
  'verifier_result': {'rewards': {'reward': 0}},
  'exception_info': None,
}))
""",
    )
    task = HarborLocalTask(case="case", path=tmp_path, revision="fixture-1")
    campaign = Campaign(
        name="harbor-saved-result",
        revision="1",
        cases=(Case(name="case", task="Run", required_checks=("harbor.verified",)),),
        variants=(
            Variant(
                name="only",
                harness=HarborHarness(
                    executable=fake_cli,
                    harbor_version="test-boundary",
                    tasks=(task,),
                ),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    terminal = report.results[0].terminal
    assert isinstance(terminal, Completed)
    assert terminal.evidence.checks[0].passed is False
    saved = next((tmp_path / "run").glob("trials/**/harbor/**/result.json"))
    assert json.loads(saved.read_text(encoding="utf-8"))["verifier_result"]["rewards"] == {
        "reward": 0
    }


def test_harbor_dataset_binding_runs_one_task_without_prelisting_the_dataset(
    tmp_path: Path,
) -> None:
    fake_cli = _cli(
        tmp_path,
        "harbor_dataset_cli.py",
        """
import json, os, pathlib, sys
arguments = sys.argv[1:]
if arguments[arguments.index('--n-tasks') + 1] != '1' or '--task-name' in arguments:
    raise SystemExit('dataset canary must select exactly one task')
if os.environ.get('DOCKER_DEFAULT_PLATFORM') != 'linux/amd64':
    raise SystemExit('dataset canary lost its declared Docker platform')
trial = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
trial = trial / arguments[arguments.index('--job-name') + 1] / 'registry-task__trial'
trial.mkdir(parents=True)
(trial / 'result.json').write_text(json.dumps({
  'task_name': 'registry-task',
  'trial_name': 'registry-task__trial',
  'agent_info': {'name': 'oracle', 'version': '1'},
  'agent_result': None,
  'verifier_result': {'rewards': {'reward': 1}},
  'exception_info': None,
}))
""",
    )
    campaign = Campaign(
        name="harbor-dataset-canary",
        revision="1",
        cases=(
            Case(name="tool-use", task="Run one BFCL task", required_checks=("harbor.verified",)),
        ),
        variants=(
            Variant(
                name="oracle",
                harness=HarborHarness(
                    executable=fake_cli,
                    harbor_version="test-boundary",
                    docker_platform="linux/amd64",
                    tasks=(
                        HarborDatasetTask(
                            case="tool-use",
                            dataset="bfcl@1.0",
                            revision="1.0",
                        ),
                    ),
                ),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    terminal = report.results[0].terminal
    assert isinstance(terminal, Completed)
    output = terminal.evidence.public.output
    assert isinstance(output, dict)
    assert output["task_name"] == "registry-task"
    assert terminal.evidence.checks[0].passed is True


RICH_TRAJECTORY = {
    "schema_version": "ATIF-v1.6",
    "session_id": "session-1",
    "agent": {"name": "claude-code", "version": "2", "model_name": "agent-model"},
    "steps": [
        {
            "step_id": 1,
            "source": "system",
            "message": "SYSTEM SECRET orchestration note",
        },
        {
            "step_id": 2,
            "source": "user",
            "message": "Please open port 443.",
        },
        {
            "step_id": 3,
            "source": "agent",
            "model_name": "agent-model",
            "message": "Checking the firewall.",
            "reasoning_content": "internal reasoning must stay hidden",
            "tool_calls": [
                {
                    "tool_call_id": "call-1",
                    "function_name": "read_firewall",
                    "arguments": {"host": "edge"},
                }
            ],
            "observation": {
                "results": [{"source_call_id": "call-1", "content": '{"blocked": false}'}]
            },
            "metrics": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "cached_tokens": 20,
                "cost_usd": 0.01,
            },
        },
        {
            "step_id": 4,
            "source": "agent",
            "message": "The port is free.",
            "metrics": {
                "prompt_tokens": 200,
                "completion_tokens": 8,
                "cost_usd": 0.02,
            },
        },
        {
            "step_id": 5,
            "source": "agent",
            "message": "Earlier context from a previous trajectory.",
            "metrics": {"prompt_tokens": 50, "completion_tokens": 5},
            "is_copied_context": True,
        },
    ],
}


def test_harbor_adapter_imports_public_dialogue_and_reported_usage(tmp_path: Path) -> None:
    fake_cli = _cli(
        tmp_path,
        "harbor_trajectory_cli.py",
        f"""
import json, pathlib, sys
arguments = sys.argv[1:]
trial = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
trial = trial / arguments[arguments.index('--job-name') + 1] / 'case__trial'
(trial / 'agent').mkdir(parents=True)
(trial / 'agent' / 'trajectory.json').write_text({json.dumps(RICH_TRAJECTORY)!r})
(trial / 'result.json').write_text(json.dumps({{
  'task_name': 'case',
  'trial_name': 'case__trial',
  'agent_info': {{'name': 'claude-code', 'version': '2',
                 'model_info': {{'name': 'agent-model', 'provider': 'anthropic'}}}},
  'agent_result': {{
      'n_input_tokens': 300, 'n_cache_tokens': 20, 'n_output_tokens': 18,
      'cost_usd': 0.03, 'metadata': {{'secret': 'agent-metadata-secret'}},
  }},
  'verifier_result': {{'rewards': {{'reward': 1, 'quality': 0.5}}}},
  'exception_info': None,
}}))
""",
    )
    harness = HarborHarness(
        executable=fake_cli,
        harbor_version="test-boundary",
        public_trajectory=True,
        tasks=(HarborLocalTask(case="case", path=tmp_path, revision="fixture-1"),),
        agent="claude-code",
    )
    campaign = Campaign(
        name="harbor-trajectory-import",
        revision="1",
        cases=(Case(name="case", task="Open the port", required_checks=("harbor.verified",)),),
        variants=(Variant(name="only", harness=harness),),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    result = report.results[0]
    terminal = result.terminal
    assert isinstance(terminal, Completed)
    messages = terminal.evidence.public.messages
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[2].content == '{"blocked": false}'
    output = terminal.evidence.public.output
    assert isinstance(output, dict)
    assert output["task_name"] == "case"
    assert "usage" not in output
    assert "agent_info" not in output
    assert terminal.evidence.checks[0].passed is True
    assert any(
        artifact.path.endswith("trajectory.json") for artifact in terminal.evidence.artifacts
    )
    assert [call.role for call in result.model_calls] == ["target", "target"]
    usage = [call.usage for call in result.model_calls if call.usage is not None]
    assert sum(item.input_tokens for item in usage) == 300
    assert sum(item.output_tokens for item in usage) == 18
    assert sum(item.cost_usd or 0 for item in usage) == pytest.approx(0.03)
    assert all(call.duration_seconds is None for call in result.model_calls)
    assert all(call.provider == "harbor" for call in result.model_calls)
    events = _events(tmp_path / "run")
    assert sum(isinstance(event.fact, MessageFact) for event in events) == 4
    assert sum(isinstance(event.fact, ToolFact) for event in events) == 2
    assert sum(isinstance(event.fact, ModelCallFact) for event in events) == 2
    reward_facts = [
        event.fact.name
        for event in events
        if isinstance(event.fact, ResourceFact) and event.fact.name.startswith("harbor-reward-")
    ]
    assert sorted(reward_facts) == ["harbor-reward-quality", "harbor-reward-reward"]
    receipt = _receipt(tmp_path / "run")
    public = json.dumps(receipt["terminal"]["evidence"]["public"])
    assert "SYSTEM SECRET" not in public
    assert "internal reasoning must stay hidden" not in public
    assert "agent-metadata-secret" not in public
    assert "quality" not in public
    assert '"agent_result"' not in public


def test_harbor_adapter_keeps_partial_dialogue_when_the_agent_fails(tmp_path: Path) -> None:
    trajectory = {
        "schema_version": "ATIF-v1.6",
        "session_id": "session-2",
        "agent": {"name": "claude-code", "version": "2", "model_name": "agent-model"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "Please open port 443."},
            {"step_id": 2, "source": "agent", "message": "Checking the firewall."},
        ],
    }
    fake_cli = _cli(
        tmp_path,
        "harbor_failed_agent_cli.py",
        f"""
import json, pathlib, sys
arguments = sys.argv[1:]
trial = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
trial = trial / arguments[arguments.index('--job-name') + 1] / 'case__trial'
(trial / 'agent').mkdir(parents=True)
(trial / 'agent' / 'trajectory.json').write_text({json.dumps(trajectory)!r})
(trial / 'result.json').write_text(json.dumps({{
  'task_name': 'case',
  'trial_name': 'case__trial',
  'agent_info': {{'name': 'claude-code', 'version': '2'}},
  'agent_result': None,
  'verifier_result': {{'rewards': {{'reward': 1}}}},
  'exception_info': {{
      'exception_type': 'AgentTimeoutError',
      'exception_message': 'agent timed out',
      'exception_traceback': 'traceback',
      'occurred_at': '2026-09-22T00:00:00',
  }},
}}))
""",
    )
    campaign = Campaign(
        name="harbor-native-failure",
        revision="1",
        cases=(Case(name="case", task="Open the port", required_checks=("harbor.verified",)),),
        variants=(
            Variant(
                name="only",
                harness=HarborHarness(
                    executable=fake_cli,
                    harbor_version="test-boundary",
                    public_trajectory=True,
                    agent="claude-code",
                    tasks=(HarborLocalTask(case="case", path=tmp_path, revision="fixture-1"),),
                ),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert terminal.failure.phase == "execute"
    assert "AgentTimeoutError" in terminal.failure.message
    assert "agent timed out" in terminal.failure.message
    assert terminal.evidence is not None
    assert [message.role for message in terminal.evidence.public.messages] == [
        "user",
        "assistant",
    ]
    assert any(
        artifact.path.endswith("trajectory.json") for artifact in terminal.evidence.artifacts
    )
    events = _events(tmp_path / "run")
    assert sum(isinstance(event.fact, MessageFact) for event in events) == 2


def test_harbor_adapter_reports_evaluator_failures_separately_from_native_ones(
    tmp_path: Path,
) -> None:
    trajectory = {
        "schema_version": "ATIF-v1.6",
        "session_id": "session-3",
        "agent": {"name": "claude-code", "version": "2", "model_name": "agent-model"},
        "steps": [{"step_id": 1, "source": "user", "message": "Please open port 443."}],
    }
    fake_cli = _cli(
        tmp_path,
        "harbor_judge_cli.py",
        f"""
import json, pathlib, sys
arguments = sys.argv[1:]
trial = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
trial = trial / arguments[arguments.index('--job-name') + 1] / 'case__trial'
(trial / 'agent').mkdir(parents=True)
(trial / 'agent' / 'trajectory.json').write_text({json.dumps(trajectory)!r})
(trial / 'result.json').write_text(json.dumps({{
  'task_name': 'case',
  'trial_name': 'case__trial',
  'agent_info': {{'name': 'claude-code', 'version': '2'}},
  'agent_result': None,
  'verifier_result': {{'rewards': {{'reward': 1}}}},
  'exception_info': None,
}}))
""",
    )

    async def judge(value: Any) -> tuple[()]:
        assert value.outcome.messages[0].content == "Please open port 443."
        raise RuntimeError("judge exploded")

    campaign = Campaign(
        name="harbor-evaluator-failure",
        revision="1",
        cases=(
            Case(
                name="case",
                task="Open the port",
                criteria=(Criterion(name="port-open", instruction="The port must be open."),),
            ),
        ),
        variants=(
            Variant(
                name="only",
                harness=HarborHarness(
                    executable=fake_cli,
                    harbor_version="test-boundary",
                    public_trajectory=True,
                    agent="claude-code",
                    tasks=(HarborLocalTask(case="case", path=tmp_path, revision="fixture-1"),),
                ),
                evaluators=(
                    FunctionEvaluator(
                        component=ComponentDefinition(kind="raising-judge", version="1"),
                        function=judge,
                    ),
                ),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert terminal.failure.phase == "evaluate"
    assert "judge exploded" in terminal.failure.message
    assert terminal.evidence is not None
    assert terminal.evidence.public.messages[0].content == "Please open port 443."


def test_harbor_adapter_rejects_the_wrong_dataset_task(tmp_path: Path) -> None:
    fake_cli = _cli(
        tmp_path,
        "harbor_wrong_task_cli.py",
        """
import json, pathlib, sys
arguments = sys.argv[1:]
trial = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
trial = trial / arguments[arguments.index('--job-name') + 1] / 'other-task__trial'
trial.mkdir(parents=True)
(trial / 'result.json').write_text(json.dumps({
  'task_name': 'other-task',
  'trial_name': 'other-task__trial',
  'agent_info': {'name': 'oracle', 'version': '1'},
  'agent_result': None,
  'verifier_result': {'rewards': {'reward': 1}},
  'exception_info': None,
}))
""",
    )
    campaign = Campaign(
        name="harbor-wrong-task",
        revision="1",
        cases=(Case(name="case", task="Run", required_checks=("harbor.verified",)),),
        variants=(
            Variant(
                name="only",
                harness=HarborHarness(
                    executable=fake_cli,
                    harbor_version="test-boundary",
                    tasks=(
                        HarborDatasetTask(
                            case="case",
                            dataset="bfcl@1.0",
                            task="wanted-task",
                            revision="1.0",
                        ),
                    ),
                ),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert terminal.failure.phase == "protocol"
    assert "Harbor returned task 'other-task', expected 'wanted-task'" in terminal.failure.message


@pytest.mark.parametrize(
    "trajectory_text",
    [
        '{"schema_version": "ATIF-v1.6", "broken"',
        json.dumps(
            {
                "schema_version": "ATIF-v2.0",
                "session_id": "session-4",
                "agent": {"name": "claude-code", "version": "2"},
                "steps": [{"step_id": 1, "source": "user", "message": "hello"}],
            }
        ),
    ],
    ids=["unparsable-json", "unsupported-schema-version"],
)
def test_harbor_adapter_rejects_a_malformed_trajectory(
    tmp_path: Path, trajectory_text: str
) -> None:
    fake_cli = _cli(
        tmp_path,
        "harbor_bad_trajectory_cli.py",
        f"""
import json, pathlib, sys
arguments = sys.argv[1:]
trial = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
trial = trial / arguments[arguments.index('--job-name') + 1] / 'case__trial'
(trial / 'agent').mkdir(parents=True)
(trial / 'agent' / 'trajectory.json').write_text({trajectory_text!r})
(trial / 'result.json').write_text(json.dumps({{
  'task_name': 'case',
  'trial_name': 'case__trial',
  'agent_info': {{'name': 'oracle', 'version': '1'}},
  'agent_result': None,
  'verifier_result': {{'rewards': {{'reward': 1}}}},
  'exception_info': None,
}}))
""",
    )
    campaign = Campaign(
        name="harbor-malformed-trajectory",
        revision="1",
        cases=(Case(name="case", task="Run", required_checks=("harbor.verified",)),),
        variants=(
            Variant(
                name="only",
                harness=HarborHarness(
                    executable=fake_cli,
                    harbor_version="test-boundary",
                    tasks=(HarborLocalTask(case="case", path=tmp_path, revision="fixture-1"),),
                ),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert terminal.failure.phase == "protocol"
    assert "invalid Harbor trajectory" in terminal.failure.message
    assert terminal.evidence is not None
    assert any(
        artifact.path.endswith("trajectory.json") for artifact in terminal.evidence.artifacts
    )


@pytest.mark.parametrize("reward_value", [True, float("nan")], ids=["boolean", "nonfinite"])
def test_harbor_adapter_rejects_malformed_native_rewards(
    tmp_path: Path, reward_value: bool | float
) -> None:
    rewards_json = json.dumps({"reward": reward_value})
    fake_cli = _cli(
        tmp_path,
        "harbor_bad_reward_cli.py",
        f"""
import json, pathlib, sys
arguments = sys.argv[1:]
trial = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
trial = trial / arguments[arguments.index('--job-name') + 1] / 'case__trial'
trial.mkdir(parents=True)
(trial / 'result.json').write_text(json.dumps({{
  'task_name': 'case',
  'trial_name': 'case__trial',
  'agent_info': {{'name': 'oracle', 'version': '1'}},
  'agent_result': None,
  'verifier_result': {{'rewards': json.loads({rewards_json!r})}},
  'exception_info': None,
}}))
""",
    )
    campaign = Campaign(
        name="harbor-malformed-reward",
        revision="1",
        cases=(Case(name="case", task="Run", required_checks=("harbor.verified",)),),
        variants=(
            Variant(
                name="only",
                harness=HarborHarness(
                    executable=fake_cli,
                    harbor_version="test-boundary",
                    tasks=(HarborLocalTask(case="case", path=tmp_path, revision="fixture-1"),),
                ),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert terminal.failure.phase == "verify"
    assert "no finite numeric reward 'reward'" in terminal.failure.message


def test_harbor_adapter_preserves_provenance_when_harbor_exits_without_a_result(
    tmp_path: Path,
) -> None:
    trajectory = {
        "schema_version": "ATIF-v1.6",
        "session_id": "session-5",
        "agent": {"name": "claude-code", "version": "2", "model_name": "agent-model"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "Please open port 443."},
            {"step_id": 2, "source": "agent", "message": "Checking the firewall."},
        ],
    }
    fake_cli = _cli(
        tmp_path,
        "harbor_exploding_cli.py",
        f"""
import json, pathlib, sys
arguments = sys.argv[1:]
trial = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
trial = trial / arguments[arguments.index('--job-name') + 1] / 'case__trial'
(trial / 'agent').mkdir(parents=True)
(trial / 'agent' / 'trajectory.json').write_text({json.dumps(trajectory)!r})
print('harbor exploded before writing a result', file=sys.stderr)
raise SystemExit(1)
""",
    )
    campaign = Campaign(
        name="harbor-subprocess-failure",
        revision="1",
        cases=(Case(name="case", task="Run", required_checks=("harbor.verified",)),),
        variants=(
            Variant(
                name="only",
                harness=HarborHarness(
                    executable=fake_cli,
                    harbor_version="test-boundary",
                    public_trajectory=True,
                    agent="claude-code",
                    tasks=(HarborLocalTask(case="case", path=tmp_path, revision="fixture-1"),),
                ),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert terminal.failure.phase == "execute"
    assert "without a trial result" in terminal.failure.message
    assert "harbor exploded" not in terminal.failure.message
    assert "harbor.stderr.txt" in terminal.failure.message
    assert terminal.evidence is not None
    [stderr_artifact] = [a for a in terminal.evidence.artifacts if a.name.endswith("stderr.txt")]
    [stderr_file] = (tmp_path / "run").rglob(stderr_artifact.path)
    assert "harbor exploded" in stderr_file.read_text()
    assert [message.role for message in terminal.evidence.public.messages] == [
        "user",
        "assistant",
    ]
    events = _events(tmp_path / "run")
    assert sum(isinstance(event.fact, MessageFact) for event in events) == 2


def test_harbor_adapter_imports_text_parts_and_never_reads_image_files(
    tmp_path: Path,
) -> None:
    trajectory = {
        "schema_version": "ATIF-v1.6",
        "session_id": "session-6",
        "agent": {"name": "claude-code", "version": "2", "model_name": "agent-model"},
        "steps": [
            {
                "step_id": 1,
                "source": "user",
                "message": [
                    {"type": "text", "text": "Here is the screenshot request."},
                    {
                        "type": "image",
                        "source": {
                            "media_type": "image/png",
                            "path": "screenshots/missing-on-purpose.png",
                        },
                    },
                ],
            },
            {"step_id": 2, "source": "agent", "message": "I cannot see images, only text."},
        ],
    }
    fake_cli = _cli(
        tmp_path,
        "harbor_image_cli.py",
        f"""
import json, pathlib, sys
arguments = sys.argv[1:]
trial = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
trial = trial / arguments[arguments.index('--job-name') + 1] / 'case__trial'
(trial / 'agent').mkdir(parents=True)
(trial / 'agent' / 'trajectory.json').write_text({json.dumps(trajectory)!r})
(trial / 'result.json').write_text(json.dumps({{
  'task_name': 'case',
  'trial_name': 'case__trial',
  'agent_info': {{'name': 'claude-code', 'version': '2'}},
  'agent_result': None,
  'verifier_result': {{'rewards': {{'reward': 1}}}},
  'exception_info': None,
}}))
""",
    )
    campaign = Campaign(
        name="harbor-image-parts",
        revision="1",
        cases=(Case(name="case", task="Run", required_checks=("harbor.verified",)),),
        variants=(
            Variant(
                name="only",
                harness=HarborHarness(
                    executable=fake_cli,
                    harbor_version="test-boundary",
                    public_trajectory=True,
                    agent="claude-code",
                    tasks=(HarborLocalTask(case="case", path=tmp_path, revision="fixture-1"),),
                ),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    terminal = report.results[0].terminal
    assert isinstance(terminal, Completed)
    assert [message.content for message in terminal.evidence.public.messages] == [
        "Here is the screenshot request.",
        "I cannot see images, only text.",
    ]
    events = _events(tmp_path / "run")
    warnings = [
        event.fact.message
        for event in events
        if isinstance(event.fact, LogFact) and event.fact.level == "warning"
    ]
    assert any("1 image content parts" in message for message in warnings)


def test_harbor_adapter_runs_a_custom_agent_from_an_import_path(tmp_path: Path) -> None:
    fake_cli = _cli(
        tmp_path,
        "harbor_custom_agent_cli.py",
        """
import json, pathlib, sys
arguments = sys.argv[1:]
if '--agent' in arguments:
    raise SystemExit('custom agent must not pass a named --agent')
if arguments[arguments.index('--agent-import-path') + 1] != 'examples.custom_agent:CustomAgent':
    raise SystemExit('custom agent import path was not forwarded')
kwargs = {}
for index, token in enumerate(arguments):
    if token in ('--agent-kwarg', '--ak') and index + 1 < len(arguments):
        key, value = arguments[index + 1].split('=', 1)
        try:
            kwargs[key] = json.loads(value)
        except json.JSONDecodeError:
            kwargs[key] = value
if kwargs != {
    'prompt_template': 'be brief',
    'retries': 3,
    'enable': True,
    'tags': ['a'],
}:
    raise SystemExit(f'agent kwargs were not encoded natively: {kwargs}')
trial = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
trial = trial / arguments[arguments.index('--job-name') + 1] / 'case__trial'
trial.mkdir(parents=True)
(trial / 'result.json').write_text(json.dumps({
  'task_name': 'case',
  'trial_name': 'case__trial',
  'agent_info': {'name': 'custom-agent', 'version': '9'},
  'agent_result': None,
  'verifier_result': {'rewards': {'reward': 1}},
  'exception_info': None,
}))
""",
    )
    harness = HarborHarness(
        executable=fake_cli,
        harbor_version="test-boundary",
        public_trajectory=True,
        agent_import_path="examples.custom_agent:CustomAgent",
        agent_arguments={
            "prompt_template": "be brief",
            "retries": 3,
            "enable": True,
            "tags": ["a"],
        },
        tasks=(HarborLocalTask(case="case", path=tmp_path, revision="fixture-1"),),
    )
    campaign = Campaign(
        name="harbor-custom-agent",
        revision="1",
        cases=(Case(name="case", task="Run", required_checks=("harbor.verified",)),),
        variants=(Variant(name="only", harness=harness),),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    config = harness.component.config
    assert isinstance(config, dict)
    assert config["agent_import_path"] == "examples.custom_agent:CustomAgent"
    assert config["agent_arguments"] == {
        "prompt_template": "be brief",
        "retries": 3,
        "enable": True,
        "tags": ["a"],
    }
    terminal = report.results[0].terminal
    assert isinstance(terminal, Completed)
    assert terminal.evidence.checks[0].passed is True


@pytest.mark.parametrize(
    ("requested", "model_info", "matches"),
    [
        ("provider/requested", None, False),
        ("provider/requested", {"name": "different", "provider": "provider"}, False),
        ("provider/requested", {"name": "requested", "provider": "other"}, False),
        ("provider/requested", {"name": "requested", "provider": "provider"}, True),
        ("provider/requested", {"name": "provider/requested"}, True),
        ("requested", {"name": "requested", "provider": "provider"}, True),
    ],
    ids=["absent", "wrong-name", "wrong-provider", "split-name", "full-name", "bare-name"],
)
def test_harbor_requires_the_requested_model_in_the_saved_result(
    tmp_path: Path, requested: str, model_info: dict[str, str] | None, matches: bool
) -> None:
    fake_cli = _cli(
        tmp_path,
        "harbor_model_cli.py",
        f"""
import json, pathlib, sys
arguments = sys.argv[1:]
assert arguments[arguments.index('--model') + 1] == {requested!r}
trial = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
trial = trial / arguments[arguments.index('--job-name') + 1] / 'case__trial'
trial.mkdir(parents=True)
(trial / 'result.json').write_text(json.dumps({{
  'task_name': 'case',
  'trial_name': 'case__trial',
  'agent_info': {{'name': 'oracle', 'version': '1', 'model_info': {model_info!r}}},
  'verifier_result': {{'rewards': {{'reward': 1}}}},
}}))
""",
    )
    campaign = Campaign(
        name="harbor-model-identity",
        revision="1",
        cases=(Case(name="case", task="Run", required_checks=("harbor.verified",)),),
        variants=(
            Variant(
                name="only",
                harness=HarborHarness(
                    executable=fake_cli,
                    harbor_version="test-boundary",
                    tasks=(HarborLocalTask(case="case", path=tmp_path, revision="1"),),
                    model=requested,
                ),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    terminal = report.results[0].terminal
    if matches:
        assert isinstance(terminal, Completed)
        assert terminal.evidence.checks[0].passed is True
    else:
        assert isinstance(terminal, Failed)
        assert terminal.failure.phase == "protocol"
        assert "model" in terminal.failure.message
        assert requested in terminal.failure.message


def test_harbor_harness_rejects_inline_credentials_and_glob_task_bindings() -> None:
    with pytest.raises(ValueError, match="must come from the environment"):
        HarborHarness(
            executable=Path("/bin/true"),
            harbor_version="test-boundary",
            agent_arguments={"OPENAI_API_KEY": "must-not-be-persisted"},
            tasks=(HarborLocalTask(case="case", path=Path("."), revision="fixture-1"),),
        )
    with pytest.raises(ValueError, match="name one task exactly"):
        HarborHarness(
            executable=Path("/bin/true"),
            harbor_version="test-boundary",
            tasks=(
                HarborDatasetTask(
                    case="case",
                    dataset="bfcl@1.0",
                    task="simple-live-*",
                    revision="1.0",
                ),
            ),
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"llm_kwargs": {"api_key": "must-not-be-persisted"}},
        {"providers": [{"authorization": "must-not-be-persisted"}]},
    ],
)
def test_harbor_rejects_nested_credentials_before_writing_a_manifest(arguments: dict) -> None:
    with pytest.raises(ValueError, match="credentials"):
        HarborHarness(
            executable=Path("harbor"),
            harbor_version="0.1.44",
            tasks=(HarborLocalTask(case="case", path=Path("task"), revision="1"),),
            agent_arguments=arguments,
        )


@pytest.mark.parametrize("unsafe", ["large-result", "linked-result", "linked-trajectory-directory"])
def test_harbor_rejects_unsafe_native_files(tmp_path: Path, unsafe: str) -> None:
    fake_cli = _cli(
        tmp_path,
        "unsafe_native_cli.py",
        f"""
import json, pathlib, sys
arguments = sys.argv[1:]
trial = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
trial = trial / arguments[arguments.index('--job-name') + 1] / 'case__trial'
trial.mkdir(parents=True)
result = json.dumps({{'task_name': 'case', 'trial_name': 'case__trial',
    'agent_info': {{'name': 'oracle', 'version': '1'}},
    'verifier_result': {{'rewards': {{'reward': 1}}}}}})
if {unsafe!r} == 'large-result':
    (trial / 'result.json').write_text(result + ' ' * 4096)
elif {unsafe!r} == 'linked-result':
    external = pathlib.Path({str(tmp_path / "outside-result.json")!r})
    external.write_text(result)
    (trial / 'result.json').symlink_to(external)
else:
    (trial / 'result.json').write_text(result)
    alternate = trial / 'alternate-agent'
    alternate.mkdir()
    (alternate / 'trajectory.json').write_text(json.dumps({{
        'session_id': 'unsafe', 'agent': {{'name': 'oracle', 'version': '1'}},
        'steps': [{{'step_id': 1, 'source': 'agent', 'message': 'must-not-be-imported'}}],
    }}))
    (trial / 'agent').symlink_to(alternate, target_is_directory=True)
""",
    )
    campaign = Campaign(
        name="unsafe-native-file",
        revision="1",
        cases=(Case(name="case", task="Run", required_checks=("harbor.verified",)),),
        variants=(
            Variant(
                name="only",
                harness=HarborHarness(
                    executable=fake_cli,
                    harbor_version="test-boundary",
                    tasks=(HarborLocalTask(case="case", path=tmp_path, revision="1"),),
                    output_bytes=1024,
                ),
            ),
        ),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert "must-not-be-imported" not in json.dumps(_receipt(tmp_path / "run"))
    assert not any(isinstance(event.fact, MessageFact) for event in _events(tmp_path / "run"))
