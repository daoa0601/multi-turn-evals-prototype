"""Public run() safety regressions for the Harbor adapter.

These tests pin the trust boundaries reviewed independently:

* agent arguments are frozen at construction, so caller mutation cannot
  change the manifest or the executed command, and credential-looking names
  are rejected recursively without echoing secret values;
* imported dialogue is operational trace data that reaches evaluators only
  when the harness opts in with public_trajectory;
* continuation chains are followed completely and rejected when unsafe;
* a native Harbor exception stays the primary failure even when the optional
  trajectory is damaged, without exposing tracebacks;
* artifact discovery stays inside the one job Harbor was asked to write.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from pydantic_multiturn_evals import (
    Campaign,
    Case,
    Completed,
    Failed,
    LogFact,
    MessageFact,
    ToolFact,
    Variant,
    run,
)
from pydantic_multiturn_evals.adapters import HarborHarness, HarborLocalTask
from pydantic_multiturn_evals.model import TraceEvent

_AGENT = {"name": "claude-code", "version": "2", "model_name": "agent-model"}


def _cli(tmp_path: Path, name: str, body: str) -> Path:
    executable = tmp_path / name
    executable.write_text(f"#!{sys.executable}\n" + body, encoding="utf-8")
    executable.chmod(0o755)
    return executable


def _campaign(fake_cli: Path, tmp_path: Path, **harness_arguments: Any) -> Campaign:
    harness = HarborHarness(
        executable=fake_cli,
        harbor_version="test-boundary",
        agent="claude-code",
        tasks=(HarborLocalTask(case="case", path=tmp_path, revision="fixture-1"),),
        **harness_arguments,
    )
    return Campaign(
        name="harbor-safety",
        revision="1",
        cases=(Case(name="case", task="Run the fixture", required_checks=("harbor.verified",)),),
        variants=(Variant(name="only", harness=harness),),
    )


def _execute(campaign: Campaign, directory: Path) -> Any:
    return asyncio.run(run(campaign, directory=directory))


def _attempt(run_directory: Path, suffix: str) -> Path:
    return next((run_directory / "trials").glob(f"*/**/attempt-*/{suffix}"))


def _receipt(run_directory: Path) -> dict[str, Any]:
    return json.loads(_attempt(run_directory, "result.json").read_text(encoding="utf-8"))


def _events(run_directory: Path) -> list[TraceEvent]:
    return [
        TraceEvent.model_validate_json(line)
        for line in _attempt(run_directory, "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _config(harness: HarborHarness) -> dict[str, Any]:
    config = harness.component.config
    assert isinstance(config, dict)
    return config


def _result_document(rewards: dict[str, float] | None = None) -> dict[str, Any]:
    return {
        "task_name": "case",
        "trial_name": "case__trial",
        "agent_info": {"name": "claude-code", "version": "2"},
        "agent_result": None,
        "verifier_result": {"rewards": rewards or {"reward": 1}},
        "exception_info": None,
    }


def _trajectory(steps: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": "ATIF-v1.6",
        "session_id": "session-safety",
        "agent": _AGENT,
        "steps": steps,
    }
    document.update(overrides)
    return document


def test_agent_arguments_snapshot_pins_config_and_command_against_mutation(
    tmp_path: Path,
) -> None:
    fake_cli = _cli(
        tmp_path,
        "harbor_capture_cli.py",
        f"""
import json, pathlib, sys
arguments = sys.argv[1:]
trial = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
trial = trial / arguments[arguments.index('--job-name') + 1] / 'case__trial'
trial.mkdir(parents=True)
kwargs = {{}}
for index, token in enumerate(arguments):
    if token == '--agent-kwarg':
        key, _, value = arguments[index + 1].partition('=')
        kwargs[key] = json.loads(value)
(trial / 'captured-kwargs.json').write_text(json.dumps(kwargs, sort_keys=True))
(trial / 'result.json').write_text(json.dumps({_result_document()!r}))
""",
    )
    arguments: dict[str, Any] = {"behavior": "good", "nested": {"retries": 3, "tags": ["a"]}}
    campaign = _campaign(fake_cli, tmp_path, agent_arguments=arguments)
    harness = campaign.variants[0].harness
    assert isinstance(harness, HarborHarness)

    arguments["behavior"] = "mutated-by-caller"
    arguments["nested"]["retries"] = 99
    harness.agent_arguments["behavior"] = "mutated-after-construction"

    frozen = {"behavior": "good", "nested": {"retries": 3, "tags": ["a"]}}
    assert _config(harness)["agent_arguments"] == frozen

    report = _execute(campaign, tmp_path / "run")

    assert isinstance(report.results[0].terminal, Completed)
    captured = _attempt(tmp_path / "run", "harbor/**/captured-kwargs.json")
    assert json.loads(captured.read_text(encoding="utf-8")) == frozen
    assert _config(harness)["agent_arguments"] == frozen


def test_agent_arguments_must_be_plain_json_data() -> None:
    with pytest.raises(ValueError, match="plain JSON data"):
        HarborHarness(
            executable=Path("harbor"),
            harbor_version="0.1.44",
            tasks=(HarborLocalTask(case="case", path=Path("task"), revision="1"),),
            agent_arguments=cast(dict[str, Any], {"handler": lambda: "not json"}),
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"client": {"api_key": "sk-live-supersecret"}},
        {"providers": [{" Access Token ": "sk-live-supersecret"}]},
        {"conf": {"client-secret": "sk-live-supersecret"}},
        {"layer": {"deep": [{"AUTHORIZATION": "Bearer sk-live-supersecret"}]}},
        {" Api Key ": "sk-live-supersecret"},
        {"openai_api_key": "sk-live-supersecret"},
        {"auth": {"tokens": [{"name": "access_token", "value": "sk-live-supersecret"}]}},
    ],
    ids=[
        "nested-api-key",
        "whitespace-access-token",
        "hyphen-client-secret",
        "deep-authorization",
        "top-level-whitespace",
        "suffix-api-key",
        "named-token-in-list",
    ],
)
def test_nested_credential_names_are_rejected_without_leaking_values(
    arguments: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="must come from the environment") as failure:
        HarborHarness(
            executable=Path("harbor"),
            harbor_version="0.1.44",
            tasks=(HarborLocalTask(case="case", path=Path("task"), revision="1"),),
            agent_arguments=arguments,
        )
    assert "sk-live-supersecret" not in str(failure.value)


SIMPLE_DIALOGUE = _trajectory(
    [
        {"step_id": 1, "source": "user", "message": "Please open port 443."},
        {
            "step_id": 2,
            "source": "agent",
            "model_name": "agent-model",
            "message": "Checking the firewall.",
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
            "metrics": {"prompt_tokens": 100, "completion_tokens": 10, "cost_usd": 0.01},
        },
    ]
)


def test_trajectory_dialogue_is_private_by_default_and_public_on_opt_in(
    tmp_path: Path,
) -> None:
    fake_cli = _cli(
        tmp_path,
        "harbor_dialogue_cli.py",
        f"""
import json, pathlib, sys
arguments = sys.argv[1:]
trial = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
trial = trial / arguments[arguments.index('--job-name') + 1] / 'case__trial'
(trial / 'agent').mkdir(parents=True)
(trial / 'agent' / 'trajectory.json').write_text({json.dumps(SIMPLE_DIALOGUE)!r})
(trial / 'result.json').write_text(json.dumps({_result_document()!r}))
""",
    )

    private = _campaign(fake_cli, tmp_path)
    harness = private.variants[0].harness
    assert isinstance(harness, HarborHarness)
    assert _config(harness)["public_trajectory"] is False

    report = _execute(private, tmp_path / "private-run")

    terminal = report.results[0].terminal
    assert isinstance(terminal, Completed)
    assert terminal.evidence.public.messages == ()
    receipt = json.dumps(_receipt(tmp_path / "private-run"))
    assert "Please open port 443." not in receipt
    assert '{"blocked": false}' not in receipt
    events = _events(tmp_path / "private-run")
    assert any(
        isinstance(event.fact, MessageFact)
        and event.fact.message.content == "Please open port 443."
        for event in events
    )
    assert any(
        isinstance(event.fact, ToolFact) and event.fact.output == '{"blocked": false}'
        for event in events
    )

    public = _campaign(fake_cli, tmp_path, public_trajectory=True)
    public_harness = public.variants[0].harness
    assert isinstance(public_harness, HarborHarness)
    assert _config(public_harness)["public_trajectory"] is True

    opt_in = _execute(public, tmp_path / "public-run")

    terminal = opt_in.results[0].terminal
    assert isinstance(terminal, Completed)
    assert [message.content for message in terminal.evidence.public.messages] == [
        "Please open port 443.",
        "Checking the firewall.",
        '{"blocked": false}',
    ]


FIRST_SEGMENT = _trajectory(
    [
        {"step_id": 1, "source": "user", "message": "Read the seed."},
        {
            "step_id": 2,
            "source": "agent",
            "model_name": "agent-model",
            "message": "Working on it.",
            "metrics": {"prompt_tokens": 100, "completion_tokens": 10, "cost_usd": 0.01},
        },
    ],
    continued_trajectory_ref="trajectory.cont-1.json",
)
CONTINUATION_SEGMENT = _trajectory(
    [
        {
            "step_id": 1,
            "source": "user",
            "message": "Read the seed.",
            "is_copied_context": True,
        },
        {
            "step_id": 2,
            "source": "agent",
            "message": "Working on it.",
            "metrics": {"prompt_tokens": 70, "completion_tokens": 7, "cost_usd": 0.007},
            "is_copied_context": True,
        },
        {"step_id": 3, "source": "user", "message": "Here is the handoff summary."},
        {
            "step_id": 4,
            "source": "agent",
            "model_name": "agent-model",
            "message": "FINAL: the port is open.",
            "metrics": {"prompt_tokens": 40, "completion_tokens": 6, "cost_usd": 0.02},
        },
    ]
)


def test_continuation_chain_imports_the_final_segment_and_usage_once(tmp_path: Path) -> None:
    fake_cli = _cli(
        tmp_path,
        "harbor_chain_cli.py",
        f"""
import json, pathlib, sys
arguments = sys.argv[1:]
trial = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
trial = trial / arguments[arguments.index('--job-name') + 1] / 'case__trial'
agent = trial / 'agent'
agent.mkdir(parents=True)
(agent / 'trajectory.json').write_text({json.dumps(FIRST_SEGMENT)!r})
(agent / 'trajectory.cont-1.json').write_text({json.dumps(CONTINUATION_SEGMENT)!r})
(trial / 'result.json').write_text(json.dumps({_result_document()!r}))
""",
    )

    report = _execute(_campaign(fake_cli, tmp_path, public_trajectory=True), tmp_path / "run")

    result = report.results[0]
    terminal = result.terminal
    assert isinstance(terminal, Completed)
    assert [message.content for message in terminal.evidence.public.messages] == [
        "Read the seed.",
        "Working on it.",
        "Here is the handoff summary.",
        "FINAL: the port is open.",
    ]
    receipts = [
        call.requested.config["receipt"] for call in result.model_calls if call.role == "target"
    ]
    assert [receipt.rsplit("/", 1)[-1] for receipt in receipts] == [
        "trajectory.json",
        "trajectory.cont-1.json",
    ]
    assert len({call.operation for call in result.model_calls}) == 2
    usage = [call.usage for call in result.model_calls if call.usage is not None]
    assert sum(item.input_tokens for item in usage) == 140
    assert sum(item.output_tokens for item in usage) == 16
    assert sum(item.cost_usd or 0 for item in usage) == pytest.approx(0.03)
    events = _events(tmp_path / "run")
    assert sum(isinstance(event.fact, MessageFact) for event in events) == 4
    artifacts = [artifact.path for artifact in terminal.evidence.artifacts]
    assert any(path.endswith("agent/trajectory.json") for path in artifacts)
    assert any(path.endswith("agent/trajectory.cont-1.json") for path in artifacts)


@pytest.mark.parametrize(
    "unsafe",
    [
        "missing-continuation",
        "cyclic-chain",
        "escaping-reference",
        "absolute-reference",
        "symlinked-continuation",
        "byte-limit",
        "segment-limit",
    ],
)
def test_unsafe_continuation_chains_are_rejected(tmp_path: Path, unsafe: str) -> None:
    first = json.dumps(FIRST_SEGMENT)
    second = json.dumps(CONTINUATION_SEGMENT)
    canary = json.dumps(
        _trajectory([{"step_id": 1, "source": "user", "message": "must-not-be-imported"}])
    )
    byte_budget = len(first) + max(len(second) // 2, 1)
    fake_cli = _cli(
        tmp_path,
        "harbor_unsafe_chain_cli.py",
        f"""
import json, pathlib, sys
arguments = sys.argv[1:]
trial = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
trial = trial / arguments[arguments.index('--job-name') + 1] / 'case__trial'
agent = trial / 'agent'
agent.mkdir(parents=True)
unsafe = {unsafe!r}
if unsafe == 'segment-limit':
    for index in range(32):
        document = {{
            'schema_version': 'ATIF-v1.6', 'session_id': 'session-limit',
            'agent': {{'name': 'claude-code', 'version': '2'}},
            'steps': [{{'step_id': 1, 'source': 'user', 'message': 'step'}}],
            'continued_trajectory_ref': f'trajectory.cont-{{index + 1}}.json',
        }}
        name = 'trajectory.json' if index == 0 else f'trajectory.cont-{{index}}.json'
        (agent / name).write_text(json.dumps(document))
elif unsafe == 'cyclic-chain':
    (agent / 'trajectory.json').write_text(json.dumps({{
        'schema_version': 'ATIF-v1.6', 'session_id': 'session-cycle',
        'agent': {{'name': 'claude-code', 'version': '2'}},
        'steps': [{{'step_id': 1, 'source': 'user', 'message': 'first'}}],
        'continued_trajectory_ref': 'trajectory.cont-1.json',
    }}))
    (agent / 'trajectory.cont-1.json').write_text(json.dumps({{
        'schema_version': 'ATIF-v1.6', 'session_id': 'session-cycle',
        'agent': {{'name': 'claude-code', 'version': '2'}},
        'steps': [{{'step_id': 1, 'source': 'user', 'message': 'second'}}],
        'continued_trajectory_ref': 'trajectory.json',
    }}))
else:
    reference = {{
        'escaping-reference': '../escape.json',
        'absolute-reference': '/etc/passwd',
    }}.get(unsafe, 'trajectory.cont-1.json')
    document = json.loads({first!r})
    document['continued_trajectory_ref'] = reference
    (agent / 'trajectory.json').write_text(json.dumps(document))
    if unsafe == 'escaping-reference':
        (trial / 'escape.json').write_text({canary!r})
    elif unsafe == 'symlinked-continuation':
        outside = pathlib.Path({str(tmp_path / "outside-continuation.json")!r})
        outside.write_text({canary!r})
        (agent / 'trajectory.cont-1.json').symlink_to(outside)
    elif unsafe == 'byte-limit':
        (agent / 'trajectory.cont-1.json').write_text({second!r})
(trial / 'result.json').write_text(json.dumps({_result_document()!r}))
""",
    )
    limits = {"trajectory_bytes": byte_budget} if unsafe == "byte-limit" else {}

    report = _execute(_campaign(fake_cli, tmp_path, **limits), tmp_path / "run")

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert terminal.failure.phase == "protocol"
    message = terminal.failure.message
    if unsafe == "missing-continuation":
        assert "trajectory.cont-1.json" in message
        assert "not a regular file" in message
    elif unsafe == "cyclic-chain":
        assert "cyclic" in message
    elif unsafe in ("escaping-reference", "absolute-reference"):
        assert "must stay inside the agent directory" in message
    elif unsafe == "symlinked-continuation":
        assert "symbolic link" in message
    elif unsafe == "byte-limit":
        assert "byte limit" in message
    else:
        assert "32 segments" in message
    assert terminal.evidence is not None
    if unsafe != "segment-limit":
        assert any(
            artifact.path.endswith("agent/trajectory.json")
            for artifact in terminal.evidence.artifacts
        )
    assert not any(
        isinstance(event.fact, MessageFact) and event.fact.message.content == "must-not-be-imported"
        for event in _events(tmp_path / "run")
    )
    assert "must-not-be-imported" not in json.dumps(_receipt(tmp_path / "run"))


def test_native_exception_stays_primary_over_a_malformed_trajectory(tmp_path: Path) -> None:
    fake_cli = _cli(
        tmp_path,
        "harbor_native_error_cli.py",
        f"""
import json, pathlib, sys
arguments = sys.argv[1:]
trial = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
trial = trial / arguments[arguments.index('--job-name') + 1] / 'case__trial'
(trial / 'agent').mkdir(parents=True)
(trial / 'agent' / 'trajectory.json').write_text('{{"schema_version": "ATIF-v1.6", "broken"')
result = {_result_document()!r}
result['exception_info'] = {{
    'exception_type': 'AgentTimeoutError',
    'exception_message': 'agent timed out after 30 seconds',
    'exception_traceback': 'Traceback SECRET-FRAME must-not-leak',
    'occurred_at': '2026-09-22T00:00:00',
}}
(trial / 'result.json').write_text(json.dumps(result))
""",
    )

    report = _execute(_campaign(fake_cli, tmp_path), tmp_path / "run")

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert terminal.failure.phase == "execute"
    assert "AgentTimeoutError" in terminal.failure.message
    assert "agent timed out after 30 seconds" in terminal.failure.message
    assert terminal.evidence is not None
    assert terminal.evidence.public.messages == ()
    assert any(
        artifact.path.endswith("agent/trajectory.json") for artifact in terminal.evidence.artifacts
    )
    warnings = [
        event.fact.message
        for event in _events(tmp_path / "run")
        if isinstance(event.fact, LogFact) and event.fact.level == "warning"
    ]
    assert any("trajectory import failed" in message for message in warnings)
    receipt = json.dumps(_receipt(tmp_path / "run"))
    assert "SECRET-FRAME" not in receipt
    assert "Traceback" not in receipt


def test_native_exception_message_is_bounded(tmp_path: Path) -> None:
    long_message = "x" * 5000 + "TAIL-MARKER"
    fake_cli = _cli(
        tmp_path,
        "harbor_long_error_cli.py",
        f"""
import json, pathlib, sys
arguments = sys.argv[1:]
trial = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
trial = trial / arguments[arguments.index('--job-name') + 1] / 'case__trial'
trial.mkdir(parents=True)
result = {_result_document()!r}
result['exception_info'] = {{
    'exception_type': 'VerifierCrashError',
    'exception_message': {long_message!r},
}}
(trial / 'result.json').write_text(json.dumps(result))
""",
    )

    report = _execute(_campaign(fake_cli, tmp_path), tmp_path / "run")

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert terminal.failure.phase == "execute"
    assert "VerifierCrashError" in terminal.failure.message
    assert "TAIL-MARKER" not in terminal.failure.message
    assert len(terminal.failure.message) < 1500


def test_artifact_discovery_is_limited_to_the_one_job(tmp_path: Path) -> None:
    decoy = json.dumps(
        _trajectory([{"step_id": 1, "source": "user", "message": "deep-decoy-must-not-appear"}])
    )
    fake_cli = _cli(
        tmp_path,
        "harbor_decoy_cli.py",
        f"""
import json, pathlib, sys
arguments = sys.argv[1:]
jobs = pathlib.Path(arguments[arguments.index('--jobs-dir') + 1])
job = jobs / arguments[arguments.index('--job-name') + 1]
trial = job / 'case__trial'
agent = trial / 'agent'
agent.mkdir(parents=True)
(agent / 'trajectory.json').write_text({json.dumps(SIMPLE_DIALOGUE)!r})
(trial / 'result.json').write_text(json.dumps({_result_document()!r}))
decoy_trajectory = {decoy!r}
deep = agent / 'nested' / 'deeper'
deep.mkdir(parents=True)
(deep / 'trajectory.json').write_text(decoy_trajectory)
decoy_job = jobs / 'attempt-9999' / 'decoy__trial'
decoy_job.mkdir(parents=True)
decoy_result = {_result_document()!r}
decoy_result['task_name'] = 'decoy-task'
(decoy_job / 'result.json').write_text(json.dumps(decoy_result))
""",
    )

    report = _execute(_campaign(fake_cli, tmp_path), tmp_path / "run")

    terminal = report.results[0].terminal
    assert isinstance(terminal, Completed)
    assert terminal.evidence.public.output == {"task_name": "case", "trial_name": "case__trial"}
    artifact_paths = [artifact.path for artifact in terminal.evidence.artifacts]
    assert all("nested" not in path for path in artifact_paths)
    assert all("attempt-9999" not in path for path in artifact_paths)
    receipt = json.dumps(_receipt(tmp_path / "run"))
    assert "deep-decoy-must-not-appear" not in receipt
    assert "decoy-task" not in receipt
