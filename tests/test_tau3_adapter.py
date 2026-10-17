from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from pydantic_multiturn_evals import (
    Campaign,
    Case,
    Completed,
    ModelCallFact,
    ToolFact,
    Variant,
    run,
)
from pydantic_multiturn_evals.adapters import Tau3Participant, Tau3Task, Tau3TextHarness
from pydantic_multiturn_evals.model import TraceEvent


def test_tau3_adapter_uses_the_official_reward_and_imports_the_trajectory(tmp_path: Path) -> None:
    worker = tmp_path / "tau3_worker.py"
    worker.write_text(
        f"#!{sys.executable}\n"
        + """
import json, pathlib, sys
request = json.loads(sys.stdin.readline())
serialized = json.dumps(request)
assert 'private orchestration instruction' not in serialized
assert 'private-case-data' not in serialized
assert request['task'] == {
    'case': 'account-change',
    'domain': 'retail',
    'task': 'retail-task-1',
    'task_set': None,
    'split': 'base',
    'revision': 'fixture-1',
}
results = pathlib.Path(request['results_path'])
results.write_text(json.dumps({
    'info': {
        'git_commit': 'tau3-fixture-commit',
        'agent_info': {
            'implementation': 'llm_agent',
            'llm': 'agent-model',
            'llm_args': {'temperature': 0},
        },
        'user_info': {
            'implementation': 'user_simulator',
            'llm': 'user-model',
            'llm_args': {'temperature': 0.2},
        },
    },
    'simulations': [{
        'id': 'simulation-1',
        'task_id': 'retail-task-1',
        'duration': 1.25,
        'termination_reason': 'user_stop',
        'agent_cost': 0.003,
        'user_cost': 0.002,
        'reward_info': {
            'reward': 0,
            'reward_basis': ['DB', 'COMMUNICATE'],
            'reward_breakdown': {'DB': 0, 'COMMUNICATE': 1},
        },
        'messages': [
            {
                'role': 'assistant',
                'content': 'Hi! How can I help?',
                'cost': 0,
            },
            {
                'role': 'user',
                'content': 'Please update my account.',
                'cost': 0.001,
                'usage': {'prompt_tokens': 10, 'completion_tokens': 4},
                'generation_time_seconds': 0.1,
            },
            {
                'role': 'assistant',
                'tool_calls': [{
                    'id': 'call-1',
                    'name': 'update_account',
                    'arguments': {'status': 'active'},
                }],
                'cost': 0.002,
                'usage': {'prompt_tokens': 20, 'completion_tokens': 5},
                'generation_time_seconds': 0.2,
            },
            {
                'role': 'tool',
                'id': 'call-1',
                'content': '{"updated": false}',
            },
            {
                'role': 'assistant',
                'content': 'The account was not updated.',
                'cost': 0.001,
                'usage': {'prompt_tokens': 30, 'completion_tokens': 6},
                'generation_time_seconds': 0.3,
            },
            {
                'role': 'user',
                'content': '###STOP###',
                'cost': 0.001,
                'usage': {'prompt_tokens': 40, 'completion_tokens': 2},
                'generation_time_seconds': 0.1,
            },
        ],
    }],
}))
pathlib.Path(request['reply_path']).write_text(json.dumps({
    'protocol': 1,
    'package_version': '1.0.1',
    'simulations': 1,
}))
print('ordinary tau3 console output')
""",
        encoding="utf-8",
    )
    worker.chmod(0o755)
    harness = Tau3TextHarness(
        worker_argv=(str(worker),),
        worker_cwd=tmp_path,
        package_version="1.0.1",
        checkout_revision="tau3-fixture-commit",
        tasks=(
            Tau3Task(
                case="account-change",
                domain="retail",
                task="retail-task-1",
                revision="fixture-1",
            ),
        ),
        agent=Tau3Participant(
            implementation="llm_agent",
            model="agent-model",
            arguments={"temperature": 0},
        ),
        user=Tau3Participant(
            implementation="user_simulator",
            model="user-model",
            arguments={"temperature": 0.2},
        ),
    )
    campaign = Campaign(
        name="tau3-native-reward",
        revision="1",
        cases=(
            Case(
                name="account-change",
                task="private orchestration instruction",
                data={"secret": "private-case-data"},
                required_checks=("tau3.reward",),
            ),
        ),
        variants=(Variant(name="only", harness=harness),),
    )

    report = asyncio.run(run(campaign, directory=tmp_path / "run"))

    result = report.results[0]
    assert isinstance(result.terminal, Completed)
    assert result.passed is False
    assert result.terminal.evidence.checks[0].reason.startswith("Tau3 reward=0;")
    assert [message.role for message in result.terminal.evidence.public.messages] == [
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert [call.role for call in result.model_calls] == ["actor", "target", "target", "actor"]
    assert sum(call.usage.input_tokens for call in result.model_calls if call.usage) == 100
    events = [
        TraceEvent.model_validate_json(line)
        for line in next((tmp_path / "run").glob("trials/**/events.jsonl"))
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert sum(isinstance(event.fact, ModelCallFact) for event in events) == 4
    assert sum(isinstance(event.fact, ToolFact) for event in events) == 2
    official = next((tmp_path / "run").glob("trials/**/tau3/results.json"))
    assert (
        json.loads(official.read_text(encoding="utf-8"))["simulations"][0]["reward_info"]["reward"]
        == 0
    )


def test_tau3_participant_rejects_a_credential_in_persisted_arguments() -> None:
    with pytest.raises(ValueError, match="credentials must come from the environment: api_key"):
        Tau3Participant(
            implementation="llm_agent",
            model="provider/model",
            arguments={"api_key": "must-not-be-persisted"},
        )
