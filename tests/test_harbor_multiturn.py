"""Local checks for the Harbor multi-turn integration fixtures.

These tests need no Docker and no real model provider. They exercise the
deterministic model fixture protocol, the judge policies, campaign
construction, and the agent module import inside Harbor's own Python.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from integration.harbor import harbor_multiturn_fixtures as fixtures
from pydantic_multiturn_evals import Criterion, EvaluationInput, Message, PublicOutcome
from pydantic_multiturn_evals.adapters import HarborHarness
from pydantic_multiturn_evals.providers import PydanticJudge
from scripts.verify_harbor_multiturn import (
    ACK_PHRASE,
    AGENT_IMPORT_PATH,
    CASE_NAME,
    CRITERION_ACK,
    CRITERION_READ,
    FIXTURES_DIR,
    ROOT,
    TASK_DIR,
    VerifyConfig,
    build_campaign,
    judge_binding,
)

HARBOR_EXECUTABLE = shutil.which("harbor")


def _criteria() -> tuple[Criterion, ...]:
    return (
        Criterion(
            name=CRITERION_READ,
            instruction=(
                "The transcript shows the agent reading environment state, storing a "
                "derived answer, and reading the stored value back."
            ),
        ),
        Criterion(
            name=CRITERION_ACK,
            instruction="The final agent message states the required confirmation phrase.",
        ),
    )


def _public_outcome(behavior: str = "good") -> PublicOutcome:
    """The guaranteed shape of imported evidence: observations plus agent text."""

    del behavior
    derived = "anchor-7f3c2b-fixturehost01"
    return PublicOutcome(
        messages=(
            Message(role="user", content="Read the seed and store the derived answer."),
            Message(
                role="tool",
                content="exit=0\nstdout:\nanchor-7f3c2b\nfixturehost01\nstderr:\n",
            ),
            Message(role="tool", content="exit=0\nstdout:\n\nstderr:\n"),
            Message(role="tool", content=f"exit=0\nstdout:\n{derived}\nstderr:\n"),
            Message(role="assistant", content=f"Stored and verified {derived}. {ACK_PHRASE}"),
        )
    )


def _evaluation_input() -> EvaluationInput:
    return EvaluationInput(
        task="Complete the Harbor stateful task.",
        criteria=_criteria(),
        outcome=_public_outcome(),
    )


def test_fixture_module_is_stdlib_only_and_importable() -> None:
    assert fixtures.BEHAVIORS == ("good", "wrong-target")
    assert fixtures.GOOD_TARGET == "/app/final/answer.txt"
    assert fixtures.WRONG_TARGET == "/app/state/answer.txt"
    imported = subprocess.run(
        [sys.executable, "-I", "-S", str(FIXTURES_DIR / "harbor_multiturn_fixtures.py")],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert imported.returncode == 0, imported.stderr


def test_fixture_reply_protocol_uses_observed_environment_state() -> None:
    messages = [{"role": "user", "content": "instruction"}]
    first = fixtures.fixture_reply("good", messages)
    command = json.loads(first["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])[
        "command"
    ]
    assert command == "cat /app/state/seed.txt && uname -n"

    observed = [{"role": "tool", "tool_call_id": "c1", "content": "anchor-7f3c2b\nhost42\n"}]
    second = fixtures.fixture_reply("good", messages + observed)
    write = json.loads(second["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])[
        "command"
    ]
    assert write == "printf '%s-%s\\n' anchor-7f3c2b host42 > /app/final/answer.txt"
    wrong = fixtures.fixture_reply("wrong-target", messages + observed)
    wrong_write = json.loads(
        wrong["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    )["command"]
    assert wrong_write.endswith("> /app/state/answer.txt")

    read_back = fixtures.fixture_reply(
        "good", messages + observed + [{"role": "tool", "content": ""}]
    )
    read_command = json.loads(
        read_back["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    )["command"]
    assert read_command == "cat /app/final/answer.txt"

    final = fixtures.fixture_reply(
        "good", messages + observed + [{"role": "tool", "content": ""}] * 2
    )
    assert final["choices"][0]["finish_reason"] == "stop"
    assert "anchor-7f3c2b-host42" in final["choices"][0]["message"]["content"]
    assert ACK_PHRASE in final["choices"][0]["message"]["content"]

    with pytest.raises(fixtures.FixtureProtocolError):
        fixtures.fixture_reply("good", messages + observed * 4)


def test_fixture_server_serves_deterministic_chat_completions() -> None:
    server = fixtures.FixtureModelServer(("127.0.0.1", 0), "good")
    thread = server.start()
    try:
        conversation = [{"role": "user", "content": "instruction"}]
        replies = []
        observations = [
            "exit=0\nstdout:\nanchor-7f3c2b\nhost99\nstderr:\n",
            "exit=0\nstdout:\n\nstderr:\n",
            "exit=0\nstdout:\nanchor-7f3c2b-host99\nstderr:\n",
        ]
        for observation in observations:
            response = httpx.post(
                f"{server.endpoint}/chat/completions",
                json={"model": "fixture/multiturn-deterministic", "messages": conversation},
                timeout=10,
            )
            assert response.status_code == 200
            reply = response.json()
            replies.append(reply)
            call = reply["choices"][0]["message"]["tool_calls"][0]
            conversation = conversation + [
                {"role": "assistant", "content": None, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "content": observation},
            ]
        final_response = httpx.post(
            f"{server.endpoint}/chat/completions",
            json={"model": "fixture/multiturn-deterministic", "messages": conversation},
            timeout=10,
        )
        final = final_response.json()
        assert final["choices"][0]["finish_reason"] == "stop"
        assert "anchor-7f3c2b-host99" in final["choices"][0]["message"]["content"]

        assert len(server.requests) == 4
        assert server.failures == []
        assert [reply["id"] for reply in replies + [final]] == [
            f"chatcmpl-fixture-good-{index:03d}" for index in range(1, 5)
        ]
        assert all(
            reply["usage"]["total_tokens"]
            == reply["usage"]["prompt_tokens"] + reply["usage"]["completion_tokens"]
            for reply in replies + [final]
        )
        missing = httpx.post(
            f"{server.endpoint}/v1/chat/completions",
            json={},
            timeout=10,
        )
        assert missing.status_code == 404
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_judge_policies_separate_native_and_evaluator_failures() -> None:
    calls: Counter[str] = Counter()
    standard = PydanticJudge(judge_binding("standard", calls))
    harsh = PydanticJudge(judge_binding("harsh", calls))
    value = _evaluation_input()

    async def judge(evaluator: PydanticJudge) -> dict[str, bool]:
        class _Trace:
            def record(self, fact: object) -> None:
                del fact

        judgments = await evaluator.evaluate(value, _Trace())  # type: ignore[arg-type]
        return {judgment.criterion: judgment.passed for judgment in judgments}

    standard_verdicts = asyncio.run(judge(standard))
    harsh_verdicts = asyncio.run(judge(harsh))
    assert standard_verdicts == {CRITERION_READ: True, CRITERION_ACK: True}
    assert harsh_verdicts == {CRITERION_READ: True, CRITERION_ACK: False}
    assert calls == {"judge-standard": 1, "judge-harsh": 1}


def test_judge_acknowledgement_uses_the_transcript_not_the_criteria() -> None:
    calls: Counter[str] = Counter()
    standard = PydanticJudge(judge_binding("standard", calls))
    value = EvaluationInput(
        task="Complete the Harbor stateful task.",
        criteria=_criteria(),
        outcome=PublicOutcome(
            messages=(
                Message(role="user", content="Read the seed and store the answer."),
                Message(role="assistant", content="Stored the answer without the phrase."),
            )
        ),
    )

    async def run_judge() -> dict[str, bool]:
        judgments = await standard.evaluate(value, _NullTrace())
        return {judgment.criterion: judgment.passed for judgment in judgments}

    verdicts = asyncio.run(run_judge())
    assert verdicts == {CRITERION_READ: False, CRITERION_ACK: False}


class _NullTrace:
    def record(self, fact: object) -> None:
        del fact


def test_task_fixture_declares_state_and_native_verifier() -> None:
    assert (TASK_DIR / "task.toml").is_file()
    assert (TASK_DIR / "instruction.md").is_file()
    assert (TASK_DIR / "environment" / "Dockerfile").is_file()
    assert (TASK_DIR / "tests" / "test.sh").is_file()
    dockerfile = (TASK_DIR / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert "anchor-7f3c2b" in dockerfile
    verifier = (TASK_DIR / "tests" / "test.sh").read_text(encoding="utf-8")
    assert "/app/final/answer.txt" in verifier
    assert "/logs/verifier/reward.txt" in verifier
    instruction = (TASK_DIR / "instruction.md").read_text(encoding="utf-8")
    assert ACK_PHRASE not in instruction
    assert "printf" not in instruction


@pytest.mark.skipif(HARBOR_EXECUTABLE is None, reason="Harbor CLI is not installed")
def test_agent_module_imports_and_runs_inside_harbor_python(tmp_path: Path) -> None:
    assert HARBOR_EXECUTABLE is not None
    executable = Path(HARBOR_EXECUTABLE).resolve()
    harbor_python = _harbor_python(executable)
    driver = tmp_path / "driver.py"
    driver.write_text(
        "\n".join(
            [
                "import asyncio, importlib.util, json, sys",
                "from pathlib import Path",
                "repo = Path(sys.argv[1])",
                "logs = Path(sys.argv[2])",
                "behavior = sys.argv[3]",
                "spec = importlib.util.spec_from_file_location("
                "'harbor_multiturn_fixtures',"
                " repo / 'integration' / 'harbor' / 'harbor_multiturn_fixtures.py')",
                "fixtures = importlib.util.module_from_spec(spec)",
                "spec.loader.exec_module(fixtures)",
                "server = fixtures.FixtureModelServer(('127.0.0.1', 0), behavior)",
                "thread = server.start()",
                "class Sandbox:",
                "    async def exec(self, command, cwd=None, env=None, timeout_sec=None):",
                "        import subprocess",
                "        from harbor.environments.base import ExecResult",
                "        rewritten = command.replace('/app/', str(logs) + '/')",
                "        proc = subprocess.run("
                "['bash', '-c', rewritten], capture_output=True, text=True)",
                "        return ExecResult(stdout=proc.stdout, stderr=proc.stderr,"
                " return_code=proc.returncode)",
                "async def main():",
                "    from harbor.models.agent.context import AgentContext",
                "    from harbor_multiturn_agent import MultiturnFixtureAgent",
                "    (logs / 'state').mkdir(parents=True, exist_ok=True)",
                "    (logs / 'final').mkdir(parents=True, exist_ok=True)",
                "    (logs / 'state' / 'seed.txt').write_text('anchor-7f3c2b\\n')",
                "    agent_logs = logs / 'agent'",
                "    agent_logs.mkdir(exist_ok=True)",
                "    agent = MultiturnFixtureAgent("
                "logs_dir=agent_logs, model_name='fixture/multiturn-deterministic',"
                " endpoint=server.endpoint, behavior=behavior)",
                "    context = AgentContext()",
                "    await agent.run('Read the seed and hostname.', Sandbox(), context)",
                "    trajectory = json.loads((agent_logs / 'trajectory.json').read_text())",
                "    print(json.dumps({",
                "        'behavior': behavior,",
                "        'requests': len(server.requests),",
                "        'steps': len(trajectory['steps']),",
                "        'final_metrics': trajectory['final_metrics'],",
                "        'answer_exists': (logs / 'final' / 'answer.txt').exists(),",
                "        'failures': server.failures,",
                "    }))",
                "asyncio.run(main())",
                "server.shutdown()",
                "thread.join()",
                "server.server_close()",
            ]
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(FIXTURES_DIR)
    for behavior, answer_exists in (("good", True), ("wrong-target", False)):
        workspace = tmp_path / behavior
        workspace.mkdir()
        result = subprocess.run(
            [str(harbor_python), str(driver), str(ROOT), str(workspace), behavior],
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        assert payload["behavior"] == behavior
        assert payload["requests"] == 4
        assert payload["steps"] == 5
        assert payload["answer_exists"] is answer_exists
        assert payload["failures"] == []
        assert payload["final_metrics"]["extra"]["model_requests"] == 4
        assert payload["final_metrics"]["extra"]["tool_calls"] == 3
        assert payload["final_metrics"]["total_prompt_tokens"] > 0


def _harbor_python(executable: Path) -> Path:
    if executable.suffix == ".exe":  # pragma: no cover - windows entrypoints
        raise RuntimeError("windows harbor entrypoints are not supported")
    first = executable.read_text(encoding="utf-8").splitlines()[0]
    if not first.startswith("#!"):
        raise RuntimeError(f"harbor executable {executable} has no python shebang")
    return Path(first[2:].strip())


def test_campaign_construction_binds_three_variants(tmp_path: Path) -> None:
    config = VerifyConfig(
        directory=tmp_path,
        executable=Path("/usr/bin/true"),
        samples=2,
        workers=4,
        docker_capacity=2,
    )
    calls: Counter[str] = Counter()
    campaign = build_campaign(config, "http://127.0.0.1:1/v1", "http://127.0.0.1:2/v1", calls)
    assert campaign.name == "harbor-multiturn-integration"
    assert [case.name for case in campaign.cases] == [CASE_NAME]
    assert [variant.name for variant in campaign.variants] == [
        "good",
        "wrong-target",
        "harsh-judge",
    ]
    for variant in campaign.variants:
        harness = variant.harness
        assert isinstance(harness, HarborHarness)
        component = cast(dict[str, Any], harness.component.config)
        assert component["agent_import_path"] == AGENT_IMPORT_PATH
        arguments = cast(dict[str, Any], component["agent_arguments"])
        if variant.name == "wrong-target":
            assert arguments["behavior"] == "wrong-target"
        else:
            assert arguments["behavior"] == "good"
        assert [claim.name for claim in variant.resources] == ["docker"]
        assert variant.evaluators, f"variant {variant.name} needs a judge"
    case = campaign.cases[0]
    assert case.required_checks == ("harbor.verified",)
    assert tuple(criterion.name for criterion in case.criteria) == (
        CRITERION_READ,
        CRITERION_ACK,
    )
    assert campaign.samples == 2


def test_verify_config_bounds_and_capacity_gating() -> None:
    import pytest as _pytest

    config = VerifyConfig(
        directory=Path("/tmp/x"),
        executable=Path("/usr/bin/true"),
    )
    assert (config.samples, config.workers, config.docker_capacity) == (2, 4, 2)
    assert config.trials == 6
    with _pytest.raises(ValueError):
        VerifyConfig(directory=Path("/tmp/x"), executable=Path("/usr/bin/true"), workers=2)
    with _pytest.raises(ValueError):
        VerifyConfig(
            directory=Path("/tmp/x"),
            executable=Path("/usr/bin/true"),
            samples=0,
            workers=4,
        )
    with _pytest.raises(ValueError):
        VerifyConfig(
            directory=Path("/tmp/x"),
            executable=Path("/usr/bin/true"),
            docker_capacity=99,
        )


FAKE_HARBOR_CLI = 'import asyncio\nimport json\nimport subprocess\nimport sys\nimport uuid\nfrom pathlib import Path\n\nimport httpx\n\n\ndef _argument_value(arguments, flag):\n    return arguments[arguments.index(flag) + 1]\n\n\ndef _agent_kwargs(arguments):\n    values = {}\n    for index, item in enumerate(arguments):\n        if item == "--agent-kwarg":\n            key, _, value = arguments[index + 1].partition("=")\n            values[key] = json.loads(value)\n    return values\n\n\nasync def main() -> int:\n    arguments = sys.argv[1:]\n    jobs = Path(_argument_value(arguments, "--jobs-dir"))\n    job_name = _argument_value(arguments, "--job-name")\n    task_dir = Path(_argument_value(arguments, "--path"))\n    behavior = _agent_kwargs(arguments)["behavior"]\n    endpoint = _agent_kwargs(arguments)["endpoint"]\n    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")\n    newline = chr(10)\n    sandbox = jobs / ("sandbox-" + uuid.uuid4().hex[:8]) / "app"\n    (sandbox / "state").mkdir(parents=True)\n    (sandbox / "final").mkdir(parents=True)\n    (sandbox / "state" / "seed.txt").write_text("anchor-7f3c2b" + newline)\n    host = uuid.uuid4().hex[:12]\n    messages = [{"role": "user", "content": instruction}]\n    steps = [{\n        "step_id": 1,\n        "timestamp": "2026-01-01T00:00:00+00:00",\n        "source": "user",\n        "message": instruction,\n    }]\n    prompt_tokens = 0\n    completion_tokens = 0\n    async with httpx.AsyncClient(timeout=30) as client:\n        for _turn in range(4):\n            response = await client.post(\n                endpoint + "/chat/completions",\n                json={"model": "fixture/multiturn-deterministic", "messages": messages},\n            )\n            response.raise_for_status()\n            reply = response.json()\n            message = reply["choices"][0]["message"]\n            usage = reply["usage"]\n            prompt_tokens += usage["prompt_tokens"]\n            completion_tokens += usage["completion_tokens"]\n            metrics = {\n                "prompt_tokens": usage["prompt_tokens"],\n                "completion_tokens": usage["completion_tokens"],\n                "cost_usd": 0.0,\n                "extra": {"model_fixture": "deterministic-local-openai-http-fixture"},\n            }\n            tool_calls = message.get("tool_calls") or []\n            if not tool_calls:\n                steps.append({\n                    "step_id": len(steps) + 1,\n                    "timestamp": "2026-01-01T00:00:01+00:00",\n                    "source": "agent",\n                    "model_name": "fixture/multiturn-deterministic",\n                    "message": message["content"],\n                    "metrics": metrics,\n                })\n                break\n            executed = []\n            for call in tool_calls:\n                command = json.loads(call["function"]["arguments"])["command"]\n                if "uname -n" in command:\n                    stdout = "anchor-7f3c2b" + newline + host + newline\n                    stderr, code = "", 0\n                else:\n                    proc = subprocess.run(\n                        ["bash", "-c", command.replace("/app/", str(sandbox) + "/")],\n                        capture_output=True,\n                        text=True,\n                    )\n                    stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode\n                observation = (\n                    "exit=" + str(code) + newline\n                    + "stdout:" + newline\n                    + (stdout or "") + newline\n                    + "stderr:" + newline\n                    + (stderr or "")\n                )\n                messages.append({\n                    "role": "assistant",\n                    "content": None,\n                    "tool_calls": [call],\n                })\n                messages.append({\n                    "role": "tool",\n                    "tool_call_id": call["id"],\n                    "content": observation,\n                })\n                executed.append((call, command, observation))\n            steps.append({\n                "step_id": len(steps) + 1,\n                "timestamp": "2026-01-01T00:00:01+00:00",\n                "source": "agent",\n                "model_name": "fixture/multiturn-deterministic",\n                "message": "",\n                "tool_calls": [\n                    {\n                        "tool_call_id": call["id"],\n                        "function_name": call["function"]["name"],\n                        "arguments": {"command": command},\n                    }\n                    for call, command, _ in executed\n                ],\n                "observation": {\n                    "results": [\n                        {"source_call_id": call["id"], "content": observation}\n                        for call, _, observation in executed\n                    ]\n                },\n                "metrics": metrics,\n            })\n    trajectory = {\n        "schema_version": "ATIF-v1.6",\n        "session_id": uuid.uuid4().hex,\n        "agent": {\n            "name": "multiturn-fixture-agent",\n            "version": "1.0.0",\n            "model_name": "fixture/multiturn-deterministic",\n            "extra": {\n                "behavior": behavior,\n                "model_fixture": "deterministic-local-openai-http-fixture",\n            },\n        },\n        "steps": steps,\n        "final_metrics": {\n            "total_prompt_tokens": prompt_tokens,\n            "total_completion_tokens": completion_tokens,\n            "total_cost_usd": 0.0,\n            "total_steps": len(steps),\n            "extra": {\n                "model_fixture": "deterministic-local-openai-http-fixture",\n                "model_requests": 4,\n                "tool_calls": 3,\n                "finished": True,\n            },\n        },\n    }\n    trial_dir = jobs / job_name / ("multiturn-task__" + uuid.uuid4().hex[:7])\n    (trial_dir / "agent").mkdir(parents=True)\n    (trial_dir / "agent" / "trajectory.json").write_text(json.dumps(trajectory, indent=2))\n    answer = sandbox / "final" / "answer.txt"\n    actual = answer.read_text().strip() if answer.is_file() else ""\n    reward = 1 if actual == "anchor-7f3c2b-" + host else 0\n    (trial_dir / "result.json").write_text(\n        json.dumps(\n            {\n                "task_name": "pydantic-multiturn-evals/multiturn-state",\n                "trial_name": trial_dir.name,\n                "agent_info": {\n                    "name": "multiturn-fixture-agent",\n                    "version": "1.0.0",\n                    "model_info": {\n                        "name": "multiturn-deterministic",\n                        "provider": "fixture",\n                    },\n                },\n                "agent_result": {\n                    "n_input_tokens": prompt_tokens,\n                    "n_output_tokens": completion_tokens,\n                    "cost_usd": 0.0,\n                },\n                "verifier_result": {"rewards": {"reward": reward}},\n                "exception_info": None,\n            }\n        )\n    )\n    return 0\n\n\nsys.exit(asyncio.run(main()))\n'  # noqa: E501


def test_verify_script_runs_end_to_end_with_a_fake_harbor_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive verify() itself with a fake Harbor CLI that speaks the real protocol.

    The fake CLI makes real fixture-model requests, executes the model-requested
    commands in a sandbox, writes an ATIF trajectory and a native reward, and
    mimics Harbor's result layout. This checks the verification script's
    assertions, comparisons, resume proof, and export isolation without Docker.
    """

    import scripts.verify_harbor_multiturn as verification

    fake_cli = tmp_path / "fake_harbor.py"
    fake_cli.write_text(f"#!{sys.executable}\n" + FAKE_HARBOR_CLI, encoding="utf-8")
    fake_cli.chmod(0o755)
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "if sys.argv[1] == 'ps':\n"
        "    print('unrelated-id unrelated-compose-project')\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    config = VerifyConfig(
        directory=tmp_path / "outcomes",
        executable=fake_cli,
        samples=2,
        workers=4,
        docker_capacity=2,
    )
    summary = verification.verify(config)

    assert summary["trials"] == 6
    assert summary["executed_trials"] == 6
    assert summary["resume_executed_trials"] == 0
    assert summary["model_fixture"]["requests_good"] == 16
    assert summary["model_fixture"]["requests_wrong_target"] == 8
    assert summary["model_fixture"]["distinct_derived_values"] == 6
    assert summary["model_fixture"]["failures"] == []
    assert summary["judge"]["calls"] == {"judge-harsh": 2, "judge-standard": 4}
    assert summary["exports"]["recorded"] == 6
    by_variant = {(result["variant"], result["sample"]): result for result in summary["results"]}
    assert len(by_variant) == 6
    for sample in (1, 2):
        good = by_variant[("good", sample)]
        wrong = by_variant[("wrong-target", sample)]
        harsh = by_variant[("harsh-judge", sample)]
        assert good["passed"] is True
        assert good["native_checks"][0]["passed"] is True
        assert wrong["passed"] is False
        assert wrong["native_checks"][0]["passed"] is False
        assert all(judgment["passed"] for judgment in wrong["judgments"])
        assert harsh["passed"] is False
        assert harsh["native_checks"][0]["passed"] is True
        verdicts = {item["criterion"]: item["passed"] for item in harsh["judgments"]}
        assert verdicts == {"read-after-write": True, "acknowledged": False}
        for result in (good, wrong, harsh):
            assert result["public_messages"] == 5
            roles = [call["role"] for call in result["model_calls"]]
            assert roles.count("target") == 4
            assert roles.count("judge") == 1
            assert all(
                call["duration_seconds"] is None
                for call in result["model_calls"]
                if call["role"] == "target"
            )
    comparisons = summary["comparisons"]
    assert comparisons["good-vs-wrong-target"]["only_left_passed"] == 2
    assert comparisons["good-vs-wrong-target"]["only_right_passed"] == 0
    assert comparisons["good-vs-harsh-judge"]["only_left_passed"] == 2
    assert comparisons["good-vs-harsh-judge"]["only_right_passed"] == 0

    outcomes = tmp_path / "outcomes"
    saved = json.loads((outcomes / "summary.json").read_text(encoding="utf-8"))
    assert saved["trials"] == 6
    receipts = sorted(outcomes.glob("trials/*/*/*/attempt-*/result.json"))
    assert len(receipts) == 6
    traces = sorted(outcomes.glob("trials/*/*/*/attempt-*/events.jsonl"))
    assert len(traces) == 6
    for trace in traces:
        facts = [
            json.loads(line)["fact"] for line in trace.read_text(encoding="utf-8").splitlines()
        ]
        kinds = [fact["kind"] for fact in facts]
        assert kinds.count("tool") >= 3
        assert "model-call" in kinds
        assert "message" in kinds
    export_states = sorted(outcomes.glob("trials/*/*/*/attempt-*/trace-exports.json"))
    assert len(export_states) == 6
    for state_path in export_states:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert [item["kind"] for item in state["succeeded"]] == ["recording-export-double"]
        errors = json.loads(
            (state_path.parent / "trace-export-errors.json").read_text(encoding="utf-8")
        )
        assert len(errors) == 1
        assert "intentional export failure" in errors[0]["error"]
    trajectories = sorted(outcomes.glob("trials/*/*/*/attempt-*/harbor/**/agent/trajectory.json"))
    assert len(trajectories) == 6
    with pytest.raises(ValueError, match="refusing to overwrite"):
        verification.verify(
            VerifyConfig(
                directory=outcomes,
                executable=fake_cli,
                samples=2,
                workers=4,
                docker_capacity=2,
            )
        )


def test_verifier_rejects_an_unsupported_harbor_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "harbor"
    executable.write_text(f"#!{sys.executable}\nprint('0.2.0')\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.verify_harbor_multiturn",
            "--directory",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode != 0
    assert "Harbor 0.1.44 is required" in result.stderr
    assert "0.2.0" in result.stderr
    assert not (tmp_path / "run").exists()


@pytest.mark.skipif(HARBOR_EXECUTABLE is None, reason="Harbor is not installed")
def test_fixture_agent_rejects_remote_hosts_with_localhost_prefixes(tmp_path: Path) -> None:
    assert HARBOR_EXECUTABLE is not None
    driver = (
        "from pathlib import Path\n"
        "from harbor_multiturn_agent import MultiturnFixtureAgent\n"
        f"logs = Path({str(tmp_path)!r})\n"
        "for endpoint in ('http://localhost.example.invalid/v1', "
        "'http://127.0.0.1.example.invalid/v1'):\n"
        "    try:\n"
        "        MultiturnFixtureAgent(logs_dir=logs, endpoint=endpoint)\n"
        "    except ValueError:\n"
        "        continue\n"
        "    raise AssertionError('accepted remote fixture endpoint: ' + endpoint)\n"
    )
    environment = {**os.environ, "PYTHONPATH": str(FIXTURES_DIR)}

    result = subprocess.run(
        [str(_harbor_python(Path(HARBOR_EXECUTABLE))), "-c", driver],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
