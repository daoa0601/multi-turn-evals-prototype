"""Verify the real Harbor multi-turn path end to end.

The campaign is the sole scheduler and result writer. Harbor owns one Docker
task with a native verifier. A custom Harbor ``BaseAgent`` runs inside native
Harbor, makes real HTTP model requests to a deterministic local
OpenAI-compatible fixture, executes real ``environment.exec`` commands, and
records a truthful ATIF trajectory. The imported public dialogue reaches an
actual ``PydanticJudge`` backed by deterministic ``FunctionModel`` policies.

Three variants prove the separation of concerns:

- ``good``: native verifier passes and the judge passes.
- ``wrong-target``: the agent stores the answer in the wrong directory. The
  judge still passes, but the native check fails, so the trial fails.
- ``harsh-judge``: the native verifier passes, but a strict evaluator policy
  fails one criterion, so the trial fails on evaluator judgment alone.

Model responses, tool choices, usage numbers, and judge verdicts are
deterministic test fixtures. This script verifies plumbing and evidence
handling, never model-agent quality, and it contacts no real model provider.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic_ai import ModelSettings
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from integration.harbor import harbor_multiturn_fixtures as fixtures
from pydantic_multiturn_evals import (
    Campaign,
    Case,
    Completed,
    Criterion,
    Failed,
    Report,
    ResourceClaim,
    TrialBudget,
    TrialResult,
    Variant,
    run,
)
from pydantic_multiturn_evals.adapters import HarborHarness, HarborLocalTask
from pydantic_multiturn_evals.model import (
    ComponentDefinition,
    Evidence,
    TraceEvent,
    TrialKey,
)
from pydantic_multiturn_evals.providers import BoundModel, PydanticJudge

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "integration" / "harbor"
TASK_DIR = FIXTURES_DIR / "multiturn-task"
AGENT_IMPORT_PATH = "harbor_multiturn_agent:MultiturnFixtureAgent"
TASK_REVISION = "1.0.0"
CAMPAIGN_NAME = "harbor-multiturn-integration"
CAMPAIGN_REVISION = "1"
CASE_NAME = "confirm-state"
HARBOR_VERSION = "0.1.44"
NATIVE_CHECK = "harbor.verified"
CRITERION_READ = "read-after-write"
CRITERION_ACK = "acknowledged"
ACK_PHRASE = "harbor-ready"
MODEL_REQUESTS_PER_TRIAL = 4
MODEL_TOOL_CALLS_PER_TRIAL = 3
DERIVED_PATTERN = re.compile(r"anchor-[0-9a-f]+-[A-Za-z0-9][A-Za-z0-9._-]*")
VARIANTS = ("good", "wrong-target", "harsh-judge")
MAX_SAMPLES = 8
MAX_WORKERS = 16
MAX_DOCKER_CAPACITY = 8


@dataclass(frozen=True, slots=True)
class VerifyConfig:
    directory: Path
    executable: Path
    harbor_version: str = HARBOR_VERSION
    samples: int = 2
    workers: int = 4
    docker_capacity: int = 2
    timeout_seconds: float = 600

    def __post_init__(self) -> None:
        if not 1 <= self.samples <= MAX_SAMPLES:
            raise ValueError(f"samples must be between 1 and {MAX_SAMPLES}")
        if not 2 <= self.workers <= MAX_WORKERS:
            raise ValueError(f"workers must be between 2 and {MAX_WORKERS}")
        if not 1 <= self.docker_capacity <= MAX_DOCKER_CAPACITY:
            raise ValueError(f"docker-capacity must be between 1 and {MAX_DOCKER_CAPACITY}")
        if self.workers <= self.docker_capacity:
            raise ValueError("workers must exceed docker-capacity to exercise capacity gating")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout must be positive")

    @property
    def trials(self) -> int:
        return self.samples * len(VARIANTS)


def judge_binding(policy: str, calls: Counter[str]) -> BoundModel:
    """Build a deterministic PydanticJudge model policy."""

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls[f"judge-{policy}"] += 1
        prompt = _last_prompt(messages)
        judgments = [
            {"criterion": CRITERION_READ, **_read_after_write(prompt)},
            {"criterion": CRITERION_ACK, **_acknowledged(prompt, policy)},
        ]
        if len(info.output_tools) != 1:
            raise RuntimeError("judge model expected one structured output tool")
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"judgments": judgments})]
        )

    return BoundModel(
        model=FunctionModel(respond, model_name=f"harbor-multiturn-judge-{policy}"),
        settings=ModelSettings(),
        provider="function",
        interface="function",
        definition=ComponentDefinition(
            kind="pydantic-function-model",
            version="1",
            config={
                "role": "judge",
                "policy": policy,
                "fixture": "deterministic-criteria-policy",
            },
        ),
    )


def _read_after_write(prompt: str) -> dict[str, Any]:
    # The derived seed-hostname value must appear both in the stored answer that
    # was read back and in the agent's closing summary. This works no matter how
    # tool-call arguments are rendered in the imported public evidence.
    matches = list(DERIVED_PATTERN.finditer(prompt))
    passed = len(matches) >= 2
    return {
        "passed": passed,
        "score": float(passed),
        "reason": (
            "derived seed-hostname value appears in both the read-back observation "
            "and the agent's closing summary"
            if passed
            else "derived seed-hostname value does not appear in both the read-back "
            "observation and the agent's closing summary"
        ),
    }


def _acknowledged(prompt: str, policy: str) -> dict[str, Any]:
    required = ACK_PHRASE.upper() if policy == "harsh" else ACK_PHRASE
    passed = required in prompt
    return {
        "passed": passed,
        "score": float(passed),
        "reason": (
            f"final agent message states the confirmation phrase {required!r} "
            "required by this evaluator policy"
            if passed
            else f"confirmation phrase {required!r} required by this evaluator "
            "policy is absent from the transcript"
        ),
    }


def _last_prompt(messages: list[ModelMessage]) -> str:
    for message in reversed(messages):
        if not isinstance(message, ModelRequest):
            continue
        for part in reversed(message.parts):
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                return part.content
    raise RuntimeError("judge model request has no text prompt")


def build_campaign(
    config: VerifyConfig,
    good_endpoint: str,
    wrong_endpoint: str,
    judge_calls: Counter[str],
) -> Campaign:
    task = HarborLocalTask(case=CASE_NAME, path=TASK_DIR, revision=TASK_REVISION)

    def harness(endpoint: str, behavior: str) -> HarborHarness:
        return HarborHarness(
            executable=config.executable,
            harbor_version=config.harbor_version,
            tasks=(task,),
            agent_import_path=AGENT_IMPORT_PATH,
            agent_arguments={"endpoint": endpoint, "behavior": behavior},
            public_trajectory=True,
            model="fixture/multiturn-deterministic",
            environment="docker",
            inherit_environment=("PATH", "HOME", "PYTHONPATH"),
            timeout_seconds=config.timeout_seconds,
        )

    standard = PydanticJudge(judge_binding("standard", judge_calls))
    harsh = PydanticJudge(judge_binding("harsh", judge_calls))
    case = Case(
        name=CASE_NAME,
        task=(
            "Complete the Harbor stateful task: read the deployment seed, store "
            "the derived answer, and confirm the stored value."
        ),
        criteria=(
            Criterion(
                name=CRITERION_READ,
                instruction=(
                    "The transcript shows the agent reading environment state, "
                    "storing a derived answer, and reading the stored value back."
                ),
            ),
            Criterion(
                name=CRITERION_ACK,
                instruction=("The final agent message states the required confirmation phrase."),
            ),
        ),
        required_checks=(NATIVE_CHECK,),
        budget=TrialBudget(seconds=config.timeout_seconds + 300),
    )
    return Campaign(
        name=CAMPAIGN_NAME,
        revision=CAMPAIGN_REVISION,
        cases=(case,),
        variants=(
            Variant(
                name="good",
                harness=harness(good_endpoint, "good"),
                evaluators=(standard,),
                resources=(ResourceClaim(name="docker"),),
            ),
            Variant(
                name="wrong-target",
                harness=harness(wrong_endpoint, "wrong-target"),
                evaluators=(standard,),
                resources=(ResourceClaim(name="docker"),),
            ),
            Variant(
                name="harsh-judge",
                harness=harness(good_endpoint, "good"),
                evaluators=(harsh,),
                resources=(ResourceClaim(name="docker"),),
            ),
        ),
        samples=config.samples,
    )


@dataclass
class RecordingExporter:
    """Test double for the common trace exporter contract."""

    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def component(self) -> ComponentDefinition:
        return ComponentDefinition(
            kind="recording-export-double",
            version="1",
            config={"purpose": "exercise the common exporter contract"},
        )

    async def export(
        self,
        key: TrialKey,
        events: Sequence[TraceEvent],
        result: TrialResult,
        export_id: str,
    ) -> None:
        self.records.append(
            {
                "key": key.model_dump(mode="json"),
                "attempt": result.attempt,
                "events": len(events),
                "export_id": export_id,
                "terminal": result.terminal.kind,
            }
        )


@dataclass(frozen=True)
class FailingExporter:
    """Test double that always fails to prove export error isolation."""

    @property
    def component(self) -> ComponentDefinition:
        return ComponentDefinition(
            kind="failing-export-double",
            version="1",
            config={"fails": "intentional isolation check"},
        )

    async def export(
        self,
        key: TrialKey,
        events: Sequence[TraceEvent],
        result: TrialResult,
        export_id: str,
    ) -> None:
        del key, events, result, export_id
        raise RuntimeError("intentional export failure for isolation check")


def verify(config: VerifyConfig) -> dict[str, Any]:
    _require_docker()
    config.directory.mkdir(parents=True, exist_ok=True)
    if (config.directory / "campaign.json").exists():
        raise ValueError(f"refusing to overwrite an existing campaign in {config.directory}")
    good_server = fixtures.FixtureModelServer(("127.0.0.1", 0), "good")
    wrong_server = fixtures.FixtureModelServer(("127.0.0.1", 0), "wrong-target")
    good_thread = good_server.start()
    wrong_thread = wrong_server.start()
    prior_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(FIXTURES_DIR), prior_pythonpath) if part
    )
    judge_calls: Counter[str] = Counter()
    recorder = RecordingExporter()
    exporters = (recorder, FailingExporter())
    try:
        campaign = build_campaign(config, good_server.endpoint, wrong_server.endpoint, judge_calls)
        started = time.monotonic()
        report = asyncio.run(
            run(
                campaign,
                directory=config.directory,
                workers=config.workers,
                capacities={"docker": config.docker_capacity},
                exporters=exporters,
                export_timeout_seconds=30,
            )
        )
        elapsed = time.monotonic() - started
        requests_before_resume = {
            "good": len(good_server.requests),
            "wrong-target": len(wrong_server.requests),
        }
        judge_calls_before_resume = sum(judge_calls.values())
        records_before_resume = len(recorder.records)
        resumed = asyncio.run(
            run(
                campaign,
                directory=config.directory,
                workers=config.workers,
                capacities={"docker": config.docker_capacity},
                exporters=exporters,
                export_timeout_seconds=30,
            )
        )
    finally:
        good_server.shutdown()
        wrong_server.shutdown()
        good_thread.join()
        wrong_thread.join()
        good_server.server_close()
        wrong_server.server_close()
        if prior_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = prior_pythonpath

    _assert_trials(config, report)
    _assert_resume(
        config,
        resumed,
        good_server,
        wrong_server,
        requests_before_resume,
        judge_calls,
        judge_calls_before_resume,
        recorder,
        records_before_resume,
    )
    derived_values = _assert_model_fixture_usage(config, good_server, wrong_server)
    _assert_export_state(config, recorder)
    _assert_cleanup(report)
    comparisons = {
        "good-vs-wrong-target": report.compare("good", "wrong-target").model_dump(mode="json"),
        "good-vs-harsh-judge": report.compare("good", "harsh-judge").model_dump(mode="json"),
    }
    summary = {
        "campaign": campaign.name,
        "revision": campaign.revision,
        "directory": str(config.directory.resolve()),
        "harbor_version": config.harbor_version,
        "agent_import_path": AGENT_IMPORT_PATH,
        "task": TASK_DIR.name,
        "samples": config.samples,
        "workers": config.workers,
        "docker_capacity": config.docker_capacity,
        "trials": len(report.results),
        "executed_trials": report.executed_trials,
        "resume_executed_trials": resumed.executed_trials,
        "elapsed_seconds": round(elapsed, 6),
        "results": [_result_summary(result) for result in report.results],
        "comparisons": comparisons,
        "model_fixture": {
            "label": fixtures.FIXTURE_LABEL,
            "requests_good": len(good_server.requests),
            "requests_wrong_target": len(wrong_server.requests),
            "requests_per_trial": MODEL_REQUESTS_PER_TRIAL,
            "distinct_derived_values": len(derived_values),
            "failures": [*good_server.failures, *wrong_server.failures],
        },
        "judge": {
            "fixture": "deterministic FunctionModel policies via PydanticJudge",
            "calls": dict(sorted(judge_calls.items())),
        },
        "exports": {
            "recorded": len(recorder.records),
            "failing_exporter": "failing-export-double@1 (intentional, isolated)",
        },
        "disclaimer": (
            "Model responses, tool choices, usage numbers, and judge verdicts are "
            "deterministic test fixtures. This verification covers Harbor plumbing, "
            "native verification, evidence import, and evaluator separation. It "
            "makes no claim about model-agent quality and contacts no real model "
            "provider."
        ),
    }
    _write_json(config.directory / "summary.json", summary)
    return summary


def _assert_trials(config: VerifyConfig, report: Report) -> None:
    if report.executed_trials != config.trials or len(report.results) != config.trials:
        raise RuntimeError(
            f"expected {config.trials} fresh trials, got {report.executed_trials} "
            f"executed and {len(report.results)} results; receipts: "
            f"{config.directory / 'trials'}"
        )
    for result in report.results:
        directory = _attempt_directory(config.directory, result)
        terminal = result.terminal
        if not isinstance(terminal, Completed):
            reason = terminal.failure.message if isinstance(terminal, Failed) else terminal.reason
            raise RuntimeError(
                f"trial {result.key.case}/{result.key.variant}/"
                f"{result.key.sample} terminated as {terminal.kind} "
                f"({reason}); "
                f"receipt: {directory / 'result.json'}"
            )
        evidence = terminal.evidence
        native = {check.name: check for check in evidence.checks}.get(NATIVE_CHECK)
        if native is None:
            raise RuntimeError(
                f"trial {result.key.variant}/{result.key.sample} has no "
                f"{NATIVE_CHECK} check; receipt: {directory / 'result.json'}"
            )
        judgments = {item.criterion: item for item in terminal.judgments}
        if set(judgments) != {CRITERION_READ, CRITERION_ACK}:
            raise RuntimeError(
                f"trial {result.key.variant}/{result.key.sample} judgments "
                f"{sorted(judgments)} do not match the case criteria; receipt: "
                f"{directory / 'result.json'}"
            )
        if result.key.variant == "good":
            if not native.passed or not result.passed:
                raise RuntimeError(
                    f"good trial failed: native={native.passed} overall="
                    f"{result.passed} reason={native.reason}; receipt: "
                    f"{directory / 'result.json'}"
                )
        elif result.key.variant == "wrong-target":
            if native.passed:
                raise RuntimeError(
                    f"wrong-target trial passed the native verifier: {native.reason}; "
                    f"receipt: {directory / 'result.json'}"
                )
            if result.passed:
                raise RuntimeError(
                    "wrong-target trial was rescued despite a failed native check; "
                    f"receipt: {directory / 'result.json'}"
                )
            if not all(item.passed for item in judgments.values()):
                raise RuntimeError(
                    "wrong-target trial should expose judge blindness: expected all "
                    f"judgments to pass, got "
                    f"{ {name: item.passed for name, item in judgments.items()} }; "
                    f"receipt: {directory / 'result.json'}"
                )
        elif result.key.variant == "harsh-judge":
            if not native.passed:
                raise RuntimeError(
                    f"harsh-judge trial failed natively: {native.reason}; receipt: "
                    f"{directory / 'result.json'}"
                )
            if result.passed:
                raise RuntimeError(
                    "harsh-judge trial should fail on evaluator judgment alone; "
                    f"receipt: {directory / 'result.json'}"
                )
            if judgments[CRITERION_ACK].passed or not judgments[CRITERION_READ].passed:
                raise RuntimeError(
                    "harsh-judge trial has the wrong judgment split: "
                    f"acknowledged={judgments[CRITERION_ACK].passed} "
                    f"read-after-write={judgments[CRITERION_READ].passed}; receipt: "
                    f"{directory / 'result.json'}"
                )
        _assert_public_evidence(config, result, evidence, directory)


def _assert_public_evidence(
    config: VerifyConfig,
    result: TrialResult,
    evidence: Evidence,
    directory: Path,
) -> None:
    del config
    identity = f"{result.key.variant}/{result.key.sample}"
    messages = evidence.public.messages
    transcript = "\n".join(message.content for message in messages)
    if len(messages) < 5:
        raise RuntimeError(
            f"trial {identity} imported only {len(messages)} public messages, "
            f"expected the multi-turn transcript; receipt: {directory / 'events.jsonl'}"
        )
    if not DERIVED_PATTERN.search(transcript):
        raise RuntimeError(
            f"trial {identity} transcript lost the derived seed-hostname value "
            "observed in the environment; receipt: "
            f"{directory / 'events.jsonl'}"
        )
    if "exit=0" not in transcript:
        raise RuntimeError(
            f"trial {identity} transcript lost the environment observation text; "
            f"receipt: {directory / 'events.jsonl'}"
        )
    if ACK_PHRASE not in transcript:
        raise RuntimeError(
            f"trial {identity} transcript lost the final agent acknowledgement; "
            f"receipt: {directory / 'events.jsonl'}"
        )
    events_path = directory / "events.jsonl"
    facts = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    tool_facts = [fact for fact in facts if fact["fact"].get("kind") == "tool"]
    if len(tool_facts) < MODEL_TOOL_CALLS_PER_TRIAL:
        raise RuntimeError(
            f"trial {identity} recorded {len(tool_facts)} typed tool facts, expected "
            f"at least {MODEL_TOOL_CALLS_PER_TRIAL}; receipt: {events_path}"
        )
    judge_calls = [call for call in result.model_calls if call.role == "judge"]
    atif_calls = [call for call in result.model_calls if call.duration_seconds is None]
    if not judge_calls or any(call.status != "completed" for call in judge_calls):
        raise RuntimeError(
            f"trial {identity} has no completed judge model-call receipt; receipt: "
            f"{directory / 'result.json'}"
        )
    if not atif_calls or any(
        call.usage is None or call.usage.input_tokens <= 0 for call in atif_calls
    ):
        raise RuntimeError(
            f"trial {identity} has no ATIF-derived model usage receipt with tokens; "
            f"receipt: {directory / 'result.json'}"
        )
    artifact_paths = [artifact.path for artifact in evidence.artifacts]
    if not any(path.endswith("trajectory.json") for path in artifact_paths):
        raise RuntimeError(
            f"trial {identity} evidence has no harbor trajectory artifact; artifacts: "
            f"{artifact_paths}; receipt: {directory / 'result.json'}"
        )
    for artifact in evidence.artifacts:
        if not (directory / artifact.path).is_file():
            raise RuntimeError(
                f"trial {identity} artifact {artifact.path} is missing; receipt: "
                f"{directory / 'result.json'}"
            )
    trajectory_files = [
        directory / path for path in artifact_paths if path.endswith("trajectory.json")
    ]
    trajectory = json.loads(trajectory_files[0].read_text(encoding="utf-8"))
    expected_behavior = "wrong-target" if result.key.variant == "wrong-target" else "good"
    if trajectory["agent"]["extra"]["behavior"] != expected_behavior:
        raise RuntimeError(
            f"trial {identity} trajectory behavior "
            f"{trajectory['agent']['extra']['behavior']!r} does not match variant "
            f"{expected_behavior!r}; receipt: {trajectory_files[0]}"
        )
    requests = trajectory["final_metrics"]["extra"]["model_requests"]
    if requests != MODEL_REQUESTS_PER_TRIAL:
        raise RuntimeError(
            f"trial {identity} trajectory reports {requests} model requests, "
            f"expected {MODEL_REQUESTS_PER_TRIAL}; receipt: {trajectory_files[0]}"
        )


def _assert_resume(
    config: VerifyConfig,
    resumed: Report,
    good_server: fixtures.FixtureModelServer,
    wrong_server: fixtures.FixtureModelServer,
    requests_before_resume: Mapping[str, int],
    judge_calls: Counter[str],
    judge_calls_before_resume: int,
    recorder: RecordingExporter,
    records_before_resume: int,
) -> None:
    if resumed.executed_trials != 0:
        raise RuntimeError(
            f"resume executed {resumed.executed_trials} trials instead of zero; "
            f"receipts: {config.directory / 'trials'}"
        )
    if (
        len(good_server.requests) != requests_before_resume["good"]
        or len(wrong_server.requests) != requests_before_resume["wrong-target"]
    ):
        raise RuntimeError(
            "resume replayed fixture model requests: good "
            f"{len(good_server.requests)} != {requests_before_resume['good']}, "
            "wrong-target "
            f"{len(wrong_server.requests)} != {requests_before_resume['wrong-target']}"
        )
    if sum(judge_calls.values()) != judge_calls_before_resume:
        raise RuntimeError(
            "resume replayed judge requests: "
            f"{sum(judge_calls.values())} != {judge_calls_before_resume}"
        )
    if len(recorder.records) != records_before_resume:
        raise RuntimeError(
            "resume re-exported already-succeeded traces: "
            f"{len(recorder.records)} != {records_before_resume}"
        )


def _assert_model_fixture_usage(
    config: VerifyConfig,
    good_server: fixtures.FixtureModelServer,
    wrong_server: fixtures.FixtureModelServer,
) -> set[str]:
    expected = {
        "good": MODEL_REQUESTS_PER_TRIAL * config.samples * 2,
        "wrong-target": MODEL_REQUESTS_PER_TRIAL * config.samples,
    }
    observed = {"good": len(good_server.requests), "wrong-target": len(wrong_server.requests)}
    if observed != expected:
        raise RuntimeError(
            f"fixture model request counts {observed} do not match {expected}; "
            "agent receipts: " + str(sorted((config.directory / "trials").glob("**/attempt-*")))
        )
    derived_values: set[str] = set()
    for server, behavior in ((good_server, "good"), (wrong_server, "wrong-target")):
        for index, body in enumerate(server.requests):
            messages = body.get("messages") or []
            results = fixtures.tool_result_messages(messages)
            if len(results) == 1:
                seed, host = fixtures.parse_seed_and_host(str(results[0].get("content")))
                reply = server.replies[index]
                command = json.loads(
                    reply["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
                )["command"]
                observed_tokens = (seed, host)
                missing = [token for token in observed_tokens if token not in command]
                if missing:
                    raise RuntimeError(
                        f"{behavior} request {index + 1} wrote {command!r} without the "
                        f"observed environment tokens {missing}; the agent did not "
                        "feed environment observations back to the model"
                    )
            if len(results) == 3:
                match = fixtures.DERIVED_VALUE.search(str(results[2].get("content")))
                if match is None:
                    raise RuntimeError(
                        f"{behavior} request {index + 1} read back an answer without "
                        "the derived seed-hostname value; the environment state never "
                        "reached the model"
                    )
                derived_values.add(match.group(0))
    if len(derived_values) != config.trials:
        raise RuntimeError(
            f"observed {len(derived_values)} distinct derived values across "
            f"{config.trials} trials; trial state is not independent; values: "
            f"{sorted(derived_values)}"
        )
    return derived_values


def _assert_export_state(config: VerifyConfig, recorder: RecordingExporter) -> None:
    if len(recorder.records) != config.trials:
        raise RuntimeError(
            f"recording exporter saw {len(recorder.records)} exports, expected {config.trials}"
        )
    directories = sorted(
        path
        for path in (config.directory / "trials").rglob("attempt-*")
        if path.parent.name.startswith("sample-")
    )
    if len(directories) != config.trials:
        raise RuntimeError(
            f"found {len(directories)} attempt directories, expected {config.trials}"
        )
    for directory in directories:
        state_path = directory / "trace-exports.json"
        errors_path = directory / "trace-export-errors.json"
        if not state_path.is_file():
            raise RuntimeError(f"missing export state receipt {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not any(item["kind"] == "recording-export-double" for item in state["succeeded"]):
            raise RuntimeError(f"recording exporter was not marked successful in {state_path}")
        if not errors_path.is_file():
            raise RuntimeError(
                f"intentional export failure was not recorded in {errors_path.parent}"
            )
        errors = json.loads(errors_path.read_text(encoding="utf-8"))
        if not any(
            "FailingExporter" in entry["exporter"]
            and "intentional export failure" in entry["error"]
            for entry in errors
        ):
            raise RuntimeError(f"missing isolated export failure receipt in {errors_path}")


def _assert_cleanup(report: Report) -> None:
    projects: set[str] = set()
    for result in report.results:
        if isinstance(result.terminal, Completed):
            output = result.terminal.evidence.public.output
            if isinstance(output, dict) and isinstance(output.get("trial_name"), str):
                projects.add(str(output["trial_name"]).lower())
    remaining = _docker_containers(projects)
    if remaining:
        raise RuntimeError(
            f"verification left its own Docker containers behind: {sorted(remaining)}"
        )


def _result_summary(result: TrialResult) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "case": result.key.case,
        "variant": result.key.variant,
        "sample": result.key.sample,
        "terminal": result.terminal.kind,
        "passed": result.passed,
        "duration_seconds": round(result.duration_seconds, 6),
        "model_calls": [
            {
                "role": call.role,
                "operation": call.operation,
                "status": call.status,
                "duration_seconds": call.duration_seconds,
            }
            for call in result.model_calls
        ],
    }
    if isinstance(result.terminal, Completed):
        summary["native_checks"] = [
            check.model_dump(mode="json") for check in result.terminal.evidence.checks
        ]
        summary["judgments"] = [
            judgment.model_dump(mode="json") for judgment in result.terminal.judgments
        ]
        summary["public_messages"] = len(result.terminal.evidence.public.messages)
    return summary


def _attempt_directory(root: Path, result: TrialResult) -> Path:
    return (
        root
        / "trials"
        / result.key.case
        / result.key.variant
        / f"sample-{result.key.sample:04d}"
        / f"attempt-{result.attempt:04d}"
    )


def _docker_containers(projects: set[str]) -> set[str]:
    output = subprocess.run(
        ("docker", "ps", "-a", "--format", '{{.ID}} {{.Label "com.docker.compose.project"}}'),
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        parts[0]
        for line in output.stdout.splitlines()
        if len(parts := line.split()) == 2 and parts[1] in projects
    }


def _require_docker() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is not installed")
    subprocess.run(
        ("docker", "info"),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _harbor_version(executable: Path) -> str:
    result = subprocess.run(
        (str(executable), "--version"),
        check=True,
        capture_output=True,
        text=True,
    )
    version = result.stdout.strip()
    if version != HARBOR_VERSION:
        raise RuntimeError(
            f"Harbor {HARBOR_VERSION} is required; {executable} reported {version!r}"
        )
    return version


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--docker-capacity", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    executable = shutil.which("harbor")
    if executable is None:
        raise SystemExit("Harbor is not installed")
    directory = arguments.directory
    if directory is None:
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
        directory = Path("outputs") / "harbor-multiturn" / timestamp
    config = VerifyConfig(
        directory=directory,
        executable=Path(executable),
        harbor_version=_harbor_version(Path(executable)),
        samples=arguments.samples,
        workers=arguments.workers,
        docker_capacity=arguments.docker_capacity,
        timeout_seconds=arguments.timeout_seconds,
    )
    summary = verify(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
