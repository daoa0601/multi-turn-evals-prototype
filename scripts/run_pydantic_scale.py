"""Run a retained high-concurrency campaign through Pydantic AI components."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import platform
import subprocess
import time
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

from pydantic_ai import ModelSettings
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_multiturn_evals import (
    Campaign,
    Case,
    Completed,
    ComponentDefinition,
    Criterion,
    ModelCall,
    TrialResult,
    Variant,
    run,
)
from pydantic_multiturn_evals.harnesses import ConversationHarness
from pydantic_multiturn_evals.providers import (
    BoundModel,
    PydanticActor,
    PydanticJudge,
    PydanticTarget,
)

ModelFunction = Callable[[list[ModelMessage], AgentInfo], Awaitable[ModelResponse]]
VARIANTS = ("plain", "tagged")


@dataclass(frozen=True, slots=True)
class ScaleConfig:
    directory: Path
    cases: int = 100
    samples: int = 5
    workers: int = 128
    model_delay_ms: float = 20

    def __post_init__(self) -> None:
        if self.cases < 1 or self.samples < 1 or self.workers < 1:
            raise ValueError("cases, samples, and workers must be positive")
        if self.model_delay_ms < 0:
            raise ValueError("model delay must not be negative")

    @property
    def trials(self) -> int:
        return self.cases * self.samples * len(VARIANTS)


class ConcurrencyMeter:
    def __init__(self) -> None:
        self.active = 0
        self.maximum = 0
        self.calls: Counter[str] = Counter()

    @asynccontextmanager
    async def measure(self, role: str, delay_seconds: float) -> AsyncIterator[None]:
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        self.calls[role] += 1
        try:
            await asyncio.sleep(delay_seconds)
            yield
        finally:
            self.active -= 1


async def execute(config: ScaleConfig) -> dict[str, object]:
    if (config.directory / "campaign.json").exists():
        raise ValueError(f"scale output already contains a campaign: {config.directory}")
    meter = ConcurrencyMeter()
    campaign = _campaign(config, meter)
    started = time.monotonic()
    first = await run(campaign, directory=config.directory, workers=config.workers)
    elapsed = time.monotonic() - started
    _verify_results(first.results, campaign)
    recorded_calls = tuple(call for result in first.results for call in result.model_calls)

    receipts = tuple(config.directory.glob("trials/**/result.json"))
    traces = tuple(config.directory.glob("trials/**/events.jsonl"))
    if len(receipts) != config.trials or len(traces) != config.trials:
        raise RuntimeError(
            f"expected {config.trials} receipts and traces, got {len(receipts)} and {len(traces)}"
        )
    expected_calls = config.trials * 4
    actual_calls = sum(meter.calls.values())
    if actual_calls != expected_calls:
        raise RuntimeError(f"expected {expected_calls} model calls, got {actual_calls}")
    if len(recorded_calls) != expected_calls:
        raise RuntimeError(
            f"expected {expected_calls} durable model-call receipts, got {len(recorded_calls)}"
        )
    minimum_concurrency = min(config.workers, config.trials, 32)
    if meter.maximum < minimum_concurrency:
        raise RuntimeError(
            f"model concurrency reached {meter.maximum}, expected at least {minimum_concurrency}"
        )

    calls_before_resume = actual_calls
    resumed = await run(campaign, directory=config.directory, workers=config.workers)
    if resumed.executed_trials != 0 or sum(meter.calls.values()) != calls_before_resume:
        raise RuntimeError("resume replayed completed Pydantic trials")

    summary: dict[str, object] = {
        "campaign": campaign.name,
        "revision": campaign.revision,
        "directory": str(config.directory.resolve()),
        "cases": config.cases,
        "samples": config.samples,
        "variants": list(VARIANTS),
        "trials": config.trials,
        "workers": config.workers,
        "model_delay_ms": config.model_delay_ms,
        "model_calls": dict(sorted(meter.calls.items())),
        "recorded_model_calls": len(recorded_calls),
        "recorded_calls_by_role": dict(
            sorted(Counter(call.role for call in recorded_calls).items())
        ),
        "usage": _usage_summary(recorded_calls),
        "peak_model_concurrency": meter.maximum,
        "elapsed_seconds": round(elapsed, 6),
        "trials_per_second": round(config.trials / elapsed, 3),
        "receipts": len(receipts),
        "traces": len(traces),
        "passed": sum(result.passed for result in first.results),
        "resume_executed_trials": resumed.executed_trials,
        "pydantic_ai_version": importlib.metadata.version("pydantic-ai-slim"),
        "python_version": platform.python_version(),
        "source_state": _source_state(),
    }
    _write_json(config.directory / "scale-summary.json", summary)
    return summary


def _campaign(config: ScaleConfig, meter: ConcurrencyMeter) -> Campaign:
    delay_seconds = config.model_delay_ms / 1000
    actor = PydanticActor(
        _model("actor", _actor_model(meter, delay_seconds), config),
        instructions="Send the task once, then accept its resolved answer.",
    )
    judge = PydanticJudge(_model("judge", _judge_model(meter, delay_seconds), config))
    variants = tuple(
        Variant(
            name=profile,
            harness=ConversationHarness(
                target=PydanticTarget(
                    _model(
                        f"target-{profile}",
                        _target_model(meter, delay_seconds, profile),
                        config,
                    ),
                    instructions=f"Resolve the request using the {profile} response profile.",
                ),
                actor=actor,
            ),
            evaluators=(judge,),
        )
        for profile in VARIANTS
    )
    cases = tuple(
        Case(
            name=f"request-{index:04d}",
            task=f"Resolve synthetic request {index:04d}.",
            data={"request_id": index},
            criteria=(
                Criterion(
                    name="resolved",
                    instruction="The public answer explicitly resolves this request.",
                ),
            ),
        )
        for index in range(config.cases)
    )
    return Campaign(
        name="pydantic-scale",
        revision=(
            f"c{config.cases}-s{config.samples}-w{config.workers}-d{config.model_delay_ms:g}ms"
        ),
        cases=cases,
        variants=variants,
        samples=config.samples,
    )


def _model(
    role: str,
    function: ModelFunction,
    config: ScaleConfig,
) -> BoundModel:
    return BoundModel(
        model=FunctionModel(function, model_name=f"scale-{role}"),
        settings=ModelSettings(),
        provider="function",
        interface="function",
        definition=ComponentDefinition(
            kind="pydantic-function-model",
            version="1",
            config={"role": role, "delay_ms": config.model_delay_ms},
        ),
    )


def _actor_model(meter: ConcurrencyMeter, delay: float) -> ModelFunction:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        async with meter.measure("actor", delay):
            prompt = _last_prompt(messages)
            turn = _xml_text(prompt, "turn")
            if turn == "0":
                output = {
                    "kind": "continue",
                    "reason": "send the authored task",
                    "message": _xml_text(prompt, "task"),
                }
            else:
                output = {"kind": "accept", "reason": "the target resolved the task"}
            return _structured(info, output)

    return respond


def _target_model(
    meter: ConcurrencyMeter,
    delay: float,
    profile: str,
) -> ModelFunction:
    async def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        async with meter.measure("target", delay):
            task = _last_prompt(messages)
            if profile == "plain":
                answer = f"resolved: {task}"
            else:
                answer = f"resolved [{profile}]: {task}"
            return ModelResponse(parts=[TextPart(answer)])

    return respond


def _judge_model(meter: ConcurrencyMeter, delay: float) -> ModelFunction:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        async with meter.measure("judge", delay):
            public_input = _last_prompt(messages)
            passed = "resolved" in public_input
            return _structured(
                info,
                {
                    "judgments": [
                        {
                            "criterion": "resolved",
                            "passed": passed,
                            "reason": "the public answer contains the resolved outcome",
                            "score": float(passed),
                        }
                    ]
                },
            )

    return respond


def _structured(info: AgentInfo, value: Mapping[str, object]) -> ModelResponse:
    if len(info.output_tools) != 1:
        raise RuntimeError("scale model expected one structured output tool")
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, dict(value))])


def _last_prompt(messages: list[ModelMessage]) -> str:
    for message in reversed(messages):
        if not isinstance(message, ModelRequest):
            continue
        for part in reversed(message.parts):
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                return part.content
    raise RuntimeError("Pydantic model request has no text prompt")


def _xml_text(value: str, name: str) -> str:
    root = ElementTree.fromstring(f"<root>{value}</root>")
    element = root.find(name)
    if element is None or element.text is None:
        raise RuntimeError(f"Pydantic actor prompt has no {name!r} field")
    return element.text


def _verify_results(results: tuple[TrialResult, ...], campaign: Campaign) -> None:
    if len(results) != len(campaign.cases) * campaign.samples * len(campaign.variants):
        raise RuntimeError("campaign returned the wrong number of trial results")
    expected_tasks = {case.name: case.task for case in campaign.cases}
    identities: set[tuple[str, str, int]] = set()
    for result in results:
        identity = (result.key.case, result.key.variant, result.key.sample)
        if identity in identities:
            raise RuntimeError(f"campaign returned duplicate result {identity}")
        identities.add(identity)
        if not isinstance(result.terminal, Completed) or not result.passed:
            raise RuntimeError(f"Pydantic trial did not pass: {result.key}")
        if [call.role for call in result.model_calls] != ["actor", "target", "actor", "judge"]:
            raise RuntimeError(f"Pydantic trial has incomplete model-call receipts: {result.key}")
        if any(call.status != "completed" or call.usage is None for call in result.model_calls):
            raise RuntimeError(f"Pydantic trial has incomplete model usage: {result.key}")
        messages = result.terminal.evidence.public.messages
        if len(messages) != 2 or expected_tasks[result.key.case] not in messages[-1].content:
            raise RuntimeError(f"Pydantic trial lost authored task data: {result.key}")


def _usage_summary(calls: Sequence[ModelCall]) -> dict[str, int | float | None]:
    usages = [call.usage for call in calls if call.usage is not None]
    costs = [usage.cost_usd for usage in usages if usage.cost_usd is not None]
    input_tokens = [usage.input_tokens for usage in usages]
    output_tokens = [usage.output_tokens for usage in usages]
    return {
        "requests": sum(usage.requests for usage in usages),
        "tool_calls": sum(usage.tool_calls for usage in usages),
        "input_tokens": (
            None
            if None in input_tokens
            else sum(value for value in input_tokens if value is not None)
        ),
        "output_tokens": (
            None
            if None in output_tokens
            else sum(value for value in output_tokens if value is not None)
        ),
        "known_cost_usd": round(sum(costs), 9),
        "calls_with_known_cost": len(costs),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_state() -> str:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return "dirty-worktree" if result.stdout else "clean-worktree"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--model-delay-ms", type=float, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    directory = arguments.directory
    if directory is None:
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
        directory = Path("outputs") / "pydantic-scale" / timestamp
    config = ScaleConfig(
        directory=directory,
        cases=arguments.cases,
        samples=arguments.samples,
        workers=arguments.workers,
        model_delay_ms=arguments.model_delay_ms,
    )
    summary = asyncio.run(execute(config))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
