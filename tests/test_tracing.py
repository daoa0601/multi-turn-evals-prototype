from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pydantic_multiturn_evals import (
    Completed,
    ComponentDefinition,
    Evidence,
    LifecycleFact,
    ModelCall,
    ModelCallFact,
    ModelUsage,
    NativeCheck,
    TrialKey,
    TrialResult,
)
from pydantic_multiturn_evals.tracing import (
    LangfuseExporter,
    LocalTrace,
    export_with_langfuse_client,
)


def test_langfuse_exporter_runs_the_installed_sdk_in_an_isolated_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = TrialKey(campaign="trace", revision="1", case="case", variant="direct", sample=1)
    trace = LocalTrace(key, 1, tmp_path / "events.jsonl")
    trace.record(LifecycleFact(phase="started"))
    trace.close()
    result = TrialResult(
        key=key,
        attempt=1,
        duration_seconds=0,
        terminal=Completed(
            evidence=Evidence(checks=(NativeCheck(name="done", passed=True, reason="completed"),))
        ),
    )
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-local-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-local-test")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    exporter = LangfuseExporter.from_environment()

    asyncio.run(exporter.export(key, trace.events, result, "1" * 32))
    assert "pk-lf-local-test" not in exporter.component.model_dump_json()


class _RecordingLangfuseClient:
    def __init__(self) -> None:
        self.observations: list[dict[str, object]] = []
        self.flushed = False

    def start_as_current_observation(self, **values: object) -> _RecordingLangfuseClient:
        self.observations.append(dict(values))
        return self

    def __enter__(self) -> _RecordingLangfuseClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def update(self, **_values: object) -> None:
        return None

    def score_trace(self, **_values: object) -> None:
        return None

    def flush(self) -> None:
        self.flushed = True


@pytest.mark.parametrize("duration_seconds", [0.5, None])
def test_langfuse_renders_model_call_receipts_as_generations(
    tmp_path: Path, duration_seconds: float | None
) -> None:
    key = TrialKey(campaign="trace", revision="1", case="case", variant="direct", sample=1)
    call = ModelCall(
        role="judge",
        operation="criteria",
        requested=ComponentDefinition(kind="provider-model", version="1"),
        reported_model="provider-model-v2",
        provider="openai",
        interface="responses",
        sampling={"temperature": 0.2, "nested": {"ignored": True}},
        duration_seconds=duration_seconds,
        status="completed",
        usage=ModelUsage(input_tokens=12, output_tokens=3, cost_usd=0.004),
        provider_response_id="response-1",
    )
    trace = LocalTrace(key, 1, tmp_path / "events.jsonl")
    trace.record(ModelCallFact(call=call))
    trace.close()
    result = TrialResult(
        key=key,
        attempt=1,
        duration_seconds=0.5,
        terminal=Completed(
            evidence=Evidence(checks=(NativeCheck(name="done", passed=True, reason="completed"),))
        ),
        model_calls=(call,),
    )
    client = _RecordingLangfuseClient()

    export_with_langfuse_client(client, key, trace.events, result, "1" * 32)

    generation = client.observations[1]
    assert generation["as_type"] == "generation"
    assert generation["name"] == "model:judge:criteria"
    assert generation["model"] == "provider-model-v2"
    metadata = generation["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["duration_seconds"] == duration_seconds
    restored = TrialResult.model_validate_json(result.model_dump_json())
    assert restored.model_calls[0].duration_seconds == duration_seconds
    assert generation["model_parameters"] == {"temperature": 0.2}
    assert generation["usage_details"] == {
        "input": 12,
        "output": 3,
        "total": 15,
        "requests": 0,
        "tool_calls": 0,
        "cache_write_tokens": 0,
        "cache_read_tokens": 0,
        "input_audio_tokens": 0,
        "cache_audio_read_tokens": 0,
        "output_audio_tokens": 0,
    }
    assert generation["cost_details"] == {"total": 0.004}
    assert client.flushed is True
