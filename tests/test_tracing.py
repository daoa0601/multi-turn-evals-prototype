from __future__ import annotations

import asyncio
import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Span

from pydantic_multiturn_evals import (
    Campaign,
    Case,
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
    Variant,
    run,
)
from pydantic_multiturn_evals.harnesses import FunctionHarness
from pydantic_multiturn_evals.model import TraceRecorder, Trial
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
    monkeypatch.setenv("LANGFUSE_PROJECT_ID", "local-project")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    exporter = LangfuseExporter.from_environment()

    asyncio.run(exporter.export(key, trace.events, result, "1" * 32))
    assert "pk-lf-local-test" not in exporter.component.model_dump_json()


@pytest.mark.parametrize(
    "rejected_path, rejected_status",
    [
        ("/api/public/otel/v1/traces", 401),
        ("/api/public/ingestion", 401),
        ("/api/public/ingestion", 207),
    ],
)
def test_rejected_langfuse_delivery_is_retried_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rejected_path: str, rejected_status: int
) -> None:
    reject = True
    requests: list[str] = []
    delivered_spans: list[Span] = []
    delivered_scores: list[dict] = []

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(self.path)
            status = rejected_status if reject and self.path == rejected_path else 200
            if status == 200:
                if self.path == "/api/public/otel/v1/traces":
                    if self.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    payload = ExportTraceServiceRequest.FromString(body)
                    delivered_spans.extend(
                        span
                        for resource in payload.resource_spans
                        for scope in resource.scope_spans
                        for span in scope.spans
                    )
                elif self.path == "/api/public/ingestion":
                    delivered_scores.extend(item["body"] for item in json.loads(body)["batch"])
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                b'{"errors": [{"status": 400, "message": "rejected score"}]}'
                if status == 207
                else b"{}"
            )

        def log_message(self, format: str, *args: object) -> None:
            pass

    async def direct(_case: Case, _trial: Trial, trace: TraceRecorder) -> Evidence:
        for tokens in (12, 7):
            trace.record(
                ModelCallFact(
                    call=ModelCall(
                        role="target",
                        operation="reply",
                        requested=ComponentDefinition(kind="test-model", version="1"),
                        provider="local",
                        interface="chat",
                        status="completed",
                        usage=ModelUsage(input_tokens=tokens, output_tokens=3, cost_usd=0.004),
                    )
                )
            )
        return Evidence(checks=(NativeCheck(name="done", passed=True, reason="completed"),))

    campaign = Campaign(
        name="langfuse-retry",
        revision="1",
        cases=(Case(name="case", task="Run", required_checks=("done",)),),
        variants=(
            Variant(
                name="direct",
                harness=FunctionHarness(
                    component=ComponentDefinition(kind="direct", version="1"), function=direct
                ),
            ),
        ),
    )
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-local-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-local-test")
    monkeypatch.setenv("LANGFUSE_PROJECT_ID", "local-project")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "1")
    monkeypatch.setenv("LANGFUSE_FLUSH_INTERVAL", "0.01")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    with ThreadingHTTPServer(("127.0.0.1", 0), Collector) as server:
        monkeypatch.setenv("LANGFUSE_BASE_URL", f"http://127.0.0.1:{server.server_port}")
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            exporter = LangfuseExporter.from_environment()
            report = asyncio.run(run(campaign, directory=tmp_path, exporters=(exporter,)))

            assert report.exit_code == 0
            assert rejected_path in requests
            error_path = next(tmp_path.glob("trials/**/trace-export-errors.json"))
            assert "Langfuse" in error_path.read_text(encoding="utf-8")
            state_path = next(tmp_path.glob("trials/**/trace-exports.json"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            assert state["succeeded"] == []
            requests.clear()
            reject = False

            resumed = asyncio.run(run(campaign, directory=tmp_path, exporters=(exporter,)))

            assert resumed.executed_trials == 0
            assert rejected_path in requests
            retried_state = json.loads(state_path.read_text(encoding="utf-8"))
            assert retried_state["export_id"] == state["export_id"]
            assert len(retried_state["succeeded"]) == 1
            assert not error_path.exists()
            stored_spans = {span.span_id: span for span in delivered_spans}
            generations = [
                span for span in stored_spans.values() if span.name == "model:target:reply"
            ]
            assert len(generations) == 2, "retry appended duplicate model generations"
            assert len({span.trace_id for span in stored_spans.values()}) == 1
            assert sum(span.name == "multiturn-eval-trial" for span in stored_spans.values()) == 1
            roots = {
                span.span_id
                for span in stored_spans.values()
                if span.name == "multiturn-eval-trial"
            }
            assert all(span.parent_span_id in roots for span in generations)
            if rejected_path == "/api/public/ingestion":
                assert len(delivered_spans) == 2 * len(stored_spans)
            assert len({score["id"] for score in delivered_scores}) == 1
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize("changed_destination", ["base-url", "project"])
def test_langfuse_resume_exports_to_each_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_destination: str
) -> None:
    requests: list[str] = []

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, format: str, *args: object) -> None:
            pass

    async def direct(_case: Case, _trial: Trial, _trace: TraceRecorder) -> Evidence:
        return Evidence(checks=(NativeCheck(name="done", passed=True, reason="completed"),))

    campaign = Campaign(
        name="langfuse-destination",
        revision="1",
        cases=(Case(name="case", task="Run", required_checks=("done",)),),
        variants=(
            Variant(
                name="direct",
                harness=FunctionHarness(
                    component=ComponentDefinition(kind="direct", version="1"), function=direct
                ),
            ),
        ),
    )
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-local-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-local-test")
    monkeypatch.setenv("LANGFUSE_PROJECT_ID", "first-project")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    with ThreadingHTTPServer(("127.0.0.1", 0), Collector) as server:
        base_url = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("LANGFUSE_BASE_URL", base_url)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            exporter = LangfuseExporter.from_environment()
            first = asyncio.run(run(campaign, directory=tmp_path, exporters=(exporter,)))
            assert first.exit_code == 0
            assert requests
            requests.clear()
            same = asyncio.run(run(campaign, directory=tmp_path, exporters=(exporter,)))
            assert same.executed_trials == 0
            assert requests == []
            if changed_destination == "base-url":
                monkeypatch.setenv("LANGFUSE_BASE_URL", base_url + "/another-collector")
            else:
                monkeypatch.setenv("LANGFUSE_PROJECT_ID", "second-project")
                monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-second-test")
                monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-second-test")
            destination = LangfuseExporter.from_environment()

            resumed = asyncio.run(run(campaign, directory=tmp_path, exporters=(destination,)))

            assert resumed.executed_trials == 0
            assert requests, "resume skipped a Langfuse destination that has no export yet"
            state_path = next(tmp_path.glob("trials/**/trace-exports.json"))
            state = json.loads(state_path.read_text())
            assert len(state["succeeded"]) == 2
            for credential in (
                "pk-lf-local-test",
                "sk-lf-local-test",
                "pk-lf-second-test",
                "sk-lf-second-test",
            ):
                assert credential not in state_path.read_text()
            requests.clear()
            monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-rotated-test")
            monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-rotated-test")
            rotated = LangfuseExporter.from_environment()
            asyncio.run(run(campaign, directory=tmp_path, exporters=(rotated,)))
            assert requests == [], "rotating credentials changed the destination identity"
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize(
    "base_url",
    [
        "https://user:password@collector.example",
        "https://collector.example?token=secret",
        "https://collector.example#secret",
    ],
)
def test_langfuse_destination_rejects_urls_that_could_persist_credentials(base_url: str) -> None:
    with pytest.raises(ValueError, match="without credentials, query or fragment"):
        LangfuseExporter(inherit_environment=(), base_url=base_url, project_id="project")


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
    }
    assert generation["cost_details"] == {"total": 0.004}
    assert client.flushed is True


def test_langfuse_omits_unknown_tokens_and_retains_known_metrics(tmp_path: Path) -> None:
    key = TrialKey(campaign="trace", revision="1", case="case", variant="direct", sample=1)
    call = ModelCall(
        role="target",
        operation="reply",
        requested=ComponentDefinition(kind="provider-model", version="1"),
        provider="litellm",
        interface="chat",
        status="completed",
        usage=ModelUsage(
            requests=1,
            tool_calls=1,
            input_tokens=None,
            output_tokens=0,
            cache_read_tokens=None,
            cost_usd=0.004,
        ),
    )
    trace = LocalTrace(key, 1, tmp_path / "events.jsonl")
    trace.record(ModelCallFact(call=call))
    trace.close()
    result = TrialResult(
        key=key,
        attempt=1,
        duration_seconds=0,
        terminal=Completed(
            evidence=Evidence(checks=(NativeCheck(name="done", passed=True, reason="completed"),))
        ),
        model_calls=(call,),
    )
    client = _RecordingLangfuseClient()

    export_with_langfuse_client(client, key, trace.events, result, "1" * 32)

    generation = client.observations[1]
    details = generation["usage_details"]
    assert isinstance(details, dict)
    assert "input" not in details
    assert "total" not in details
    assert "cache_read_tokens" not in details
    assert details["output"] == 0
    assert details["requests"] == 1
    assert details["tool_calls"] == 1
    assert generation["cost_details"] == {"total": 0.004}
