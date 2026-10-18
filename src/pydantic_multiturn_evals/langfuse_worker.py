"""Isolated process entry point for the synchronous Langfuse SDK."""

from __future__ import annotations

import json
import os
import sys
from base64 import b64encode
from collections.abc import Sequence
from typing import Any

import httpx
from langfuse import Langfuse
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

from pydantic_multiturn_evals.model import TraceEvent, TrialKey, TrialResult
from pydantic_multiturn_evals.tracing import (
    LANGFUSE_OBSERVATION_ID,
    export_with_langfuse_client,
)


def main() -> int:
    payload = json.loads(sys.stdin.readline())
    public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
    secret_key = os.environ["LANGFUSE_SECRET_KEY"]
    base_url = payload["base_url"]
    authorization = b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii")
    timeout = int(os.environ.get("LANGFUSE_TIMEOUT", "5"))
    spans = _CheckedSpanExporter(
        endpoint=f"{base_url}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {authorization}", "x-langfuse-public-key": public_key},
        timeout=timeout,
    )
    with _CheckedScoreClient(timeout=timeout) as scores:
        client = Langfuse(
            base_url=base_url,
            span_exporter=spans,
            httpx_client=scores,
            id_generator=_ObservationIdGenerator(),
        )
        try:
            export_with_langfuse_client(
                client,
                TrialKey.model_validate(payload["key"]),
                tuple(TraceEvent.model_validate(event) for event in payload["events"]),
                TrialResult.model_validate(payload["result"]),
                payload["export_id"],
            )
        finally:
            client.shutdown()
        failures = spans.failures + scores.failures
        if failures:
            raise RuntimeError("Langfuse delivery failed; export remains pending") from failures[0]
    return 0


class _ObservationIdGenerator(RandomIdGenerator):
    def generate_span_id(self) -> int:
        identifier = LANGFUSE_OBSERVATION_ID.get()
        return identifier if identifier is not None else super().generate_span_id()


class _CheckedSpanExporter(OTLPSpanExporter):
    def __init__(self, *, endpoint: str, headers: dict[str, str], timeout: int) -> None:
        super().__init__(endpoint=endpoint, headers=headers, timeout=timeout)
        # The SDK flushes background queues without propagating their delivery failures.
        self.failures: list[Exception] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            result = super().export(spans)
        except Exception as error:
            self.failures.append(error)
            raise
        if result != SpanExportResult.SUCCESS:
            self.failures.append(RuntimeError("Langfuse span collector rejected delivery"))
        return result


class _CheckedScoreClient(httpx.Client):
    def __init__(self, *, timeout: int) -> None:
        super().__init__(timeout=timeout)
        self.failures: list[Exception] = []

    def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        try:
            response = super().send(request, **kwargs)
        except httpx.HTTPError as error:
            self.failures.append(error)
            raise
        if response.is_error:
            self.failures.append(
                RuntimeError(f"Langfuse score collector returned HTTP {response.status_code}")
            )
        elif response.status_code == 207:
            response.read()
            try:
                rejected = response.json().get("errors")
            except ValueError as error:
                self.failures.append(error)
                raise
            if rejected:
                self.failures.append(RuntimeError("Langfuse score collector rejected batch items"))
        return response


if __name__ == "__main__":
    raise SystemExit(main())
