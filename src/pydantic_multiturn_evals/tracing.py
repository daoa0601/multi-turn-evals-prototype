"""Local trial traces and optional post-receipt exporters."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import time
from collections.abc import Sequence
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from pydantic_multiturn_evals.harnesses._process import (
    ProcessOutputLimitError,
    collect_process_output,
    inherited_environment,
)
from pydantic_multiturn_evals.model import (
    Completed,
    ComponentDefinition,
    Failed,
    Interrupted,
    ModelCallFact,
    ModelUsage,
    TraceEvent,
    TraceFact,
    TrialKey,
    TrialResult,
)


class TraceExporter(Protocol):
    @property
    def component(self) -> ComponentDefinition: ...

    async def export(
        self,
        key: TrialKey,
        events: Sequence[TraceEvent],
        result: TrialResult,
        export_id: str,
    ) -> None: ...


class LocalTrace:
    """Append-only evidence local to one trial attempt."""

    def __init__(self, key: TrialKey, attempt: int, path: Path) -> None:
        self._key = key
        self._attempt = attempt
        self._started = time.monotonic()
        self._events: list[TraceEvent] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("x", encoding="utf-8")

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def record(self, fact: TraceFact) -> None:
        event = TraceEvent(
            key=self._key,
            attempt=self._attempt,
            sequence=len(self._events) + 1,
            elapsed_seconds=time.monotonic() - self._started,
            recorded_at=datetime.now(UTC).isoformat(),
            fact=fact,
        )
        self._events.append(event)
        self._stream.write(event.model_dump_json() + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


_LANGFUSE_ENVIRONMENT = (
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_HOST",
    "LANGFUSE_TIMEOUT",
    "LANGFUSE_TRACING_ENABLED",
    "LANGFUSE_DEBUG",
    "LANGFUSE_FLUSH_AT",
    "LANGFUSE_FLUSH_INTERVAL",
    "LANGFUSE_TRACING_ENVIRONMENT",
    "LANGFUSE_RELEASE",
    "LANGFUSE_SAMPLE_RATE",
    "LANGFUSE_MEDIA_UPLOAD_THREAD_COUNT",
    "OTEL_SDK_DISABLED",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "SSL_CERT_FILE",
)


class LangfuseExporter:
    """Run the synchronous Langfuse SDK in a killable child process."""

    def __init__(self, *, inherit_environment: tuple[str, ...]) -> None:
        self._inherit_environment = inherit_environment

    @property
    def component(self) -> ComponentDefinition:
        return ComponentDefinition(
            kind="langfuse",
            version="3",
            config={"inherit_environment": list(self._inherit_environment)},
        )

    @classmethod
    def from_environment(cls) -> LangfuseExporter:
        missing = [
            name
            for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
            if not os.environ.get(name)
        ]
        if missing:
            raise ValueError(f"Langfuse export requires {', '.join(missing)}")
        if importlib.util.find_spec("langfuse") is None:
            raise RuntimeError("Langfuse export requires: uv sync --extra tracing")
        inherited = tuple(name for name in _LANGFUSE_ENVIRONMENT if name in os.environ)
        return cls(inherit_environment=inherited)

    async def export(
        self,
        key: TrialKey,
        events: Sequence[TraceEvent],
        result: TrialResult,
        export_id: str,
    ) -> None:
        payload = {
            "key": key.model_dump(mode="json"),
            "events": [event.model_dump(mode="json") for event in events],
            "result": result.model_dump(mode="json"),
            "export_id": export_id,
        }
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pydantic_multiturn_evals.langfuse_worker",
            env=inherited_environment(self._inherit_environment),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            output, error_output = await collect_process_output(
                process,
                (json.dumps(payload) + "\n").encode(),
                stdout_bytes=10_000,
                stderr_bytes=100_000,
            )
        except ProcessOutputLimitError as error:
            raise RuntimeError(f"Langfuse worker {error}") from error
        if process.returncode != 0:
            detail = error_output.decode(errors="replace").strip()
            raise RuntimeError(
                f"Langfuse worker exited with code {process.returncode}"
                + (f": {detail}" if detail else "")
            )
        if output.strip():
            raise RuntimeError("Langfuse worker returned unexpected output")


class _LangfuseObservation(Protocol):
    def start_as_current_observation(
        self,
        *,
        name: str,
        as_type: str,
        input: object | None,
        metadata: dict[str, object],
        model: str | None = None,
        model_parameters: dict[str, object] | None = None,
        usage_details: dict[str, int] | None = None,
        cost_details: dict[str, float] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> AbstractContextManager[_LangfuseObservation]: ...

    def update(self, **values: object) -> None: ...

    def score_trace(self, *, name: str, value: float, comment: str | None = None) -> None: ...


class _LangfuseClient(Protocol):
    def start_as_current_observation(
        self,
        *,
        name: str,
        as_type: str,
        input: object,
        metadata: dict[str, object],
        trace_context: dict[str, str],
        version: str,
    ) -> AbstractContextManager[_LangfuseObservation]: ...

    def flush(self) -> None: ...


def export_with_langfuse_client(
    client_value: object,
    key: TrialKey,
    events: Sequence[TraceEvent],
    result: TrialResult,
    export_id: str,
) -> None:
    client = cast("_LangfuseClient", client_value)
    metadata: dict[str, object] = key.model_dump(mode="json")
    metadata["attempt"] = result.attempt
    with client.start_as_current_observation(
        name="multiturn-eval-trial",
        as_type="evaluator",
        input={"key": metadata},
        metadata=metadata,
        trace_context={"trace_id": export_id},
        version=key.revision,
    ) as root:
        for event in events:
            _export_event(root, event)
        root.update(output=result.model_dump(mode="json"))
        if isinstance(result.terminal, Completed):
            for judgment in result.terminal.judgments:
                root.score_trace(
                    name=judgment.criterion,
                    value=(
                        judgment.score if judgment.score is not None else float(judgment.passed)
                    ),
                    comment=judgment.reason,
                )
            root.score_trace(name="trial_passed", value=float(result.terminal.passed))
        elif isinstance(result.terminal, Failed):
            root.score_trace(
                name="trial_completed",
                value=0.0,
                comment=result.terminal.failure.message,
            )
        elif isinstance(result.terminal, Interrupted):
            root.score_trace(name="trial_completed", value=0.0, comment=result.terminal.reason)
    client.flush()


def _export_event(root: _LangfuseObservation, event: TraceEvent) -> None:
    fact = event.fact.model_dump(mode="json")
    metadata: dict[str, object] = {
        "sequence": event.sequence,
        "elapsed_seconds": event.elapsed_seconds,
    }
    if isinstance(event.fact, ModelCallFact):
        call = event.fact.call
        metadata.update(
            {
                "provider": call.provider,
                "interface": call.interface,
                "requested": call.requested.model_dump(mode="json"),
                "sampling": call.sampling,
                "provider_response_id": call.provider_response_id,
                "duration_seconds": call.duration_seconds,
            }
        )
        with root.start_as_current_observation(
            name=f"model:{call.role}:{call.operation}",
            as_type="generation",
            input=None,
            metadata=metadata,
            model=call.reported_model or call.requested.kind,
            model_parameters=_model_parameters(call.sampling),
            usage_details=_usage_details(call.usage),
            cost_details=_cost_details(call.usage),
            level={"failed": "ERROR", "cancelled": "WARNING"}.get(call.status),
            status_message=call.error_message,
        ) as observation:
            observation.update(output={"status": call.status})
        return
    with root.start_as_current_observation(
        name=f"event:{event.fact.kind}",
        as_type="span",
        input=fact,
        metadata=metadata,
    ) as observation:
        observation.update(output=fact)


def _model_parameters(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    supported: dict[str, object] = {}
    for name, item in value.items():
        if not isinstance(name, str):
            continue
        if item is None or isinstance(item, str | int | float | bool):
            supported[name] = item
        elif isinstance(item, list) and all(isinstance(entry, str) for entry in item):
            supported[name] = item
    return supported or None


def _usage_details(usage: ModelUsage | None) -> dict[str, int] | None:
    if usage is None:
        return None
    details = {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "requests": usage.requests,
        "tool_calls": usage.tool_calls,
        "cache_write_tokens": usage.cache_write_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "input_audio_tokens": usage.input_audio_tokens,
        "cache_audio_read_tokens": usage.cache_audio_read_tokens,
        "output_audio_tokens": usage.output_audio_tokens,
    }
    if usage.input_tokens is not None and usage.output_tokens is not None:
        details["total"] = usage.input_tokens + usage.output_tokens
    details.update({f"provider_{name}": value for name, value in usage.details.items()})
    return {name: value for name, value in details.items() if value is not None}


def _cost_details(usage: ModelUsage | None) -> dict[str, float] | None:
    if usage is None or usage.cost_usd is None:
        return None
    return {"total": usage.cost_usd}


def write_export_errors(path: Path, errors: Sequence[dict[str, str]]) -> None:
    if not errors:
        path.unlink(missing_ok=True)
        return
    path.write_text(json.dumps(list(errors), indent=2) + "\n", encoding="utf-8")
