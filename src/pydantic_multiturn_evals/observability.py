"""Opt-in Langfuse initialization for Pydantic AI OpenTelemetry spans."""

from __future__ import annotations

from typing import Protocol

from pydantic_ai import Agent


class TraceRuntime(Protocol):
    def flush(self) -> None: ...


def enable_langfuse() -> TraceRuntime:
    try:
        from langfuse import get_client
    except ImportError as error:
        raise RuntimeError("Langfuse tracing requires: uv sync --extra tracing") from error
    client = get_client()
    Agent.instrument_all()
    return client
