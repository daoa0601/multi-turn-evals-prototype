"""Small tracing boundary with an optional Langfuse implementation."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic_ai import Agent


@dataclass(frozen=True, slots=True)
class TraceFields:
    suite: str
    target: str | None = None
    harness: Literal["pydantic_ai", "command"] | None = None
    comparison_id: str | None = None
    arm: Literal["baseline", "candidate"] | None = None
    scenario_id: str | None = None
    repeat_index: int | None = None
    run_id: str | None = None
    turn_index: int | None = None
    version: str | None = None

    def metadata(self) -> dict[str, str | int]:
        return {
            key: value
            for key, value in {
                "suite": self.suite,
                "target": self.target,
                "harness": self.harness,
                "comparison_id": self.comparison_id,
                "arm": self.arm,
                "scenario_id": self.scenario_id,
                "repeat_index": self.repeat_index,
                "run_id": self.run_id,
                "turn_index": self.turn_index,
            }.items()
            if value is not None
        }


class TraceSpan(Protocol):
    def child(
        self,
        name: str,
        fields: TraceFields,
        *,
        input: object | None = None,
        as_type: str = "span",
    ) -> AbstractContextManager[TraceSpan]: ...

    def update(self, output: object) -> None: ...

    def score(self, name: str, value: float, reason: str | None = None) -> None: ...


class TraceRuntime(Protocol):
    def span(
        self,
        name: str,
        fields: TraceFields,
        *,
        input: object | None = None,
        as_type: str = "span",
    ) -> AbstractContextManager[TraceSpan]: ...

    def score_run(
        self,
        run_id: str,
        *,
        case_name: str,
        score: float | None,
        assertion: bool | None,
        passed: bool,
        reason: str | None,
    ) -> None: ...

    def shutdown(self) -> None: ...


class NoOpTraceSpan:
    @contextmanager
    def child(
        self,
        name: str,
        fields: TraceFields,
        *,
        input: object | None = None,
        as_type: str = "span",
    ) -> Iterator[TraceSpan]:
        yield self

    def update(self, output: object) -> None:
        return None

    def score(self, name: str, value: float, reason: str | None = None) -> None:
        return None


class NoOpTraceRuntime:
    @contextmanager
    def span(
        self,
        name: str,
        fields: TraceFields,
        *,
        input: object | None = None,
        as_type: str = "span",
    ) -> Iterator[TraceSpan]:
        yield NoOpTraceSpan()

    def score_run(
        self,
        run_id: str,
        *,
        case_name: str,
        score: float | None,
        assertion: bool | None,
        passed: bool,
        reason: str | None,
    ) -> None:
        return None

    def shutdown(self) -> None:
        return None


NO_TRACE = NoOpTraceRuntime()


class _LangfuseSpan:
    def __init__(self, observation: Any) -> None:
        self._observation = observation

    @contextmanager
    def child(
        self,
        name: str,
        fields: TraceFields,
        *,
        input: object | None = None,
        as_type: str = "span",
    ) -> Iterator[TraceSpan]:
        with self._observation.start_as_current_observation(
            name=name,
            as_type=as_type,
            input=input,
            metadata=fields.metadata(),
            version=fields.version,
        ) as child:
            yield _LangfuseSpan(child)

    def update(self, output: object) -> None:
        self._observation.update(output=output)

    def score(self, name: str, value: float, reason: str | None = None) -> None:
        self._observation.score_trace(name=name, value=value, comment=reason)


class LangfuseTraceRuntime:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._runs: dict[str, Any] = {}

    @contextmanager
    def span(
        self,
        name: str,
        fields: TraceFields,
        *,
        input: object | None = None,
        as_type: str = "span",
    ) -> Iterator[TraceSpan]:
        from langfuse import propagate_attributes
        from opentelemetry.context import Context, attach, detach

        metadata = fields.metadata()
        tags = ["multiturn-eval"]
        if fields.arm is not None:
            tags.append(f"variant:{fields.arm}")
        context_token = attach(Context())
        try:
            with ExitStack() as stack:
                stack.enter_context(
                    propagate_attributes(
                        session_id=fields.run_id,
                        trace_name=name,
                        tags=tags,
                        version=fields.version,
                        metadata=metadata,
                    )
                )
                observation = stack.enter_context(
                    self._client.start_as_current_observation(
                        name=name,
                        as_type=as_type,
                        input=input,
                        metadata=metadata,
                        version=fields.version,
                    )
                )
                if fields.run_id is not None and fields.scenario_id is not None:
                    self._runs[fields.run_id] = observation
                yield _LangfuseSpan(observation)
        finally:
            detach(context_token)

    def score_run(
        self,
        run_id: str,
        *,
        case_name: str,
        score: float | None,
        assertion: bool | None,
        passed: bool,
        reason: str | None,
    ) -> None:
        observation = self._runs.pop(run_id, None)
        if observation is None:
            return
        observation.update(metadata={"case_name": case_name})
        if score is not None:
            observation.score_trace(name="judge_score", value=score, comment=reason)
        if assertion is not None:
            observation.score_trace(name="judge_pass", value=float(assertion), comment=reason)
        observation.score_trace(name="case_gate_pass", value=float(passed))

    def shutdown(self) -> None:
        self._client.shutdown()


def enable_langfuse() -> TraceRuntime:
    missing = [
        name for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY") if not os.environ.get(name)
    ]
    if missing:
        raise ValueError(f"Langfuse tracing requires {', '.join(missing)}")
    try:
        from langfuse import get_client
    except ImportError as error:
        raise RuntimeError("Langfuse tracing requires: uv sync --extra tracing") from error
    client = get_client()
    Agent.instrument_all()
    return LangfuseTraceRuntime(client)
