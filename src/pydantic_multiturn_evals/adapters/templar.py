"""One-trial HTTP adapter for a Templar workflow service."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator

from pydantic_multiturn_evals.evaluation import HarnessError
from pydantic_multiturn_evals.model import (
    Artifact,
    Case,
    ComponentDefinition,
    Evidence,
    LifecycleFact,
    Name,
    NativeCheck,
    PublicOutcome,
    StrictModel,
    Text,
    TraceRecorder,
    Trial,
)


class TemplarArtifact(StrictModel):
    endpoint: Name
    input_field: Name
    path: Path
    revision: Text


class TemplarTask(StrictModel):
    case: Name
    workflow: Name
    input: JsonValue
    artifact: TemplarArtifact | None = None

    @model_validator(mode="after")
    def require_object_input(self) -> TemplarTask:
        if not isinstance(self.input, dict):
            raise ValueError("Templar task input must be a JSON object")
        return self


class _StoredArtifact(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    artifact_id: str


class _RunView(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True)

    run_id: str = Field(alias="runId")
    status: str
    last_sequence: int = Field(alias="lastSequence")
    applied: bool | None = None


class _Submitted(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    run_id: str
    run: _RunView


class _Evaluation(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    passed: bool


class _TemplarResult(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    run: _RunView
    result: JsonValue
    report: str
    evaluation: _Evaluation


@dataclass(frozen=True, slots=True)
class TemplarHarness:
    endpoint: str
    tasks: tuple[TemplarTask, ...]
    api_version: str = "1"
    token_environment: str | None = None
    check: str = "templar.verified"
    poll_seconds: float = 1
    request_timeout_seconds: float = 30
    response_bytes: int = 8_000_000

    def __post_init__(self) -> None:
        cases = [task.case for task in self.tasks]
        if not self.tasks:
            raise ValueError("TemplarHarness needs at least one task binding")
        if len(cases) != len(set(cases)):
            raise ValueError("Templar task case bindings must be unique")
        if self.poll_seconds <= 0 or self.request_timeout_seconds <= 0:
            raise ValueError("Templar polling and request timeouts must be positive")
        if self.response_bytes < 1:
            raise ValueError("Templar response limit must be positive")

    @property
    def component(self) -> ComponentDefinition:
        return ComponentDefinition(
            kind="templar",
            version="1",
            config={
                "endpoint": self.endpoint,
                "tasks": [_task_identity(task) for task in self.tasks],
                "api_version": self.api_version,
                "token_environment": self.token_environment,
                "check": self.check,
                "poll_seconds": self.poll_seconds,
                "request_timeout_seconds": self.request_timeout_seconds,
                "response_bytes": self.response_bytes,
            },
        )

    async def run(self, case: Case, trial: Trial, trace: TraceRecorder) -> Evidence:
        task = self._task(case.name)
        headers = {"accept": "application/json"}
        if self.token_environment is not None:
            token = os.environ.get(self.token_environment)
            if token is None:
                raise HarnessError(
                    "start",
                    f"environment variable {self.token_environment!r} is not set",
                )
            headers["authorization"] = f"Bearer {token}"
        input_value = dict(task.input) if isinstance(task.input, dict) else {}
        state: dict[str, JsonValue] = {
            "protocol": 1,
            "phase": "preparing",
            "workflow": task.workflow,
        }
        state_path = trial.directory / "templar-state.json"
        _write_json(state_path, state)
        run_id: str | None = None
        async with httpx.AsyncClient(
            base_url=self.endpoint.rstrip("/"),
            headers=headers,
            timeout=self.request_timeout_seconds,
        ) as client:
            try:
                if task.artifact is not None:
                    source = _read_json(task.artifact.path)
                    uploaded = _StoredArtifact.model_validate(
                        await self._json(
                            client,
                            "POST",
                            f"/api/artifacts/{task.artifact.endpoint}",
                            source,
                        )
                    )
                    input_value[task.artifact.input_field] = uploaded.artifact_id
                    state["artifact_id"] = uploaded.artifact_id
                    state["phase"] = "artifact-staged"
                    _write_json(state_path, state)
                submission = await self._json(
                    client,
                    "POST",
                    f"/api/workflows/{task.workflow}/runs",
                    input_value,
                )
                # Cleanup still needs the run id if the accompanying view is malformed.
                identifier = submission.get("run_id") if isinstance(submission, dict) else None
                if isinstance(identifier, str):
                    run_id = identifier
                submitted = _Submitted.model_validate(submission)
                run_id = submitted.run_id
                state["run_id"] = run_id
                state["phase"] = "submitted"
                _write_json(state_path, state)
                final = await self._wait(client, run_id, trial, trace, state, state_path)
                final_path = trial.directory / "templar-result.json"
                report_path = trial.directory / "templar-report.md"
                events_path = trial.directory / "templar-events.jsonl"
                _write_json(final_path, final.model_dump(mode="json"))
                report_path.write_text(final.report, encoding="utf-8")
                if not events_path.exists():
                    events_path.touch()
                state["phase"] = "finished"
                _write_json(state_path, state)
                return Evidence(
                    public=PublicOutcome(output=final.result),
                    checks=(
                        NativeCheck(
                            name=self.check,
                            passed=final.evaluation.passed,
                            reason="Templar accepted result evaluation passed"
                            if final.evaluation.passed
                            else "Templar accepted result evaluation failed",
                        ),
                    ),
                    artifacts=(
                        Artifact(name="templar-result", path=final_path.name),
                        Artifact(
                            name="templar-report", path=report_path.name, media_type="text/markdown"
                        ),
                        Artifact(
                            name="templar-events",
                            path=events_path.name,
                            media_type="application/x-ndjson",
                        ),
                    ),
                )
            except asyncio.CancelledError:
                if run_id is not None:
                    await asyncio.shield(self._cancel(client, run_id, state, state_path))
                raise
            except Exception as error:
                if run_id is not None:
                    await asyncio.shield(self._cancel(client, run_id, state, state_path))
                if isinstance(error, HarnessError):
                    raise
                raise HarnessError("protocol", f"invalid Templar response: {error}") from error

    async def _wait(
        self,
        client: httpx.AsyncClient,
        run_id: str,
        trial: Trial,
        trace: TraceRecorder,
        state: dict[str, JsonValue],
        state_path: Path,
    ) -> _TemplarResult:
        cursor = 0
        events_path = trial.directory / "templar-events.jsonl"
        while True:
            events = await self._json(client, "GET", f"/api/runs/{run_id}/events?after={cursor}")
            if not isinstance(events, list):
                raise HarnessError("protocol", "Templar events response must be a JSON array")
            with events_path.open("a", encoding="utf-8") as stream:
                for event in events:
                    if not isinstance(event, dict):
                        raise HarnessError("protocol", "Templar event must be a JSON object")
                    sequence = event.get("sequence")
                    event_type = event.get("type")
                    if not isinstance(sequence, int) or sequence <= cursor:
                        raise HarnessError("protocol", "Templar event sequence is not increasing")
                    if not isinstance(event_type, str) or not event_type:
                        raise HarnessError("protocol", "Templar event type is missing")
                    cursor = sequence
                    stream.write(json.dumps(event, sort_keys=True) + "\n")
                    trace.record(LifecycleFact(phase="templar-event", detail=event_type))
            view = _RunView.model_validate(await self._json(client, "GET", f"/api/runs/{run_id}"))
            state["last_sequence"] = cursor
            state["status"] = view.status
            _write_json(state_path, state)
            if view.status == "accepted":
                if view.applied is not True:
                    raise HarnessError("verify", "Templar accepted run was not applied")
                if cursor >= view.last_sequence:
                    return _TemplarResult.model_validate(
                        await self._json(client, "GET", f"/api/runs/{run_id}/result")
                    )
            if view.status in {"failed", "interrupted", "stopped", "budget_exhausted"}:
                raise HarnessError("execute", f"Templar run ended with status {view.status!r}")
            await asyncio.sleep(self.poll_seconds)

    async def _cancel(
        self,
        client: httpx.AsyncClient,
        run_id: str,
        state: dict[str, JsonValue],
        state_path: Path,
    ) -> None:
        state["run_id"] = run_id
        try:
            async with asyncio.timeout(self.request_timeout_seconds):
                value = _RunView.model_validate(
                    await self._json(client, "POST", f"/api/runs/{run_id}/cancel", None)
                )
            state["phase"] = "cancelled"
            state["status"] = value.status
        except Exception as error:
            state["phase"] = "cancellation-unconfirmed"
            state["cancel_error"] = str(error) or repr(error)
        try:
            _write_json(state_path, state)
        except OSError:
            # Recording cleanup must not hide the failure that triggered it.
            pass

    async def _json(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        body: JsonValue = None,
    ) -> JsonValue:
        request = client.build_request(method, path, json=body if method == "POST" else None)
        response = await client.send(request, stream=True)
        try:
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > self.response_bytes:
                    raise HarnessError(
                        "protocol",
                        f"Templar response exceeds {self.response_bytes} bytes",
                    )
                chunks.append(chunk)
            raw = b"".join(chunks)
        finally:
            await response.aclose()
        if response.status_code < 200 or response.status_code >= 300:
            detail = raw.decode(errors="replace")[:1000]
            raise HarnessError(
                "execute",
                f"Templar {method} {path} returned {response.status_code}: {detail}",
            )
        try:
            return TypeAdapter(JsonValue).validate_python(json.loads(raw))
        except (json.JSONDecodeError, ValueError) as error:
            message = f"Templar {method} {path} returned invalid JSON"
            raise HarnessError("protocol", message) from error

    def _task(self, case: str) -> TemplarTask:
        for task in self.tasks:
            if task.case == case:
                return task
        raise HarnessError("prepare", f"no Templar task is bound to case {case!r}")


def _read_json(path: Path) -> JsonValue:
    try:
        return TypeAdapter(JsonValue).validate_python(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise HarnessError("prepare", f"could not read Templar artifact {path}: {error}") from error


def _task_identity(task: TemplarTask) -> JsonValue:
    value = task.model_dump(mode="json", exclude={"artifact"})
    if task.artifact is not None:
        value["artifact"] = task.artifact.model_dump(mode="json", exclude={"path"})
    return value


def _write_json(path: Path, value: JsonValue) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
