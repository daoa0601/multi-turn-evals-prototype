from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest

from pydantic_multiturn_evals import Campaign, Case, Completed, Failed, TrialBudget, Variant, run
from pydantic_multiturn_evals.adapters import TemplarArtifact, TemplarHarness, TemplarTask
from pydantic_multiturn_evals.evaluation import HarnessError
from pydantic_multiturn_evals.model import LifecycleFact, TraceEvent, Trial, TrialKey
from pydantic_multiturn_evals.tracing import LocalTrace


class _TemplarService(ThreadingHTTPServer):
    requests: list[tuple[str, str, Any]]
    inspections: int
    never_finish: bool
    events: list[dict[str, Any]]
    publish_on_accept: bool
    status_override: tuple[int, Any] | None
    submission_override: Any
    cancel_status: int
    cancel_started: threading.Event
    cancel_release: threading.Event
    ready: threading.Event

    def service_actions(self) -> None:
        self.ready.set()


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        body = self._body()
        self._service.requests.append(("POST", self.path, body))
        if self.path == "/api/artifacts/source-snapshot":
            self._send(201, {"artifact_id": "source_sha256_123"})
        elif self.path == "/api/workflows/source_security_audit/runs":
            if self._service.submission_override is not None:
                self._send(202, self._service.submission_override)
                return
            self._send(
                202,
                {
                    "run_id": "run-1",
                    "run": {
                        "runId": "run-1",
                        "status": "running",
                        "lastSequence": 0,
                    },
                },
            )
        elif self.path == "/api/runs/run-1/cancel":
            self._service.cancel_started.set()
            self._service.cancel_release.wait(timeout=5)
            self._send(
                self._service.cancel_status,
                {"runId": "run-1", "status": "interrupted", "lastSequence": 1},
            )
        else:
            self._send(404, {"error": "not found"})

    def do_GET(self) -> None:  # noqa: N802
        self._service.requests.append(("GET", self.path, None))
        if self.path.startswith("/api/runs/run-1/events"):
            cursor = int(self.path.split("after=")[1])
            events = self._service.events
            if self._service.publish_on_accept and self._service.inspections < 2:
                events = events[:1]
            self._send(200, [event for event in events if event["sequence"] > cursor][:1])
        elif self.path == "/api/runs/run-1":
            if self._service.status_override is not None:
                self._send(*self._service.status_override)
                return
            self._service.inspections += 1
            accepted = self._service.inspections > 1 and not self._service.never_finish
            self._send(
                200,
                {
                    "runId": "run-1",
                    "status": "accepted" if accepted else "running",
                    "lastSequence": len(self._service.events),
                    "applied": True if accepted else None,
                },
            )
        elif self.path == "/api/runs/run-1/result":
            self._send(
                200,
                {
                    "run": {
                        "runId": "run-1",
                        "status": "accepted",
                        "lastSequence": len(self._service.events),
                        "applied": True,
                    },
                    "result": {"finding": "bounded"},
                    "report": "# Report\n",
                    "evaluation": {"passed": True},
                },
            )
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, format: str, *args: object) -> None:
        pass

    @property
    def _service(self) -> _TemplarService:
        return cast(_TemplarService, self.server)

    def _body(self) -> Any:
        length = int(self.headers.get("content-length", "0"))
        return json.loads(self.rfile.read(length)) if length else None

    def _send(self, status: int, value: Any) -> None:
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            # Deadline tests can close the connection before the service responds.
            pass


def test_templar_adapter_stages_polls_and_uses_the_accepted_evaluation(
    tmp_path: Path,
) -> None:
    service, thread = _service()
    try:
        snapshot = tmp_path / "source.json"
        snapshot.write_text('{"schema_version":"1","files":[]}', encoding="utf-8")
        report = asyncio.run(
            run(
                _campaign(service, snapshot),
                directory=tmp_path / "run",
            )
        )
    finally:
        service.shutdown()
        thread.join()

    terminal = report.results[0].terminal
    assert isinstance(terminal, Completed)
    assert terminal.evidence.checks[0].passed is True
    submission = next(
        body
        for method, path, body in service.requests
        if method == "POST" and path.endswith("/runs")
    )
    assert submission == {"schema_version": "1", "source_snapshot_id": "source_sha256_123"}
    event_file = next((tmp_path / "run").glob("trials/**/templar-events.jsonl"))
    assert len(event_file.read_text(encoding="utf-8").splitlines()) == 1


def test_templar_adapter_cancels_a_remote_run_when_the_case_times_out(tmp_path: Path) -> None:
    service, thread = _service(never_finish=True)
    try:
        snapshot = tmp_path / "source.json"
        snapshot.write_text('{"schema_version":"1","files":[]}', encoding="utf-8")
        campaign = _campaign(service, snapshot, budget=TrialBudget(seconds=0.05))
        report = asyncio.run(run(campaign, directory=tmp_path / "run"))
    finally:
        service.shutdown()
        thread.join()

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert terminal.failure.phase == "timeout"
    assert any(path == "/api/runs/run-1/cancel" for _method, path, _body in service.requests)
    state = json.loads(next((tmp_path / "run").glob("trials/**/templar-state.json")).read_text())
    assert state["phase"] == "cancelled"


@pytest.mark.parametrize("trigger", ["cancellation", "protocol-error"])
@pytest.mark.parametrize("cancel_response", [True, False], ids=["confirmed", "timed-out"])
def test_templar_waits_for_remote_cleanup_through_additional_cancellations(
    tmp_path: Path, trigger: str, cancel_response: bool
) -> None:
    service, thread = _service(never_finish=True)
    service.cancel_release.clear()
    if trigger == "protocol-error":
        service.status_override = (200, {"status": "running"})
    harness = TemplarHarness(
        endpoint=f"http://127.0.0.1:{service.server_port}",
        tasks=(TemplarTask(case="audit", workflow="source_security_audit", input={}),),
        poll_seconds=0.01,
        request_timeout_seconds=0.2,
    )
    key = TrialKey(campaign="cleanup", revision="1", case="audit", variant="only", sample=1)
    trace = LocalTrace(key, 1, tmp_path / "events.jsonl")

    async def exercise() -> None:
        trial = Trial(key=key, attempt=1, seed=1, directory=tmp_path)
        case = Case(name="audit", task="Audit", required_checks=("templar.verified",))
        task = asyncio.create_task(harness.run(case, trial, trace))
        try:
            async with asyncio.timeout(3):
                while service.inspections == 0 and not service.cancel_started.is_set():
                    await asyncio.sleep(0.01)
                if trigger == "cancellation":
                    task.cancel()
                while not service.cancel_started.is_set():
                    await asyncio.sleep(0.01)
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done(), "Templar released its HTTP client during remote cleanup"
            if cancel_response:
                service.cancel_release.set()
            expected = asyncio.CancelledError if trigger == "cancellation" else HarnessError
            async with asyncio.timeout(3):
                with pytest.raises(expected):
                    await task
            state = json.loads((tmp_path / "templar-state.json").read_text())
            if cancel_response:
                assert state["phase"] == "cancelled", state
                assert state["status"] == "interrupted"
            else:
                assert state["phase"] == "cancellation-unconfirmed", state
                assert state["cancel_error"]
        finally:
            service.cancel_release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    try:
        asyncio.run(exercise())
    finally:
        trace.close()
        service.shutdown()
        thread.join()


@pytest.mark.parametrize("publish_on_accept", [False, True], ids=["paginated", "late-events"])
def test_templar_adapter_drains_events_before_accepting(
    tmp_path: Path, publish_on_accept: bool
) -> None:
    service, thread = _service(publish_on_accept=publish_on_accept)
    service.events.extend(
        [
            {"sequence": 2, "type": "evaluation.completed"},
            {"sequence": 3, "type": "result.applied"},
            {"sequence": 4, "type": "run.accepted"},
        ]
    )
    try:
        snapshot = tmp_path / "source.json"
        snapshot.write_text("{}", encoding="utf-8")
        report = asyncio.run(run(_campaign(service, snapshot), directory=tmp_path / "run"))
    finally:
        service.shutdown()
        thread.join()

    assert isinstance(report.results[0].terminal, Completed)
    event_file = next((tmp_path / "run").glob("trials/**/templar-events.jsonl"))
    events = [json.loads(line) for line in event_file.read_text().splitlines()]
    assert [event["type"] for event in events] == [event["type"] for event in service.events]
    trace_file = next((tmp_path / "run").glob("trials/**/events.jsonl"))
    facts = [
        TraceEvent.model_validate_json(line).fact for line in trace_file.read_text().splitlines()
    ]
    assert [
        fact.detail
        for fact in facts
        if isinstance(fact, LifecycleFact) and fact.phase == "templar-event"
    ] == [event["type"] for event in service.events]


@pytest.mark.parametrize("cancel_status", [200, 502], ids=["cancelled", "cancel-failed"])
@pytest.mark.parametrize(
    ("status", "response", "phase", "message"),
    [
        (503, {"error": "service unavailable"}, "execute", "returned 503"),
        (200, {"status": "running"}, "protocol", "invalid Templar response"),
    ],
    ids=["http-error", "malformed-response"],
)
def test_templar_adapter_cancels_after_submission_errors_without_masking_them(
    tmp_path: Path, cancel_status: int, status: int, response: Any, phase: str, message: str
) -> None:
    service, thread = _service()
    service.status_override = (status, response)
    service.cancel_status = cancel_status
    try:
        snapshot = tmp_path / "source.json"
        snapshot.write_text("{}", encoding="utf-8")
        report = asyncio.run(run(_campaign(service, snapshot), directory=tmp_path / "run"))
    finally:
        service.shutdown()
        thread.join()

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert terminal.failure.phase == phase
    assert message in terminal.failure.message
    assert ("POST", "/api/runs/run-1/cancel", None) in service.requests
    state = json.loads(next((tmp_path / "run").glob("trials/**/templar-state.json")).read_text())
    assert state["phase"] == ("cancelled" if cancel_status == 200 else "cancellation-unconfirmed")


def test_templar_cancels_a_known_run_even_when_the_submission_view_is_malformed(
    tmp_path: Path,
) -> None:
    service, thread = _service()
    service.submission_override = {"run_id": "run-1", "run": {"status": "running"}}
    try:
        snapshot = tmp_path / "source.json"
        snapshot.write_text("{}", encoding="utf-8")
        report = asyncio.run(run(_campaign(service, snapshot), directory=tmp_path / "run"))
    finally:
        service.shutdown()
        thread.join()

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert terminal.failure.phase == "protocol"
    assert ("POST", "/api/runs/run-1/cancel", None) in service.requests


def test_templar_event_drain_obeys_the_case_deadline(tmp_path: Path) -> None:
    service, thread = _service()
    service.status_override = (
        200,
        {"runId": "run-1", "status": "accepted", "lastSequence": 2, "applied": True},
    )
    try:
        snapshot = tmp_path / "source.json"
        snapshot.write_text("{}", encoding="utf-8")
        campaign = _campaign(service, snapshot, budget=TrialBudget(seconds=0.05))
        report = asyncio.run(run(campaign, directory=tmp_path / "run"))
    finally:
        service.shutdown()
        thread.join()

    terminal = report.results[0].terminal
    assert isinstance(terminal, Failed)
    assert terminal.failure.phase == "timeout"
    assert ("POST", "/api/runs/run-1/cancel", None) in service.requests


def _campaign(
    service: _TemplarService,
    snapshot: Path,
    *,
    budget: TrialBudget | None = None,
) -> Campaign:
    host, port = cast(tuple[str, int], service.server_address)
    task = TemplarTask(
        case="audit",
        workflow="source_security_audit",
        input={"schema_version": "1"},
        artifact=TemplarArtifact(
            endpoint="source-snapshot",
            input_field="source_snapshot_id",
            path=snapshot,
            revision="fixture-1",
        ),
    )
    return Campaign(
        name="templar-boundary",
        revision="1",
        cases=(
            Case(
                name="audit",
                task="Audit the source",
                data={"secret": "must not be submitted"},
                required_checks=("templar.verified",),
                budget=budget or TrialBudget(seconds=2),
            ),
        ),
        variants=(
            Variant(
                name="only",
                harness=TemplarHarness(
                    endpoint=f"http://{host}:{port}",
                    tasks=(task,),
                    poll_seconds=0.01,
                ),
            ),
        ),
    )


def _service(
    *, never_finish: bool = False, publish_on_accept: bool = False
) -> tuple[_TemplarService, threading.Thread]:
    service = _TemplarService(("127.0.0.1", 0), _Handler)
    service.requests = []
    service.inspections = 0
    service.never_finish = never_finish
    service.events = [{"sequence": 1, "at": "now", "type": "candidate.completed"}]
    service.publish_on_accept = publish_on_accept
    service.status_override = None
    service.submission_override = None
    service.cancel_status = 200
    service.cancel_started = threading.Event()
    service.cancel_release = threading.Event()
    service.cancel_release.set()
    service.ready = threading.Event()
    thread = threading.Thread(
        target=service.serve_forever,
        kwargs={"poll_interval": 0.01},
    )
    thread.start()
    if not service.ready.wait(timeout=1):
        raise RuntimeError("Templar test server did not start")
    return service, thread
