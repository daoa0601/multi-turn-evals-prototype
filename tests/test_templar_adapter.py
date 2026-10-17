from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

from pydantic_multiturn_evals import Campaign, Case, Completed, Failed, TrialBudget, Variant, run
from pydantic_multiturn_evals.adapters import TemplarArtifact, TemplarHarness, TemplarTask


class _TemplarService(ThreadingHTTPServer):
    requests: list[tuple[str, str, Any]]
    inspections: int
    never_finish: bool
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
            self._send(
                200,
                {"runId": "run-1", "status": "interrupted", "lastSequence": 1},
            )
        else:
            self._send(404, {"error": "not found"})

    def do_GET(self) -> None:  # noqa: N802
        self._service.requests.append(("GET", self.path, None))
        if self.path.startswith("/api/runs/run-1/events"):
            events = (
                [{"sequence": 1, "at": "now", "type": "candidate.completed"}]
                if "after=0" in self.path
                else []
            )
            self._send(200, events)
        elif self.path == "/api/runs/run-1":
            self._service.inspections += 1
            accepted = self._service.inspections > 1 and not self._service.never_finish
            self._send(
                200,
                {
                    "runId": "run-1",
                    "status": "accepted" if accepted else "running",
                    "lastSequence": 1,
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
                        "lastSequence": 1,
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
        self.wfile.write(body)


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


def _service(*, never_finish: bool = False) -> tuple[_TemplarService, threading.Thread]:
    service = _TemplarService(("127.0.0.1", 0), _Handler)
    service.requests = []
    service.inspections = 0
    service.never_finish = never_finish
    service.ready = threading.Event()
    thread = threading.Thread(
        target=service.serve_forever,
        kwargs={"poll_interval": 0.01},
    )
    thread.start()
    if not service.ready.wait(timeout=1):
        raise RuntimeError("Templar test server did not start")
    return service, thread
