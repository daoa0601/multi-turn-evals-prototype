"""Run the real tau3-bench mock domain against a deterministic local model server."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from pydantic_multiturn_evals import (
    Campaign,
    Case,
    Completed,
    Report,
    ResourceClaim,
    TrialBudget,
    Variant,
    run,
)
from pydantic_multiturn_evals.adapters import Tau3Participant, Tau3Task, Tau3TextHarness
from scripts.install_tau3 import DEFAULT_CHECKOUT, TAU3_COMMIT, TAU3_VERSION

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "integration" / "tau3_worker.py"


class _ModelServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128
    requests: list[dict[str, Any]]


class _ModelHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        body = self._body()
        self._server.requests.append(body)
        messages = body.get("messages")
        if not isinstance(messages, list):
            self._send(400, {"error": "messages must be a list"})
            return
        system = _message_text(messages[0]) if messages else ""
        conversation = "\n".join(_message_text(message) for message in messages)
        if "<scenario>" in system:
            if "EHGLP3" in system:
                refusals = conversation.count("cannot cancel reservation EHGLP3")
                if refusals == 0:
                    content = "I want to cancel reservation EHGLP3 and receive a refund."
                elif refusals == 1:
                    content = "I was told I did not need insurance on this booking."
                else:
                    content = "###STOP###"
            else:
                content = (
                    "###STOP###"
                    if "Important Meeting task was created" in conversation
                    else "Create a new task called Important Meeting for user_1."
                )
            response = _completion(body, content=content)
        elif "EHGLP3" in conversation or "did not need insurance" in conversation:
            response = _completion(
                body,
                content=(
                    "I cannot cancel reservation EHGLP3 because it is outside the refundable "
                    "window, so I will not proceed."
                ),
            )
        elif any(
            isinstance(message, dict) and message.get("role") == "tool" for message in messages
        ):
            response = _completion(body, content="The Important Meeting task was created.")
        else:
            response = _completion(
                body,
                tool_name="create_task",
                arguments={"user_id": "user_1", "title": "Important Meeting"},
            )
        self._send(200, response)

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    @property
    def _server(self) -> _ModelServer:
        return cast(_ModelServer, self.server)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("request body must be an object")
        return value

    def _send(self, status: int, value: object) -> None:
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _completion(
    request: dict[str, Any],
    *,
    content: str | None = None,
    tool_name: str | None = None,
    arguments: dict[str, str] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    finish_reason = "stop"
    if tool_name is not None:
        message["tool_calls"] = [
            {
                "id": f"call_{uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments or {}),
                },
            }
        ]
        finish_reason = "tool_calls"
    model = request.get("model")
    return {
        "id": f"chatcmpl_{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model if isinstance(model, str) else "local-tau3",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "total_tokens": 25,
        },
    }


def _message_text(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    content = value.get("content")
    return content if isinstance(content, str) else ""


def verify(
    checkout: Path,
    directory: Path,
    *,
    samples: int = 1,
    workers: int = 1,
) -> dict[str, object]:
    if samples < 1 or workers < 1:
        raise ValueError("samples and workers must be positive")
    if (directory / "campaign.json").exists():
        raise ValueError(f"Tau3 verification output already contains a campaign: {directory}")
    checkout = checkout.resolve()
    python = checkout / ".venv" / "bin" / "python"
    if not python.is_file():
        raise ValueError(f"Tau3 is not installed at {checkout}; run make install-tau3")
    server = _ModelServer(("127.0.0.1", 0), _ModelHandler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address)
    endpoint = f"http://{host}:{port}/v1"
    try:
        with _environment("OPENAI_API_KEY", "local-tau3-verification"):
            started = time.monotonic()
            report = asyncio.run(
                _run(checkout, python, directory, endpoint, samples=samples, workers=workers)
            )
            elapsed = time.monotonic() - started
            requests_before_resume = len(server.requests)
            resumed = asyncio.run(
                _run(checkout, python, directory, endpoint, samples=samples, workers=workers)
            )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    expected_trials = samples * 2
    if len(report.results) != expected_trials:
        raise RuntimeError(
            f"Tau3 returned {len(report.results)} trials, expected {expected_trials}"
        )
    failed = [
        result
        for result in report.results
        if not isinstance(result.terminal, Completed) or not result.passed
    ]
    if failed:
        raise RuntimeError(f"Tau3 native verification did not pass: {failed[0].terminal}")
    recorded_calls = sum(len(result.model_calls) for result in report.results)
    if recorded_calls != len(server.requests):
        raise RuntimeError(
            f"Tau3 preserved {recorded_calls} calls from {len(server.requests)} model requests"
        )
    if resumed.executed_trials != 0 or len(server.requests) != requests_before_resume:
        raise RuntimeError("Tau3 resume replayed a completed simulation")
    first = report.results[0]
    assert isinstance(first.terminal, Completed)
    summary: dict[str, object] = {
        "campaign": report.campaign,
        "directory": str(directory.resolve()),
        "package_version": TAU3_VERSION,
        "checkout_revision": TAU3_COMMIT,
        "tasks": ["mock:create_task_1", "airline:0"],
        "domains": ["mock", "airline"],
        "samples": samples,
        "workers": workers,
        "trials": expected_trials,
        "executed_trials": report.executed_trials,
        "resume_executed_trials": resumed.executed_trials,
        "elapsed_seconds": round(elapsed, 6),
        "trials_per_second": round(expected_trials / elapsed, 3),
        "aggregate_trial_seconds": round(
            sum(result.duration_seconds for result in report.results),
            6,
        ),
        "passed": sum(result.passed for result in report.results),
        "native_check": first.terminal.evidence.checks[0].model_dump(mode="json"),
        "public_messages": sum(
            len(result.terminal.evidence.public.messages)
            for result in report.results
            if isinstance(result.terminal, Completed)
        ),
        "model_calls": recorded_calls,
        "local_model_requests": len(server.requests),
        "source_state": _source_state(),
    }
    _write_json(directory / "tau3-verify-summary.json", summary)
    return summary


async def _run(
    checkout: Path,
    python: Path,
    directory: Path,
    endpoint: str,
    *,
    samples: int,
    workers: int,
) -> Report:
    participant_arguments = {"api_base": endpoint, "temperature": 0, "num_retries": 0}
    harness = Tau3TextHarness(
        worker_argv=(str(python), str(WORKER)),
        worker_cwd=checkout,
        package_version=TAU3_VERSION,
        checkout_revision=TAU3_COMMIT,
        tasks=(
            Tau3Task(
                case="create-task",
                domain="mock",
                task="create_task_1",
                revision=TAU3_COMMIT,
            ),
            Tau3Task(
                case="airline-refusal",
                domain="airline",
                task="0",
                revision=TAU3_COMMIT,
            ),
        ),
        agent=Tau3Participant(
            implementation="llm_agent",
            model="openai/local-tau3",
            provider="openai",
            arguments=participant_arguments,
        ),
        user=Tau3Participant(
            implementation="user_simulator",
            model="openai/local-tau3",
            provider="openai",
            arguments=participant_arguments,
        ),
        inherit_environment=("PATH", "HOME", "OPENAI_API_KEY"),
        max_steps=8,
        simulation_timeout_seconds=120,
    )
    campaign = Campaign(
        name="tau3-adapter-smoke",
        revision="1.0.1",
        cases=(
            Case(
                name="create-task",
                task="Run the selected Tau3 task.",
                required_checks=("tau3.reward",),
                budget=TrialBudget(seconds=150),
            ),
            Case(
                name="airline-refusal",
                task="Run the selected Tau3 airline task.",
                required_checks=("tau3.reward",),
                budget=TrialBudget(seconds=150),
            ),
        ),
        variants=(
            Variant(
                name="local-model",
                harness=harness,
                resources=(ResourceClaim(name="tau3"),),
            ),
        ),
        samples=samples,
    )
    return await run(
        campaign,
        directory=directory,
        workers=workers,
        capacities={"tau3": workers},
    )


@contextmanager
def _environment(name: str, value: str) -> Iterator[None]:
    prior = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prior


def _source_state() -> str:
    result = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return "dirty-worktree" if result.stdout else "clean-worktree"


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.directory is None:
        with tempfile.TemporaryDirectory(prefix="mte-tau3-") as temporary:
            summary = verify(
                arguments.checkout,
                Path(temporary),
                samples=arguments.samples,
                workers=arguments.workers,
            )
    else:
        summary = verify(
            arguments.checkout,
            arguments.directory,
            samples=arguments.samples,
            workers=arguments.workers,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
