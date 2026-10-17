"""Deterministic local OpenAI-compatible model fixture.

Stdlib only, so both the project environment and Harbor's own Python can run
it. The server implements a fixed multi-turn tool protocol:

1. read the deployment seed and the container hostname,
2. store ``<seed>-<hostname>`` in the answer file,
3. read the stored value back,
4. reply with a fixed completion message.

The ``wrong-target`` behavior performs the same conversational dance but
stores the answer in the wrong directory, which the native task verifier
fails regardless of how the transcript looks to an evaluator.

Every response, tool choice, and usage number is a deterministic test
fixture. Nothing here measures model quality or contacts a real provider.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

FIXTURE_LABEL = "deterministic-local-openai-http-fixture"
BEHAVIORS = ("good", "wrong-target")
SEED_FILE = "/app/state/seed.txt"
GOOD_TARGET = "/app/final/answer.txt"
WRONG_TARGET = "/app/state/answer.txt"
READ_COMMAND = f"cat {SEED_FILE} && uname -n"

SEED_VALUE = re.compile(r"anchor-[0-9a-f]+")
DERIVED_VALUE = re.compile(r"anchor-[0-9a-f]+-[A-Za-z0-9][A-Za-z0-9._-]*")
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_MAX_TOOL_RESULTS = 3


class FixtureProtocolError(RuntimeError):
    """The conversation does not match the fixture protocol."""


def parse_seed_and_host(text: str) -> tuple[str, str]:
    """Extract the seed and hostname the agent observed in the environment."""

    match = re.search(r"(anchor-[0-9a-f]+)[\s\n]+([A-Za-z0-9][A-Za-z0-9._-]*)", text)
    if match is None:
        raise FixtureProtocolError(f"no seed and hostname observation in {text[:120]!r}")
    seed, host = match.group(1), match.group(2)
    if not _SAFE_TOKEN.match(seed) or not _SAFE_TOKEN.match(host):
        raise FixtureProtocolError(f"unsafe environment tokens: {seed!r} {host!r}")
    return seed, host


def tool_result_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "tool"
    ]


def target_path(behavior: str) -> str:
    if behavior not in BEHAVIORS:
        raise FixtureProtocolError(f"unknown behavior {behavior!r}")
    return GOOD_TARGET if behavior == "good" else WRONG_TARGET


def fixture_reply(
    behavior: str,
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    request_index: int = 1,
) -> dict[str, Any]:
    """Return the deterministic chat completion for one request."""

    results = tool_result_messages(messages)
    prompt_tokens = sum(
        len(str(message.get("content") or "")) + len(str(message.get("tool_call_id") or ""))
        for message in messages
    )
    if not results:
        command = READ_COMMAND
        return _tool_completion(
            behavior,
            model,
            request_index,
            {"command": command},
            prompt_tokens=prompt_tokens,
        )
    seed, host = parse_seed_and_host(str(results[0].get("content") or ""))
    derived = f"{seed}-{host}"
    target = target_path(behavior)
    if len(results) == 1:
        command = f"printf '%s-%s\\n' {seed} {host} > {target}"
        return _tool_completion(
            behavior,
            model,
            request_index,
            {"command": command},
            prompt_tokens=prompt_tokens,
        )
    if len(results) == 2:
        return _tool_completion(
            behavior,
            model,
            request_index,
            {"command": f"cat {target}"},
            prompt_tokens=prompt_tokens,
        )
    if len(results) == _MAX_TOOL_RESULTS:
        content = (
            f"Stored and verified the derived answer {derived} in the answer file. harbor-ready"
        )
        return _text_completion(
            behavior,
            model,
            request_index,
            content,
            prompt_tokens=prompt_tokens,
        )
    raise FixtureProtocolError(
        f"fixture protocol allows at most {_MAX_TOOL_RESULTS} tool results, saw {len(results)}"
    )


class FixtureModelServer(ThreadingHTTPServer):
    """HTTP server exposing /v1/chat/completions for one behavior."""

    daemon_threads = True
    request_queue_size = 64

    def __init__(self, address: tuple[str, int], behavior: str) -> None:
        super().__init__(address, _FixtureHandler)
        if behavior not in BEHAVIORS:
            raise ValueError(f"unknown fixture behavior {behavior!r}")
        self.behavior = behavior
        self.requests: list[dict[str, Any]] = []
        self.replies: list[dict[str, Any]] = []
        self.failures: list[str] = []

    @property
    def endpoint(self) -> str:
        host, port = cast(tuple[str, int], self.server_address)
        return f"http://{host}:{port}/v1"

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, daemon=True)
        thread.start()
        return thread


class _FixtureHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        server = cast(FixtureModelServer, self.server)
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": {"message": f"unknown path {self.path!r}"}})
            return
        try:
            body = self._body()
        except Exception as error:  # malformed request bodies must be visible
            server.failures.append(f"invalid request body: {error}")
            self._send(400, {"error": {"message": str(error)}})
            return
        server.requests.append(body)
        try:
            reply = fixture_reply(
                server.behavior,
                list(body.get("messages") or []),
                model=body.get("model") if isinstance(body.get("model"), str) else None,
                request_index=len(server.requests),
            )
        except Exception as error:
            server.failures.append(f"reply construction failed: {error}")
            self._send(500, {"error": {"message": str(error)}})
            return
        server.replies.append(reply)
        self._send(200, reply)

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("request body must be an object")
        messages = value.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("request must carry a non-empty messages list")
        return value

    def _send(self, status: int, value: object) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _tool_completion(
    behavior: str,
    model: str | None,
    index: int,
    arguments: dict[str, str],
    *,
    prompt_tokens: int,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call-fixture-{index:03d}",
                "type": "function",
                "function": {
                    "name": "run_command",
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }
    return _completion(
        behavior,
        model,
        index,
        message,
        "tool_calls",
        prompt_tokens=prompt_tokens,
        completion_tokens=17 + len(arguments.get("command") or ""),
    )


def _text_completion(
    behavior: str,
    model: str | None,
    index: int,
    content: str,
    *,
    prompt_tokens: int,
) -> dict[str, Any]:
    return _completion(
        behavior,
        model,
        index,
        {"role": "assistant", "content": content},
        "stop",
        prompt_tokens=prompt_tokens,
        completion_tokens=len(content),
    )


def _completion(
    behavior: str,
    model: str | None,
    index: int,
    message: dict[str, Any],
    finish_reason: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-fixture-{behavior}-{index:03d}",
        "object": "chat.completion",
        "created": 0,
        "model": model or f"fixture-multiturn-{behavior}",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens + 11,
            "completion_tokens": completion_tokens + 3,
            "total_tokens": prompt_tokens + completion_tokens + 14,
        },
        "system_fingerprint": FIXTURE_LABEL,
    }
