"""Deterministic multi-turn Harbor agent fixture.

This module runs inside Harbor's own Python environment (no project imports).
It drives a scripted, fully deterministic conversation against a local
OpenAI-compatible HTTP fixture and executes every model-requested action with
the real Harbor ``environment.exec`` API. The trajectory it records is the
truthful ATIF view of what actually happened: real commands, real observation
output, and the usage numbers returned by the fixture server.

Model responses and usage counts are deterministic test fixtures. They say
nothing about model quality and no real model provider is contacted.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories.agent import Agent as TrajectoryAgent
from harbor.models.trajectories.final_metrics import FinalMetrics
from harbor.models.trajectories.metrics import Metrics
from harbor.models.trajectories.observation import Observation, ObservationResult
from harbor.models.trajectories.step import Step
from harbor.models.trajectories.tool_call import ToolCall
from harbor.models.trajectories.trajectory import Trajectory
from harbor.utils.trajectory_utils import format_trajectory_json

AGENT_NAME = "multiturn-fixture-agent"
AGENT_VERSION = "1.0.0"
FIXTURE_LABEL = "deterministic-local-openai-http-fixture"
BEHAVIORS = ("good", "wrong-target")

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command in the task environment and return its exit "
                "code, stdout, and stderr."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }
]


@dataclass(frozen=True, slots=True)
class _RequestedTool:
    call_id: str
    function_name: str
    command: str


@dataclass(frozen=True, slots=True)
class _FixtureReply:
    content: str
    tool_calls: tuple[_RequestedTool, ...]
    response_id: str
    prompt_tokens: int
    completion_tokens: int


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _observation_text(return_code: int, stdout: str | None, stderr: str | None) -> str:
    return f"exit={return_code}\nstdout:\n{stdout or ''}\nstderr:\n{stderr or ''}"


class MultiturnFixtureAgent(BaseAgent):
    """A BaseAgent that performs real model requests and real environment actions."""

    SUPPORTS_ATIF = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        logger: Any = None,
        mcp_servers: list[Any] | None = None,
        *,
        endpoint: str,
        behavior: str = "good",
        request_timeout_seconds: float = 30.0,
        command_timeout_seconds: int = 30,
        max_turns: int = 6,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            logger=logger,
            mcp_servers=mcp_servers,
            **kwargs,
        )
        url = urlsplit(endpoint)
        if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError(f"fixture agent refuses non-local model endpoints: {endpoint!r}")
        if behavior not in BEHAVIORS:
            raise ValueError(f"unknown fixture behavior {behavior!r}; use {BEHAVIORS}")
        if request_timeout_seconds <= 0 or command_timeout_seconds <= 0:
            raise ValueError("fixture timeouts must be positive")
        if max_turns < 2:
            raise ValueError("fixture agent needs at least two model turns")
        self._endpoint = endpoint.rstrip("/")
        self._behavior = behavior
        self._request_timeout_seconds = request_timeout_seconds
        self._command_timeout_seconds = command_timeout_seconds
        self._max_turns = max_turns

    @staticmethod
    def name() -> str:
        return AGENT_NAME

    def version(self) -> str:
        return AGENT_VERSION

    async def setup(self, environment: BaseEnvironment) -> None:
        del environment
        return None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        messages: list[dict[str, Any]] = [{"role": "user", "content": instruction}]
        steps: list[Step] = [Step(step_id=1, timestamp=_now(), source="user", message=instruction)]
        prompt_tokens = 0
        completion_tokens = 0
        requests = 0
        tool_calls = 0
        finished = False
        try:
            for _ in range(self._max_turns):
                reply = await self._chat(messages)
                requests += 1
                prompt_tokens += reply.prompt_tokens
                completion_tokens += reply.completion_tokens
                metrics = Metrics(
                    prompt_tokens=reply.prompt_tokens,
                    completion_tokens=reply.completion_tokens,
                    cost_usd=0.0,
                    extra={
                        "model_fixture": FIXTURE_LABEL,
                        "response_id": reply.response_id,
                    },
                )
                if not reply.tool_calls:
                    messages.append({"role": "assistant", "content": reply.content})
                    steps.append(
                        Step(
                            step_id=len(steps) + 1,
                            timestamp=_now(),
                            source="agent",
                            model_name=self.model_name,
                            message=reply.content,
                            metrics=metrics,
                        )
                    )
                    finished = True
                    break
                assistant: dict[str, Any] = {
                    "role": "assistant",
                    "content": reply.content,
                    "tool_calls": [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.function_name,
                                "arguments": json.dumps({"command": call.command}),
                            },
                        }
                        for call in reply.tool_calls
                    ],
                }
                messages.append(assistant)
                executed: list[tuple[_RequestedTool, str]] = []
                for call in reply.tool_calls:
                    tool_calls += 1
                    result = await environment.exec(
                        command=call.command,
                        timeout_sec=self._command_timeout_seconds,
                    )
                    text = _observation_text(result.return_code, result.stdout, result.stderr)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.call_id,
                            "content": text,
                        }
                    )
                    executed.append((call, text))
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        timestamp=_now(),
                        source="agent",
                        model_name=self.model_name,
                        message=reply.content,
                        tool_calls=[
                            ToolCall(
                                tool_call_id=call.call_id,
                                function_name=call.function_name,
                                arguments={"command": call.command},
                            )
                            for call, _ in executed
                        ],
                        observation=Observation(
                            results=[
                                ObservationResult(
                                    source_call_id=call.call_id,
                                    content=text,
                                )
                                for call, text in executed
                            ]
                        ),
                        metrics=metrics,
                    )
                )
            if not finished:
                raise RuntimeError(
                    f"fixture agent did not finish within {self._max_turns} model turns"
                )
        except Exception as error:
            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    timestamp=_now(),
                    source="system",
                    message=f"fixture agent stopped early: {error}",
                )
            )
            self._record(
                steps=steps,
                context=context,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                requests=requests,
                tool_calls=tool_calls,
                finished=False,
            )
            raise

        self._record(
            steps=steps,
            context=context,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            requests=requests,
            tool_calls=tool_calls,
            finished=True,
        )

    async def _chat(self, messages: list[dict[str, Any]]) -> _FixtureReply:
        payload = {
            "model": self.model_name or "fixture-multiturn",
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
            "tool_choice": "auto",
            "temperature": 0,
        }
        url = f"{self._endpoint}/chat/completions"
        async with httpx.AsyncClient(timeout=self._request_timeout_seconds) as client:
            response = await client.post(url, json=payload)
        if response.status_code != 200:
            raise RuntimeError(
                f"fixture model endpoint {url} returned {response.status_code}: "
                f"{response.text[:500]}"
            )
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
        requested: list[_RequestedTool] = []
        for call in message.get("tool_calls") or []:
            function = call["function"]
            arguments = json.loads(function.get("arguments") or "{}")
            command = arguments.get("command")
            if not isinstance(command, str) or not command:
                raise ValueError(f"fixture tool call {call.get('id')!r} has no shell command")
            requested.append(
                _RequestedTool(
                    call_id=call["id"],
                    function_name=function["name"],
                    command=command,
                )
            )
        usage = data.get("usage") or {}
        return _FixtureReply(
            content=message.get("content") or "",
            tool_calls=tuple(requested),
            response_id=data.get("id") or "",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )

    def _record(
        self,
        *,
        steps: list[Step],
        context: AgentContext,
        prompt_tokens: int,
        completion_tokens: int,
        requests: int,
        tool_calls: int,
        finished: bool,
    ) -> None:
        trajectory = Trajectory(
            session_id=uuid4().hex,
            agent=TrajectoryAgent(
                name=AGENT_NAME,
                version=AGENT_VERSION,
                model_name=self.model_name,
                tool_definitions=TOOL_DEFINITIONS,
                extra={
                    "behavior": self._behavior,
                    "model_fixture": FIXTURE_LABEL,
                    "endpoint": self._endpoint,
                },
            ),
            steps=steps,
            notes=(
                "Integration-verification fixture trajectory. Model responses, tool "
                "choices, and usage numbers come from a deterministic local "
                "OpenAI-compatible HTTP fixture; they are not evidence of model "
                "quality and contact no real provider."
            ),
            final_metrics=FinalMetrics(
                total_prompt_tokens=prompt_tokens,
                total_completion_tokens=completion_tokens,
                total_cost_usd=0.0,
                total_steps=len(steps),
                extra={
                    "model_fixture": FIXTURE_LABEL,
                    "model_requests": requests,
                    "tool_calls": tool_calls,
                    "finished": finished,
                },
            ),
        )
        path = self.logs_dir / "trajectory.json"
        path.write_text(
            format_trajectory_json(trajectory.to_json_dict(exclude_none=True)) + "\n",
            encoding="utf-8",
        )
        context.n_input_tokens = prompt_tokens
        context.n_output_tokens = completion_tokens
        context.cost_usd = 0.0
        context.metadata = {
            "behavior": self._behavior,
            "model_fixture": FIXTURE_LABEL,
            "endpoint": self._endpoint,
            "model_requests": requests,
            "tool_calls": tool_calls,
            "finished": finished,
            "recorded_at": time.time(),
        }
