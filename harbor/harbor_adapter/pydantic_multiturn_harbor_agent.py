"""Dependency-light Harbor user agent for pydantic-multiturn-evals."""

from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

CASE_START = "<pydantic-multiturn-eval-case>"
CASE_END = "</pydantic-multiturn-eval-case>"
MAX_ACTOR_RESPONSE_BYTES = 1024 * 1024
MAX_ACTOR_REQUEST_BYTES = 4 * 1024 * 1024


class PydanticAdaptiveUserAgent(BaseAgent):
    """Drive one ACP target and delegate private actor decisions to the evaluator."""

    def __init__(
        self,
        logs_dir: Path,
        *args: object,
        actor_command: list[str],
        actor_timeout_seconds: float = 180,
        **kwargs: object,
    ) -> None:
        if not actor_command:
            raise ValueError("actor_command cannot be empty")
        self._actor_command = tuple(actor_command)
        self._actor_timeout_seconds = actor_timeout_seconds
        super().__init__(logs_dir, *args, **kwargs)

    @staticmethod
    def name() -> str:
        return "pydantic-multiturn-user"

    def version(self) -> str:
        return "1"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        case = parse_case(instruction)
        limits = required_mapping(case, "limits")
        timeout_seconds = required_positive_number(limits, "timeout_seconds")
        try:
            await asyncio.wait_for(
                self._run_case(case, environment, context),
                timeout=timeout_seconds,
            )
        except TimeoutError as error:
            scenario = required_mapping(case, "scenario")
            scenario_id = required_string(scenario, "id")
            raise RuntimeError(
                f"scenario {scenario_id!r} exceeded {timeout_seconds:g} seconds"
            ) from error

    async def _run_case(
        self,
        case: dict[str, object],
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        scenario = required_mapping(case, "scenario")
        limits = required_mapping(case, "limits")
        exchanges: list[dict[str, object]] = []
        decisions: list[dict[str, object]] = []
        user_text = required_string(scenario, "first_prompt")
        max_turns = required_positive_integer(limits, "max_target_turns")

        for _turn_index in range(1, max_turns + 1):
            response = await environment.exec(command=f"acpx prompt {shlex.quote(user_text)}")
            if response.return_code != 0:
                raise RuntimeError(f"ACP target turn exited with status {response.return_code}")
            assistant_text = (response.stdout or "").strip()
            if not assistant_text:
                raise RuntimeError("ACP target returned an empty assistant response")
            exchanges.append(
                {
                    "user": {"role": "user", "content": user_text},
                    "assistant": {"role": "assistant", "content": assistant_text},
                }
            )
            remaining = max_turns - len(exchanges)
            decision = await self._decide(case, exchanges, remaining)
            decisions.append(decision)
            kind = decision.get("kind")
            if kind == "continue" and remaining > 0:
                user_text = required_string(decision, "next_user_message")
                continue
            if kind == "accept":
                termination = {
                    "kind": "actor_accepted",
                    "reason": required_string(decision, "reason"),
                }
            elif kind == "stop":
                termination = {
                    "kind": "actor_stopped",
                    "reason": required_string(decision, "reason"),
                }
            else:
                termination = {"kind": "turn_limit_reached", "limit": max_turns}
            key = required_mapping(case, "key")
            result = {
                "run_id": required_string(case, "run_id"),
                "scenario_id": required_string(key, "scenario_id"),
                "repeat_index": required_positive_integer(key, "repeat_index"),
                "transcript": {"exchanges": exchanges},
                "target_evidence": [],
                "decisions": decisions,
                "termination": termination,
                "completion": {"environment": None, "details": {}},
            }
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            path = self.logs_dir / "multiturn-result.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
            context.metadata = {
                "scenario_id": result["scenario_id"],
                "repeat_index": result["repeat_index"],
                "target_turns": len(exchanges),
                "termination": termination["kind"],
            }
            return

        raise RuntimeError("adaptive conversation exhausted without a terminal result")

    async def _decide(
        self,
        case: dict[str, object],
        exchanges: list[dict[str, object]],
        remaining_target_turns: int,
    ) -> dict[str, object]:
        scenario = required_mapping(case, "scenario")
        request = {
            "actor": required_mapping(case, "actor"),
            "scenario_id": required_string(scenario, "id"),
            "brief": required_mapping(scenario, "actor"),
            "exchanges": exchanges,
            "remaining_target_turns": remaining_target_turns,
        }
        request_data = json.dumps(request).encode()
        if len(request_data) > MAX_ACTOR_REQUEST_BYTES:
            raise RuntimeError("adaptive actor request exceeded its byte limit")
        process = await asyncio.create_subprocess_exec(
            *self._actor_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise RuntimeError("adaptive actor subprocess pipes are unavailable")
        process.stdin.write(request_data)
        await process.stdin.drain()
        process.stdin.close()
        stdout_task = asyncio.create_task(read_bounded(process.stdout, MAX_ACTOR_RESPONSE_BYTES))
        stderr_task = asyncio.create_task(read_bounded(process.stderr, MAX_ACTOR_RESPONSE_BYTES))
        try:
            await asyncio.wait_for(process.wait(), timeout=self._actor_timeout_seconds)
        except TimeoutError as error:
            process.kill()
            await process.wait()
            await stdout_task
            await stderr_task
            raise RuntimeError(
                f"adaptive actor exceeded {self._actor_timeout_seconds:g} seconds"
            ) from error
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            await stdout_task
            await stderr_task
            raise
        stdout, stdout_truncated = await stdout_task
        stderr, _stderr_truncated = await stderr_task
        if process.returncode != 0:
            raise RuntimeError(
                f"adaptive actor exited with status {process.returncode}; "
                f"captured {len(stderr)} stderr byte(s)"
            )
        if stdout_truncated:
            raise RuntimeError("adaptive actor response exceeded its byte limit")
        try:
            decision = json.loads(stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("adaptive actor returned invalid JSON") from error
        if not isinstance(decision, dict) or decision.get("kind") not in {
            "continue",
            "accept",
            "stop",
        }:
            raise RuntimeError("adaptive actor returned an invalid decision")
        return decision


def parse_case(instruction: str) -> dict[str, object]:
    start = instruction.find(CASE_START)
    end = instruction.find(CASE_END)
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Harbor instruction has no multi-turn case envelope")
    payload = instruction[start + len(CASE_START) : end].strip()
    try:
        case = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("Harbor instruction has an invalid multi-turn case envelope") from error
    if not isinstance(case, dict) or case.get("schema_version") != 1:
        raise ValueError("Harbor instruction has an invalid multi-turn case envelope")
    return case


def required_mapping(value: dict[str, object], key: str) -> dict[str, object]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"Harbor case field {key!r} must be an object")
    return result


def required_string(value: dict[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"Harbor case field {key!r} must be a non-empty string")
    return result


def required_positive_integer(value: dict[str, object], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool) or result < 1:
        raise ValueError(f"Harbor case field {key!r} must be a positive integer")
    return result


def required_positive_number(value: dict[str, object], key: str) -> float:
    result = value.get(key)
    if not isinstance(result, int | float) or isinstance(result, bool) or result <= 0:
        raise ValueError(f"Harbor case field {key!r} must be a positive number")
    return float(result)


async def read_bounded(stream: asyncio.StreamReader, limit: int) -> tuple[bytes, bool]:
    captured = bytearray()
    truncated = False
    while chunk := await stream.read(8192):
        remaining = limit - len(captured)
        if remaining > 0:
            captured.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated = True
    return bytes(captured), truncated
