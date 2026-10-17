"""Run one tau3-bench text simulation inside its pinned Python environment."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any

from tau2.data_model.simulation import TextRunConfig
from tau2.evaluator.evaluator import EvaluationType
from tau2.runner import get_tasks, run_tasks


def main() -> None:
    request = _object(json.loads(sys.stdin.readline()), "request")
    _require(request.get("protocol") == 1, "request protocol must be 1")
    expected_version = _string(request, "package_version")
    actual_version = importlib.metadata.version("tau2")
    _require(
        actual_version == expected_version,
        f"tau2 version {actual_version!r} does not match {expected_version!r}",
    )
    task = _object(request.get("task"), "task")
    agent = _object(request.get("agent"), "agent")
    user = _object(request.get("user"), "user")
    root = Path(_required_environment("MTE_TRIAL_DIRECTORY")).resolve()
    results_path = _output_path(request, "results_path", root)
    reply_path = _output_path(request, "reply_path", root)
    output_directory = _output_path(request, "output_directory", root)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    domain = _string(task, "domain")
    task_id = _string(task, "task")
    task_set_value = task.get("task_set")
    if task_set_value is None:
        task_set = domain
    elif isinstance(task_set_value, str) and task_set_value:
        task_set = task_set_value
    else:
        raise ValueError("task.task_set must be null or a non-empty string")
    split = _string(task, "split")
    selected = get_tasks(task_set, task_split_name=split, task_ids=[task_id])
    config = TextRunConfig(
        domain=domain,
        task_set_name=task_set,
        task_split_name=split,
        task_ids=[task_id],
        num_tasks=1,
        agent=_string(agent, "implementation"),
        llm_agent=_string(agent, "model"),
        llm_args_agent=_object(agent.get("arguments"), "agent.arguments"),
        user=_string(user, "implementation"),
        llm_user=_string(user, "model"),
        llm_args_user=_object(user.get("arguments"), "user.arguments"),
        num_trials=1,
        max_steps=_positive_int(request, "max_steps"),
        max_errors=_positive_int(request, "max_errors"),
        timeout=_positive_number(request, "simulation_timeout_seconds"),
        save_to=None,
        max_concurrency=1,
        workers=0,
        seed=_int(request, "seed"),
        log_level="ERROR",
        verbose_logs=False,
        max_retries=0,
        retry_delay=1,
        auto_resume=False,
        auto_review=False,
        hallucination_retries=0,
        enforce_communication_protocol=_bool(request, "enforce_communication_protocol"),
    )
    results = run_tasks(
        config,
        selected,
        save_path=results_path,
        save_dir=output_directory,
        evaluation_type=EvaluationType.ALL,
        console_display=False,
        results_format="json",
    )
    _require(results_path.is_file(), "tau2 did not write its official results file")
    _write_json(
        reply_path,
        {
            "protocol": 1,
            "package_version": actual_version,
            "simulations": len(results.simulations),
        },
    )


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _string(values: dict[str, Any], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _int(values: dict[str, Any], name: str) -> int:
    value = values.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _positive_int(values: dict[str, Any], name: str) -> int:
    value = _int(values, name)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_number(values: dict[str, Any], name: str) -> float:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _bool(values: dict[str, Any], name: str) -> bool:
    value = values.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _output_path(values: dict[str, Any], name: str, root: Path) -> Path:
    path = Path(_string(values, name)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must stay inside the trial directory") from error
    return path


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is not set")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
