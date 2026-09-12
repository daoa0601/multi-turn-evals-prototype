"""Validate generated tasks with the Harbor version used for execution."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from harbor.models.task.config import TaskConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="+", type=Path)
    args = parser.parse_args()

    for task_directory in args.tasks:
        task_file = task_directory / "task.toml"
        payload = tomllib.loads(task_file.read_text(encoding="utf-8"))
        TaskConfig.model_validate(payload)

    print(f"Harbor parsed {len(args.tasks)} generated task(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
