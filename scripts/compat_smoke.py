"""Rerunnable compatibility checks for Harbor and AgentENV."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from pydantic_multiturn_evals.cli import main as cli_main
from pydantic_multiturn_evals.harbor_runner import (
    compile_harbor_tasks,
    load_harbor_arm,
    prepare_harbor_job,
)
from pydantic_multiturn_evals.spec import load_suite, load_target

ROOT = Path(__file__).parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("offline")

    agentenv = commands.add_parser("agentenv")
    agentenv.add_argument("--template", required=True)

    harbor = commands.add_parser("harbor")
    harbor.add_argument("--suite", type=Path, required=True)
    harbor.add_argument("--baseline", type=Path, required=True)
    harbor.add_argument("--candidate", type=Path, required=True)
    harbor.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "offline":
        return offline_smoke()
    if args.mode == "agentenv":
        return agentenv_smoke(args.template)
    return cli_main(
        [
            "harbor-compare",
            str(args.suite),
            "--baseline",
            str(args.baseline),
            "--candidate",
            str(args.candidate),
            "--out",
            str(args.out),
            "--max-concurrency",
            "1",
            "--repeat",
            "1",
            "--no-progress",
        ]
    )


def offline_smoke() -> int:
    suite = load_suite(ROOT / "scenarios" / "support.yaml")
    load_target(ROOT / "targets" / "agentenv-example.yaml")
    baseline = load_harbor_arm(ROOT / "harbor" / "baseline.example.yaml")
    candidate = load_harbor_arm(ROOT / "harbor" / "candidate.example.yaml")
    with TemporaryDirectory() as raw_directory:
        root = Path(raw_directory)
        first = compile_harbor_tasks(
            suite,
            repeat=2,
            destination=root / "tasks",
            task_template=baseline.spec.task_template,
            comparison_id="offline-smoke",
            arm="baseline",
        )
        second = compile_harbor_tasks(
            suite,
            repeat=2,
            destination=root / "tasks",
            task_template=baseline.spec.task_template,
            comparison_id="offline-smoke",
            arm="baseline",
        )
        if [case.task_name for case in first] != [case.task_name for case in second]:
            raise RuntimeError("Harbor task compilation was not stable")
        subprocess.run(
            [
                "uv",
                "run",
                "--python",
                "3.12",
                "--isolated",
                "--with",
                "harbor[e2b]==0.22.0",
                "python",
                str(ROOT / "scripts" / "validate_harbor_tasks.py"),
                *(str(case.task_path) for case in second),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                "uv",
                "run",
                "--python",
                "3.12",
                "--isolated",
                "--with",
                "harbor[e2b]==0.22.0",
                "--with",
                str(ROOT / "harbor" / "harbor_adapter"),
                "--with",
                "pytest",
                "python",
                "-m",
                "pytest",
                "-q",
                str(ROOT / "harbor" / "harbor_adapter" / "test_pydantic_multiturn_harbor_agent.py"),
            ],
            cwd=ROOT,
            check=True,
        )
        plans = [
            prepare_harbor_job(
                suite,
                loaded=loaded,
                role=role,
                comparison_id="offline-smoke",
                repeat=2,
                max_concurrency=3,
                work_directory=root / role,
            )
            for loaded, role in ((baseline, "baseline"), (candidate, "candidate"))
        ]
        for plan in plans:
            payload = yaml.safe_load(plan.config_path.read_text(encoding="utf-8"))
            if payload["n_attempts"] != 1 or payload["n_concurrent_trials"] != 3:
                raise RuntimeError("Harbor job bounds were not preserved")
            if len(payload["tasks"]) != len(first):
                raise RuntimeError("Harbor job did not contain every planned case")
            actor_command = payload["user_agent"]["kwargs"]["actor_command"]
            if not Path(actor_command[0]).is_file():
                raise RuntimeError("Harbor actor worker command does not exist")
    print(f"offline compatibility passed for {len(first)} Harbor case(s) per arm")
    return 0


def agentenv_smoke(template: str) -> int:
    missing = [
        name
        for name in ("E2B_API_URL", "E2B_SANDBOX_URL", "E2B_API_KEY")
        if not os.environ.get(name)
    ]
    if missing:
        raise ValueError(f"AgentENV smoke requires {', '.join(missing)}")
    try:
        from e2b import Sandbox  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("AgentENV smoke requires: uv sync --extra agentenv") from error

    sandbox = Sandbox.create(
        template=template,
        timeout=300,
        secure=True,
        api_url=os.environ["E2B_API_URL"],
        sandbox_url=os.environ["E2B_SANDBOX_URL"],
        api_key=os.environ["E2B_API_KEY"],
        request_timeout=60,
    )
    try:
        path = "/tmp/pydantic-multiturn-agentenv-smoke.json"
        expected = {"status": "ready"}
        sandbox.files.write(path, json.dumps(expected))
        command = sandbox.commands.run(f"test -s {path}", timeout=30)
        if command.exit_code != 0:
            raise RuntimeError("AgentENV command compatibility check failed")
        actual = json.loads(sandbox.files.read(path))
        if actual != expected:
            raise RuntimeError("AgentENV file compatibility check failed")
    finally:
        sandbox.kill()
    print("AgentENV create, command, file, and kill compatibility passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
