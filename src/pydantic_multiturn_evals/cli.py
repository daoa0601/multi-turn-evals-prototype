"""Command-line entry point for validation and live evaluations."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic_multiturn_evals.evaluation import evaluate_suite
from pydantic_multiturn_evals.observability import TraceRuntime, enable_langfuse
from pydantic_multiturn_evals.providers import PydanticAITarget
from pydantic_multiturn_evals.spec import load_suite, load_target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multiturn-evals")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate suite and target YAML")
    validate.add_argument("suite", type=Path)
    validate.add_argument("--target", type=Path)

    run = commands.add_parser("run", help="run and judge an adaptive scenario suite")
    run.add_argument("suite", type=Path)
    run.add_argument("--target", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--max-concurrency", type=int, default=1)
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--no-progress", action="store_true")
    run.add_argument("--langfuse", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            suite = load_suite(args.suite)
            if args.target is not None:
                load_target(args.target)
            print(f"valid suite {suite.name!r} with {len(suite.scenarios)} scenario(s)")
            return 0
        return _run(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be positive")
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")
    suite = load_suite(args.suite)
    target = PydanticAITarget(load_target(args.target))
    trace_runtime: TraceRuntime | None = enable_langfuse() if args.langfuse else None
    try:
        result = asyncio.run(
            evaluate_suite(
                suite,
                target=target,
                max_concurrency=args.max_concurrency,
                repeat=args.repeat,
                progress=not args.no_progress,
            )
        )
    finally:
        if trace_runtime is not None:
            trace_runtime.flush()
    result.report.print(include_reasons=True)
    result.write_artifacts(args.out)
    print(f"wrote evaluation artifacts to {args.out}")
    return 0 if result.gate.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
