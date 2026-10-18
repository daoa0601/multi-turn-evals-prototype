"""Command line entry point for Python-authored campaigns."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic_multiturn_evals.campaign import read_report, run
from pydantic_multiturn_evals.model import ArtifactLimits, Campaign, Shard
from pydantic_multiturn_evals.shards import merge_reports
from pydantic_multiturn_evals.tracing import LangfuseExporter


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        campaign = _load_campaign(arguments.campaign)
        exporters = (LangfuseExporter.from_environment(),) if arguments.langfuse else ()
        report = asyncio.run(
            run(
                campaign,
                directory=arguments.directory,
                workers=arguments.workers,
                retry_interrupted=arguments.retry_interrupted,
                exporters=exporters,
                export_timeout_seconds=arguments.trace_export_timeout,
                capacities=_capacities(arguments.capacity),
                artifact_limits=ArtifactLimits(
                    files=arguments.artifact_files,
                    file_bytes=arguments.artifact_file_bytes,
                    total_bytes=arguments.artifact_total_bytes,
                ),
                shard=arguments.shard,
            )
        )
    elif arguments.command == "merge":
        campaign = _load_campaign(arguments.campaign)
        report = merge_reports(campaign, arguments.directories)
    else:
        report = read_report(arguments.directory)
    output: dict[str, object] = report.model_dump(mode="json")
    if arguments.compare is not None:
        output["comparison"] = report.compare(*arguments.compare).model_dump(mode="json")
    print(json.dumps(output, indent=2))
    return report.exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multiturn-evals")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run or resume a campaign")
    run_parser.add_argument("campaign", help="Python value as module:attribute")
    run_parser.add_argument("--directory", type=Path, required=True)
    run_parser.add_argument("--workers", type=int, default=4)
    run_parser.add_argument("--retry-interrupted", action="store_true")
    run_parser.add_argument("--langfuse", action="store_true")
    run_parser.add_argument("--trace-export-timeout", type=float, default=10)
    run_parser.add_argument(
        "--capacity",
        action="append",
        default=[],
        metavar="NAME=SLOTS",
        help="shared local capacity; repeat for more resources",
    )
    run_parser.add_argument("--shard", type=_shard, metavar="INDEX/COUNT")
    run_parser.add_argument("--artifact-files", type=int, default=32)
    run_parser.add_argument("--artifact-file-bytes", type=int, default=64_000_000)
    run_parser.add_argument("--artifact-total-bytes", type=int, default=256_000_000)
    run_parser.add_argument("--compare", nargs=2, metavar=("LEFT", "RIGHT"))
    report_parser = commands.add_parser("report", help="read a saved report")
    report_parser.add_argument("directory", type=Path)
    report_parser.add_argument("--compare", nargs=2, metavar=("LEFT", "RIGHT"))
    merge_parser = commands.add_parser("merge", help="merge a complete set of static shards")
    merge_parser.add_argument("campaign", help="Python value as module:attribute")
    merge_parser.add_argument("directories", type=Path, nargs="+")
    merge_parser.add_argument("--compare", nargs=2, metavar=("LEFT", "RIGHT"))
    return parser


def _load_campaign(reference: str) -> Campaign:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("campaign must use module:attribute syntax")
    directory = str(Path.cwd())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    value = getattr(importlib.import_module(module_name), attribute)
    if not isinstance(value, Campaign):
        raise TypeError(f"{reference} is not a Campaign")
    return value


def _capacities(values: Sequence[str]) -> dict[str, int]:
    capacities: dict[str, int] = {}
    for value in values:
        name, separator, slots_text = value.partition("=")
        if not separator or not name:
            raise ValueError(f"capacity must use NAME=SLOTS syntax: {value!r}")
        try:
            slots = int(slots_text)
        except ValueError as error:
            raise ValueError(f"capacity slots must be an integer: {value!r}") from error
        if name in capacities:
            raise ValueError(f"capacity {name!r} is specified more than once")
        capacities[name] = slots
    return capacities


def _shard(value: str) -> Shard:
    index_text, separator, count_text = value.partition("/")
    if not separator:
        raise argparse.ArgumentTypeError("shard must use INDEX/COUNT syntax")
    try:
        return Shard(index=int(index_text), count=int(count_text))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


if __name__ == "__main__":
    raise SystemExit(main())
