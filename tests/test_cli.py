from __future__ import annotations

from pathlib import Path

from pydantic_multiturn_evals.cli import build_parser, main

ROOT = Path(__file__).parents[1]


def test_validate_command_needs_no_provider_key(capsys: object) -> None:
    exit_code = main(
        [
            "validate",
            str(ROOT / "scenarios" / "support.yaml"),
            "--target",
            str(ROOT / "targets" / "support.yaml"),
        ]
    )

    assert exit_code == 0


def test_compare_command_requires_two_named_configs() -> None:
    args = build_parser().parse_args(
        [
            "compare",
            "suite.yaml",
            "--baseline",
            "baseline.yaml",
            "--candidate",
            "candidate.yaml",
            "--out",
            "output",
        ]
    )

    assert args.baseline == Path("baseline.yaml")
    assert args.candidate == Path("candidate.yaml")
