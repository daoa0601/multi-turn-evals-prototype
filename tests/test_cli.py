from __future__ import annotations

from pathlib import Path

from pydantic_multiturn_evals.cli import main

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
