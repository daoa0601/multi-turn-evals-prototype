from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


def test_plan_compiles_an_experiment_without_provider_credentials(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    for name in ("ZAI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "planned"

    exit_code = main(
        [
            "plan",
            str(ROOT / "experiments" / "support-ab.yaml"),
            "--out",
            str(output),
        ]
    )

    assert exit_code == 0
    assert (output / "plan.json").exists()
    assert (output / "summary.json").exists()


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


def test_harbor_compare_has_two_explicit_arm_configs() -> None:
    args = build_parser().parse_args(
        [
            "harbor-compare",
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


def test_harbor_compare_writes_results_and_removes_successful_workdir(
    tmp_path: Path, monkeypatch: Any
) -> None:
    class FakeTrace:
        def shutdown(self) -> None:
            return None

    class FakeResult:
        passed = True

        def write_artifacts(self, output: Path) -> None:
            output.mkdir(parents=True)
            (output / "gate.json").write_text("{}")

        def summary(self) -> SimpleNamespace:
            return SimpleNamespace(model_dump_json=lambda indent: '{"passed":true}')

    async def fake_compare(*args: object, **kwargs: object) -> FakeResult:
        work_directory = kwargs["work_directory"]
        assert isinstance(work_directory, Path)
        work_directory.mkdir(parents=True)
        return FakeResult()

    monkeypatch.setattr("pydantic_multiturn_evals.cli.load_suite", lambda path: object())
    monkeypatch.setattr("pydantic_multiturn_evals.cli.load_harbor_arm", lambda path: object())
    monkeypatch.setattr("pydantic_multiturn_evals.cli.compare_harbor_suite", fake_compare)
    monkeypatch.setattr("pydantic_multiturn_evals.cli._trace_runtime", lambda enabled: FakeTrace())
    output = tmp_path / "output"

    exit_code = main(
        [
            "harbor-compare",
            "suite.yaml",
            "--baseline",
            "baseline.yaml",
            "--candidate",
            "candidate.yaml",
            "--out",
            str(output),
        ]
    )

    assert exit_code == 0
    assert (output / "gate.json").exists()
    assert not (tmp_path / "output.harbor-work").exists()
