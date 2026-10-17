"""Install the pinned tau3-bench checkout in its own uv environment."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

TAU3_REPOSITORY = "https://github.com/sierra-research/tau2-bench.git"
TAU3_VERSION = "1.0.1"
TAU3_COMMIT = "2174a603f6d014ef94473ffa95957f6ce27100db"
TAU3_TEXT_RUNTIME = "websockets==15.0.1"
DEFAULT_CHECKOUT = Path(".external") / f"tau3-{TAU3_VERSION}"


def install(checkout: Path) -> dict[str, str]:
    checkout = checkout.resolve()
    if checkout.exists():
        _verify_checkout(checkout)
    else:
        git = _required_command("git")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            (git, "clone", "--filter=blob:none", TAU3_REPOSITORY, str(checkout)),
            check=True,
        )
        subprocess.run(
            (git, "checkout", "--detach", TAU3_COMMIT),
            cwd=checkout,
            check=True,
        )
    uv = _required_command("uv")
    subprocess.run((uv, "sync", "--project", str(checkout)), check=True)
    _verify_checkout(checkout)
    python = checkout / ".venv" / "bin" / "python"
    # Tau3 1.0.1 imports its OpenAI voice provider from the core text path, but
    # declares websockets only in the voice extra. Use its locked version without
    # installing the unrelated audio stack.
    subprocess.run(
        (uv, "pip", "install", "--python", str(python), TAU3_TEXT_RUNTIME),
        check=True,
    )
    version = subprocess.run(
        (
            str(python),
            "-c",
            "import importlib.metadata; print(importlib.metadata.version('tau2'))",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if version != TAU3_VERSION:
        raise RuntimeError(f"installed tau2 version {version!r}, expected {TAU3_VERSION!r}")
    return {
        "checkout": str(checkout),
        "commit": TAU3_COMMIT,
        "package_version": version,
        "python": str(python),
        "text_runtime_workaround": TAU3_TEXT_RUNTIME,
    }


def _verify_checkout(checkout: Path) -> None:
    git = _required_command("git")
    if not (checkout / ".git").is_dir():
        raise ValueError(f"existing Tau3 path is not a Git checkout: {checkout}")
    status = subprocess.run(
        (git, "status", "--porcelain"),
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError(f"existing Tau3 checkout is dirty: {checkout}")
    head = subprocess.run(
        (git, "rev-parse", "HEAD"),
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != TAU3_COMMIT:
        raise ValueError(
            f"existing Tau3 checkout is at {head}, expected {TAU3_COMMIT}; use another directory"
        )


def _required_command(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} is not installed")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    arguments = parser.parse_args()
    print(json.dumps(install(arguments.checkout), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
