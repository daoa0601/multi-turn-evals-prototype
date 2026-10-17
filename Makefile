.PHONY: install install-tau3 sync format lint typecheck test verify verify-pydantic-scale verify-harbor verify-harbor-multiturn verify-harbor-pack verify-swebench verify-tau3 build check

install:
	uv sync --all-extras

install-tau3:
	uv run python -m scripts.install_tau3

sync:
	uv sync --all-extras

format:
	uv run ruff format .

lint:
	uv run ruff check .

typecheck:
	uv run basedpyright

test:
	uv run pytest

verify:
	uv run python -m scripts.verify_offline

verify-pydantic-scale:
	uv run python -m scripts.run_pydantic_scale

verify-harbor:
	uv run python -m scripts.verify_harbor

verify-harbor-multiturn:
	uv run python -m scripts.verify_harbor_multiturn

verify-harbor-pack:
	uv run python -m scripts.run_harbor_benchmark_pack

verify-swebench:
	uv run python -m scripts.verify_swebench

verify-tau3:
	uv run python -m scripts.verify_tau3

build:
	uv build

check: lint typecheck test verify build
