.PHONY: sync check test validate compat live

sync:
	uv sync --all-extras

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run basedpyright
	uv run pytest --cov=pydantic_multiturn_evals --cov-report=term-missing

test:
	uv run pytest -v

validate:
	uv run multiturn-evals validate scenarios/support.yaml --target targets/support.yaml

compat:
	uv run python scripts/compat_smoke.py offline

live:
	uv run multiturn-evals compare scenarios/support.yaml --baseline targets/support.yaml --candidate targets/support-candidate.yaml --out outputs/support-ab
