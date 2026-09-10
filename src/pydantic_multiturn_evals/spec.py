"""Load authored YAML at one explicit validation boundary."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from pydantic_multiturn_evals.models import PydanticAITargetSpec, SuiteSpec

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_suite(path: str | Path) -> SuiteSpec:
    return _load_model(path, SuiteSpec)


def load_target(path: str | Path) -> PydanticAITargetSpec:
    return _load_model(path, PydanticAITargetSpec)


def _load_model(path: str | Path, model_type: type[ModelT]) -> ModelT:
    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"could not read {source}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {source}: {error}") from error
    try:
        return model_type.model_validate(payload)
    except ValidationError as error:
        raise ValueError(f"invalid {model_type.__name__} in {source}:\n{error}") from error
