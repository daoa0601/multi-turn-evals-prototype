"""Load authored YAML at one explicit validation boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

import yaml
from pydantic import BaseModel, TypeAdapter, ValidationError

from pydantic_multiturn_evals.models import CommandTargetSpec, SuiteSpec, TargetSpec

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_suite(path: str | Path) -> SuiteSpec:
    return _load_model(path, SuiteSpec)


@dataclass(frozen=True, slots=True)
class LoadedTarget:
    spec: TargetSpec
    source_directory: Path


def load_target(path: str | Path) -> LoadedTarget:
    source = Path(path).resolve()
    payload = _read_yaml(source)
    try:
        spec = TypeAdapter(TargetSpec).validate_python(payload)
    except ValidationError as error:
        raise ValueError(f"invalid target in {source}:\n{error}") from error
    if isinstance(spec, CommandTargetSpec):
        cwd = spec.cwd if spec.cwd.is_absolute() else source.parent / spec.cwd
        spec = spec.model_copy(update={"cwd": cwd.resolve()})
    return LoadedTarget(spec=spec, source_directory=source.parent)


def _load_model(path: str | Path, model_type: type[ModelT]) -> ModelT:
    source = Path(path)
    payload = _read_yaml(source)
    try:
        return model_type.model_validate(payload)
    except ValidationError as error:
        raise ValueError(f"invalid {model_type.__name__} in {source}:\n{error}") from error


def _read_yaml(source: Path) -> object:
    try:
        return cast(object, yaml.safe_load(source.read_text(encoding="utf-8")))
    except OSError as error:
        raise ValueError(f"could not read {source}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {source}: {error}") from error
