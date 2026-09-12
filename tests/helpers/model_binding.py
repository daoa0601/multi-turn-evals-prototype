from __future__ import annotations

from pydantic_ai import ModelSettings
from pydantic_ai.models import Model

from pydantic_multiturn_evals.model_bindings import BoundModel, ModelIdentity


def fake_model_binding(
    model: Model,
    settings: ModelSettings | None = None,
) -> BoundModel:
    return BoundModel(
        model=model,
        settings=settings or ModelSettings(),
        identity=ModelIdentity(
            provider="test",
            requested_model=model.model_name,
            interface="test",
        ),
        required_environment=(),
    )
