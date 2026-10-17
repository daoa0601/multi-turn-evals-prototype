"""Optional adapters for external evaluation systems."""

from pydantic_multiturn_evals.adapters.agentenv import AgentEnvArtifact, AgentEnvTarget
from pydantic_multiturn_evals.adapters.harbor import (
    HarborDatasetTask,
    HarborHarness,
    HarborLocalTask,
)
from pydantic_multiturn_evals.adapters.swebench import SweBenchHarness, SweBenchTask
from pydantic_multiturn_evals.adapters.tau3 import Tau3Participant, Tau3Task, Tau3TextHarness
from pydantic_multiturn_evals.adapters.templar import (
    TemplarArtifact,
    TemplarHarness,
    TemplarTask,
)

__all__ = [
    "AgentEnvArtifact",
    "AgentEnvTarget",
    "HarborDatasetTask",
    "HarborHarness",
    "HarborLocalTask",
    "SweBenchHarness",
    "SweBenchTask",
    "Tau3Participant",
    "Tau3Task",
    "Tau3TextHarness",
    "TemplarArtifact",
    "TemplarHarness",
    "TemplarTask",
]
