"""A bounded live Z.AI agent comparison through Harbor and PydanticJudge.

Requires Harbor 0.1.44, Docker, and ZAI_API_KEY. This calls the Z.AI coding
endpoint. The no-op baseline is expected to fail quality.
"""

from pathlib import Path
from shutil import which

from pydantic_multiturn_evals import (
    Campaign,
    Case,
    Criterion,
    ResourceClaim,
    TrialBudget,
    Variant,
)
from pydantic_multiturn_evals.adapters import HarborHarness, HarborLocalTask
from pydantic_multiturn_evals.providers import (
    CODING_ZAI_BASE_URL,
    ModelOptions,
    ModelSpec,
    PydanticJudge,
    ZAIProviderSpec,
)

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "integration" / "harbor" / "multiturn-task"
executable = which("harbor")
if executable is None:
    raise RuntimeError("Install Harbor 0.1.44 in its own environment before running this campaign")

binding = HarborLocalTask(case="confirm-state", path=TASK, revision="1.0.0")
judge = PydanticJudge(
    ModelSpec(
        name="glm-4.7",
        provider=ZAIProviderSpec(endpoint="coding"),
        options=ModelOptions(max_tokens=2048, timeout_seconds=60),
    )
)

campaign = Campaign(
    name="harbor-zai-state",
    revision="public-trajectory-1",
    cases=(
        Case(
            name=binding.case,
            task=(TASK / "instruction.md").read_text(encoding="utf-8"),
            criteria=(
                Criterion(
                    name="confirmed-state",
                    instruction=(
                        "The public transcript contains a read-back of the stored answer "
                        "before the agent claims completion. An empty transcript fails."
                    ),
                ),
            ),
            required_checks=("harbor.verified",),
            budget=TrialBudget(seconds=300),
        ),
    ),
    variants=(
        Variant(
            name="noop",
            harness=HarborHarness(
                executable=Path(executable),
                harbor_version="0.1.44",
                tasks=(binding,),
                agent="nop",
                inherit_environment=("PATH", "HOME"),
                timeout_seconds=240,
            ),
            evaluators=(judge,),
            resources=(ResourceClaim(name="docker"),),
        ),
        Variant(
            name="candidate",
            harness=HarborHarness(
                executable=Path(executable),
                harbor_version="0.1.44",
                tasks=(binding,),
                agent="terminus-2",
                public_trajectory=True,
                model="zai/glm-4.7",
                agent_arguments={
                    "api_base": CODING_ZAI_BASE_URL,
                    "max_turns": 6,
                    "temperature": 0.2,
                    "record_terminal_session": False,
                    "llm_kwargs": {
                        "max_tokens": 2048,
                        "timeout": 45,
                        "num_retries": 0,
                        "extra_body": {"thinking": {"type": "disabled"}},
                    },
                },
                inherit_environment=("PATH", "HOME", "ZAI_API_KEY"),
                timeout_seconds=240,
            ),
            evaluators=(judge,),
            resources=(ResourceClaim(name="docker"),),
        ),
    ),
)
