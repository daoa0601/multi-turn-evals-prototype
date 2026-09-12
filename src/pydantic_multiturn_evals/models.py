"""Domain models and strict YAML contracts for adaptive conversations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]*$")]
EnvironmentName = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]*$")]
TargetKind: TypeAlias = Literal["pydantic_ai", "command", "agentenv"]
ExecutionKind: TypeAlias = Literal["pydantic_ai", "command", "agentenv", "harbor"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UserTurn(StrictModel):
    role: Literal["user"] = "user"
    content: Text


class AssistantTurn(StrictModel):
    role: Literal["assistant"] = "assistant"
    content: Text


class Exchange(StrictModel):
    user: UserTurn
    assistant: AssistantTurn


@dataclass(frozen=True, slots=True)
class ConversationView:
    """A complete history plus the one user message awaiting a response."""

    run_id: str
    scenario_id: str
    exchanges: tuple[Exchange, ...]
    pending_user: UserTurn

    @property
    def messages(self) -> tuple[UserTurn | AssistantTurn, ...]:
        messages: list[UserTurn | AssistantTurn] = []
        for exchange in self.exchanges:
            messages.extend((exchange.user, exchange.assistant))
        messages.append(self.pending_user)
        return tuple(messages)


class ContinueDecision(StrictModel):
    kind: Literal["continue"] = "continue"
    next_user_message: Text
    reason: Text


class AcceptDecision(StrictModel):
    kind: Literal["accept"] = "accept"
    reason: Text


class StopDecision(StrictModel):
    kind: Literal["stop"] = "stop"
    reason: Text


ActorDecision: TypeAlias = Annotated[
    ContinueDecision | AcceptDecision | StopDecision,
    Field(discriminator="kind"),
]


class ActorAccepted(StrictModel):
    kind: Literal["actor_accepted"] = "actor_accepted"
    reason: Text


class ActorStopped(StrictModel):
    kind: Literal["actor_stopped"] = "actor_stopped"
    reason: Text


class TurnLimitReached(StrictModel):
    kind: Literal["turn_limit_reached"] = "turn_limit_reached"
    limit: int = Field(ge=1)


Termination: TypeAlias = Annotated[
    ActorAccepted | ActorStopped | TurnLimitReached,
    Field(discriminator="kind"),
]


class TargetReply(StrictModel):
    assistant_text: Text
    session_id: str | None = None
    evidence: dict[str, JsonValue] = Field(default_factory=dict)


class TargetTurnEvidence(StrictModel):
    turn_index: int = Field(ge=1)
    duration_seconds: float = Field(ge=0)
    session_id: str | None = None
    executable: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class TargetFailureEvidence(StrictModel):
    run_id: str
    scenario_id: Identifier
    repeat_index: int = Field(ge=1)
    error_type: str
    stderr: str = ""
    stderr_truncated: bool = False
    exit_code: int | None = None


class Transcript(StrictModel):
    """The only part of a result shown to the LLM judge."""

    exchanges: tuple[Exchange, ...]


class SessionOutcome(StrictModel):
    """Terminal conversation state passed to a target before its resources close."""

    transcript: Transcript
    decisions: tuple[ActorDecision, ...]
    termination: Termination


class EnvironmentEvidence(StrictModel):
    """Structured verifier output kept outside the judge transcript."""

    provider: Identifier
    environment_id: Text | None = None
    trial_id: Text | None = None
    verifier: Text
    passed: bool
    reward: float | None = Field(default=None, ge=0, le=1)
    reason: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class TargetCompletion(StrictModel):
    environment: EnvironmentEvidence | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class ScenarioResult(StrictModel):
    run_id: str
    scenario_id: Identifier
    repeat_index: int = Field(default=1, ge=1)
    transcript: Transcript
    target_evidence: tuple[TargetTurnEvidence, ...] = ()
    decisions: tuple[ActorDecision, ...]
    termination: Termination
    completion: TargetCompletion = TargetCompletion()


class ConversationState(StrictModel):
    """Short-lived runner state saved after every complete transition."""

    run_id: str
    scenario_id: Identifier
    exchanges: tuple[Exchange, ...] = ()
    pending_user: UserTurn | None
    decisions: tuple[ActorDecision, ...] = ()
    termination: Termination | None = None

    @model_validator(mode="after")
    def validate_phase(self) -> ConversationState:
        running = self.termination is None
        if running != (self.pending_user is not None):
            raise ValueError(
                "running state needs a pending user turn; terminal state cannot have one"
            )
        return self


class ActorBrief(StrictModel):
    persona: Text
    goal: Text
    rules: tuple[Text, ...] = ()


class ScenarioLimits(StrictModel):
    max_target_turns: int = Field(default=4, ge=1, le=50)
    timeout_seconds: float = Field(default=120, gt=0, le=1800)


class Scenario(StrictModel):
    id: Identifier
    first_prompt: Text
    actor: ActorBrief
    judge_rubric: Text
    tags: frozenset[Identifier] = frozenset()
    limits: ScenarioLimits | None = None


class ZAIProviderSpec(StrictModel):
    kind: Literal["zai"] = "zai"
    endpoint_plan: Literal["general", "coding"] = "coding"
    api_key_env: EnvironmentName = "ZAI_API_KEY"


class OpenAIProviderSpec(StrictModel):
    kind: Literal["openai"] = "openai"
    interface: Literal["responses", "chat"] = "responses"
    api_key_env: EnvironmentName = "OPENAI_API_KEY"


class AnthropicProviderSpec(StrictModel):
    kind: Literal["anthropic"] = "anthropic"
    api_key_env: EnvironmentName = "ANTHROPIC_API_KEY"


class OpenAICompatibleProviderSpec(StrictModel):
    kind: Literal["openai-compatible"] = "openai-compatible"
    interface: Literal["responses", "chat"] = "chat"
    base_url: AnyHttpUrl
    api_key_env: EnvironmentName | None = None


ProviderSpec: TypeAlias = Annotated[
    ZAIProviderSpec | OpenAIProviderSpec | AnthropicProviderSpec | OpenAICompatibleProviderSpec,
    Field(discriminator="kind"),
]


class ModelOptions(StrictModel):
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    thinking: bool | Literal["minimal", "low", "medium", "high", "xhigh"] | None = None
    max_tokens: int = Field(default=2048, ge=1, le=131072)
    timeout_seconds: float = Field(default=90, gt=0, le=600)


class ModelSpec(StrictModel):
    name: Text = "glm-5.3-flash"
    provider: ProviderSpec = ZAIProviderSpec()
    options: ModelOptions = ModelOptions()


class ActorSpec(StrictModel):
    model: ModelSpec = ModelSpec()
    instructions: Text = (
        "Stay in character. Inspect the latest assistant response and choose whether to continue, "
        "accept, or stop. Never grade against criteria you were not given."
    )
    model_request_limit: int = Field(default=2, ge=1, le=5)


class JudgeSpec(StrictModel):
    model: ModelSpec = ModelSpec()


class GatePolicy(StrictModel):
    minimum_case_pass_rate: float = Field(default=1.0, ge=0, le=1)
    minimum_mean_score: float = Field(default=0.7, ge=0, le=1)


class SuiteSpec(StrictModel):
    version: Literal[1]
    name: Identifier
    actor: ActorSpec = ActorSpec()
    judge: JudgeSpec = JudgeSpec()
    limits: ScenarioLimits = ScenarioLimits()
    gate: GatePolicy = GatePolicy()
    scenarios: tuple[Scenario, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_scenarios(self) -> SuiteSpec:
        ids = [scenario.id for scenario in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario ids must be unique")
        return self

    def limits_for(self, scenario: Scenario) -> ScenarioLimits:
        return scenario.limits or self.limits


class PydanticAITargetSpec(StrictModel):
    version: Literal[1]
    name: Identifier
    kind: Literal["pydantic_ai"]
    model: ModelSpec = ModelSpec()
    instructions: Text


class CommandLimits(StrictModel):
    startup_seconds: float = Field(default=10, gt=0, le=120)
    turn_seconds: float = Field(default=90, gt=0, le=600)
    shutdown_seconds: float = Field(default=2, gt=0, le=30)
    stdout_bytes_per_message: int = Field(default=1024 * 1024, gt=0, le=16 * 1024 * 1024)
    stderr_bytes_per_session: int = Field(default=64 * 1024, gt=0, le=16 * 1024 * 1024)


class CommandTargetSpec(StrictModel):
    version: Literal[1]
    name: Identifier
    kind: Literal["command"]
    argv: tuple[Text, ...] = Field(min_length=1)
    cwd: Path = Path(".")
    inherit_env: tuple[EnvironmentName, ...] = ()
    limits: CommandLimits = CommandLimits()

    @model_validator(mode="after")
    def reject_duplicate_environment_names(self) -> CommandTargetSpec:
        if len(self.inherit_env) != len(set(self.inherit_env)):
            raise ValueError("inherit_env names must be unique")
        return self


class SandboxCommandSpec(StrictModel):
    argv: tuple[Text, ...] = Field(min_length=1)
    cwd: str | None = None


class AgentEnvCredentials(StrictModel):
    api_url_env: EnvironmentName = "E2B_API_URL"
    sandbox_url_env: EnvironmentName = "E2B_SANDBOX_URL"
    api_key_env: EnvironmentName = "E2B_API_KEY"


class AgentEnvLimits(StrictModel):
    create_seconds: float = Field(default=60, gt=0, le=600)
    turn_seconds: float = Field(default=120, gt=0, le=1800)
    verifier_seconds: float = Field(default=120, gt=0, le=1800)
    destroy_seconds: float = Field(default=20, gt=0, le=120)
    sandbox_ttl_seconds: int = Field(default=900, ge=60, le=86400)
    response_bytes: int = Field(default=1024 * 1024, gt=0, le=16 * 1024 * 1024)


class AgentEnvTargetSpec(StrictModel):
    version: Literal[1]
    name: Identifier
    kind: Literal["agentenv"]
    template: Text
    turn: SandboxCommandSpec
    verifier: SandboxCommandSpec
    guest_env: tuple[EnvironmentName, ...] = ()
    credentials: AgentEnvCredentials = AgentEnvCredentials()
    limits: AgentEnvLimits = AgentEnvLimits()

    @model_validator(mode="after")
    def reject_duplicate_environment_names(self) -> AgentEnvTargetSpec:
        if len(self.guest_env) != len(set(self.guest_env)):
            raise ValueError("guest_env names must be unique")
        return self


TargetSpec: TypeAlias = Annotated[
    PydanticAITargetSpec | CommandTargetSpec | AgentEnvTargetSpec,
    Field(discriminator="kind"),
]


class CaseKey(StrictModel):
    scenario_id: Identifier
    repeat_index: int = Field(ge=1)


class PlannedCase(StrictModel):
    key: CaseKey
    case_name: Text
    scenario: Scenario


@dataclass(frozen=True, slots=True)
class SessionContext:
    suite_name: str
    target_name: str
    target_kind: TargetKind
    target_version: int
    key: CaseKey
    run_id: str
    comparison_id: str | None = None
    arm: Literal["baseline", "candidate"] | None = None


class CaseGate(StrictModel):
    case_name: str
    scenario_id: str
    repeat_index: int = Field(default=1, ge=1)
    passed: bool
    score: float | None = None
    assertion: bool | None = None
    environment_passed: bool | None = None
    environment_reward: float | None = None
    reason: str | None = None
    errors: tuple[str, ...] = ()


class GateResult(StrictModel):
    schema_version: Literal[1] = 1
    passed: bool
    case_pass_rate: float
    mean_score: float
    cases: tuple[CaseGate, ...]
