"""The values shared by every harness, evaluator, and trace exporter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator

Name = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
Text = Annotated[str, Field(min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentDefinition(StrictModel):
    """Stable identity written into the campaign manifest."""

    kind: Name
    version: Text
    config: JsonValue = None


class Criterion(StrictModel):
    name: Name
    instruction: Text


class TrialBudget(StrictModel):
    turns: Annotated[int, Field(ge=1)] = 8
    seconds: Annotated[float, Field(gt=0)] = 300


class ResourceClaim(StrictModel):
    """Units a trial occupies until its harness has cleaned up."""

    name: Name
    slots: Annotated[int, Field(ge=1)] = 1


class ArtifactLimits(StrictModel):
    """Attempt-local evidence limits enforced before evaluation."""

    files: Annotated[int, Field(ge=1)] = 32
    file_bytes: Annotated[int, Field(ge=1)] = 64_000_000
    total_bytes: Annotated[int, Field(ge=1)] = 256_000_000

    @model_validator(mode="after")
    def require_one_file_to_fit(self) -> Self:
        if self.file_bytes > self.total_bytes:
            raise ValueError("file_bytes cannot exceed total_bytes")
        return self


class Shard(StrictModel):
    """One static share of case/sample pairs. Indexes are zero based."""

    index: Annotated[int, Field(ge=0)]
    count: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def require_valid_index(self) -> Self:
        if self.index >= self.count:
            raise ValueError("shard index must be less than shard count")
        return self


class Case(StrictModel):
    """A task plus the quality signals required to decide it."""

    name: Name
    task: Text
    criteria: tuple[Criterion, ...] = ()
    required_checks: tuple[Name, ...] = ()
    data: JsonValue = None
    budget: TrialBudget = TrialBudget()

    @model_validator(mode="after")
    def require_quality_signal(self) -> Case:
        criterion_names = [criterion.name for criterion in self.criteria]
        if len(criterion_names) != len(set(criterion_names)):
            raise ValueError("criterion names must be unique")
        if len(self.required_checks) != len(set(self.required_checks)):
            raise ValueError("required check names must be unique")
        if not self.criteria and not self.required_checks:
            raise ValueError("a case needs at least one criterion or required check")
        return self


class Message(StrictModel):
    role: Name
    content: Text
    actor: Name | None = None


class PublicOutcome(StrictModel):
    """Evidence safe to give an evaluator."""

    messages: tuple[Message, ...] = ()
    output: JsonValue = None


class NativeCheck(StrictModel):
    name: Name
    passed: bool
    reason: Text


class Artifact(StrictModel):
    name: Name
    path: Text
    media_type: Text | None = None


class Evidence(StrictModel):
    """Harness output. Native checks are facts, not model judgments."""

    public: PublicOutcome = PublicOutcome()
    checks: tuple[NativeCheck, ...] = ()
    artifacts: tuple[Artifact, ...] = ()

    @model_validator(mode="after")
    def reject_duplicate_names(self) -> Evidence:
        check_names = [check.name for check in self.checks]
        artifact_names = [artifact.name for artifact in self.artifacts]
        if len(check_names) != len(set(check_names)):
            raise ValueError("native check names must be unique")
        if len(artifact_names) != len(set(artifact_names)):
            raise ValueError("artifact names must be unique")
        return self


class EvaluationArtifact(StrictModel):
    """A validated trial-local artifact exposed to non-model evaluators."""

    name: Name
    path: Path
    media_type: Text | None = None


class EvaluationInput(StrictModel):
    """The only case and evidence fields visible to an evaluator."""

    task: Text
    criteria: tuple[Criterion, ...]
    outcome: PublicOutcome
    artifacts: tuple[EvaluationArtifact, ...] = ()


class Judgment(StrictModel):
    criterion: Name
    passed: bool
    reason: Text
    score: Annotated[float, Field(ge=0, le=1)] | None = None


class TrialKey(StrictModel):
    campaign: Name
    revision: Name
    case: Name
    variant: Name
    sample: Annotated[int, Field(ge=1)]


@dataclass(frozen=True, slots=True)
class Trial:
    key: TrialKey
    attempt: int
    directory: Path
    seed: int


class LifecycleFact(StrictModel):
    kind: Literal["lifecycle"] = "lifecycle"
    phase: Name
    detail: str | None = None


class MessageFact(StrictModel):
    kind: Literal["message"] = "message"
    message: Message


class ToolFact(StrictModel):
    kind: Literal["tool"] = "tool"
    name: Text
    input: JsonValue = None
    output: JsonValue = None


class CheckFact(StrictModel):
    kind: Literal["check"] = "check"
    check: NativeCheck


class ArtifactFact(StrictModel):
    kind: Literal["artifact"] = "artifact"
    artifact: Artifact


class ResourceFact(StrictModel):
    kind: Literal["resource"] = "resource"
    name: Name
    value: float
    unit: Name


class ModelUsage(StrictModel):
    """Reported usage; an absent token count is unknown."""

    requests: Annotated[int, Field(ge=0)] = 0
    tool_calls: Annotated[int, Field(ge=0)] = 0
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    cache_write_tokens: Annotated[int, Field(ge=0)] | None = None
    cache_read_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    input_audio_tokens: Annotated[int, Field(ge=0)] | None = None
    cache_audio_read_tokens: Annotated[int, Field(ge=0)] | None = None
    output_audio_tokens: Annotated[int, Field(ge=0)] | None = None
    cost_usd: Annotated[float, Field(ge=0)] | None = None
    details: dict[str, int] = Field(default_factory=dict)


class ModelCall(StrictModel):
    """One durable model request group made by an eval component."""

    role: Name
    operation: Name
    requested: ComponentDefinition
    reported_model: str | None = None
    provider: Name
    interface: Name
    sampling: JsonValue = None
    duration_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    status: Literal["completed", "failed", "cancelled"]
    usage: ModelUsage | None = None
    provider_response_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class ModelCallFact(StrictModel):
    kind: Literal["model-call"] = "model-call"
    call: ModelCall


class LogFact(StrictModel):
    kind: Literal["log"] = "log"
    level: Literal["debug", "info", "warning", "error"]
    message: Text


TraceFact = Annotated[
    LifecycleFact
    | MessageFact
    | ToolFact
    | CheckFact
    | ArtifactFact
    | ResourceFact
    | ModelCallFact
    | LogFact,
    Field(discriminator="kind"),
]


class TraceEvent(StrictModel):
    key: TrialKey
    attempt: Annotated[int, Field(ge=1)]
    sequence: Annotated[int, Field(ge=1)]
    elapsed_seconds: Annotated[float, Field(ge=0)]
    recorded_at: Text
    fact: TraceFact


class TraceRecorder(Protocol):
    def record(self, fact: TraceFact) -> None: ...


class Harness(Protocol):
    @property
    def component(self) -> ComponentDefinition: ...

    async def run(self, case: Case, trial: Trial, trace: TraceRecorder) -> Evidence: ...


class Evaluator(Protocol):
    @property
    def component(self) -> ComponentDefinition: ...

    async def evaluate(
        self,
        value: EvaluationInput,
        trace: TraceRecorder,
    ) -> tuple[Judgment, ...]: ...


class Failure(StrictModel):
    phase: Name
    message: Text
    exception_type: Text | None = None


class Completed(StrictModel):
    kind: Literal["completed"] = "completed"
    evidence: Evidence
    judgments: tuple[Judgment, ...] = ()

    @model_validator(mode="after")
    def require_decision_evidence(self) -> Completed:
        if not self.evidence.checks and not self.judgments:
            raise ValueError("completed trial needs a native check or evaluator judgment")
        return self

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.evidence.checks) and all(
            judgment.passed for judgment in self.judgments
        )


class Failed(StrictModel):
    kind: Literal["failed"] = "failed"
    failure: Failure
    evidence: Evidence | None = None
    judgments: tuple[Judgment, ...] = ()


class Interrupted(StrictModel):
    kind: Literal["interrupted"] = "interrupted"
    reason: Text


Terminal = Annotated[Completed | Failed | Interrupted, Field(discriminator="kind")]


class TrialResult(StrictModel):
    key: TrialKey
    attempt: Annotated[int, Field(ge=1)]
    duration_seconds: Annotated[float, Field(ge=0)]
    terminal: Terminal
    model_calls: tuple[ModelCall, ...] = ()

    @property
    def passed(self) -> bool:
        return isinstance(self.terminal, Completed) and self.terminal.passed


@dataclass(frozen=True, slots=True)
class Variant:
    """A harness and the evaluators applied to its evidence."""

    name: str
    harness: Harness
    evaluators: tuple[Evaluator, ...] = ()
    resources: tuple[ResourceClaim, ...] = ()

    def __post_init__(self) -> None:
        names = [claim.name for claim in self.resources]
        if len(names) != len(set(names)):
            raise ValueError(f"variant {self.name!r} has duplicate resource claims")


@dataclass(frozen=True, slots=True)
class Campaign:
    name: str
    revision: str
    cases: tuple[Case, ...]
    variants: tuple[Variant, ...]
    samples: int = 1

    def __post_init__(self) -> None:
        names = [case.name for case in self.cases]
        if len(names) != len(set(names)):
            raise ValueError("case names must be unique")
        variants = [variant.name for variant in self.variants]
        if len(variants) != len(set(variants)):
            raise ValueError("variant names must be unique")
        if not self.cases:
            raise ValueError("a campaign needs at least one case")
        if not self.variants:
            raise ValueError("a campaign needs at least one variant")
        if self.samples < 1:
            raise ValueError("samples must be positive")
        for value in (self.name, self.revision, *variants):
            try:
                TypeAdapter(Name).validate_python(value)
            except ValueError as error:
                raise ValueError(f"invalid campaign identifier: {value!r}") from error
