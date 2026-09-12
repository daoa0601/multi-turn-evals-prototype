"""Compile authored experiment choices into a self-contained session plan."""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias, TypeVar, cast

import yaml
from pydantic import Field, JsonValue, ValidationError, model_validator

from pydantic_multiturn_evals.model_bindings import required_environment
from pydantic_multiturn_evals.models import (
    ActorSpec,
    AgentEnvTargetSpec,
    CommandTargetSpec,
    EnvironmentName,
    GatePolicy,
    Identifier,
    JudgeSpec,
    ModelSpec,
    PydanticAITargetSpec,
    Scenario,
    ScenarioLimits,
    StrictModel,
    SuiteSpec,
    TargetSpec,
    Text,
)
from pydantic_multiturn_evals.spec import load_target

AxisName: TypeAlias = Literal[
    "actor",
    "target",
    "judge",
    "observer",
    "prompts.actor",
    "prompts.target",
    "prompts.judge",
    "prompts.opening",
    "prompts.trajectory",
    "fixture",
    "tasks",
    "harness",
    "execution",
    "limits",
]
PromptRole: TypeAlias = Literal["actor", "target", "judge", "opening", "trajectory"]

_AXIS_ORDER: tuple[AxisName, ...] = (
    "actor",
    "target",
    "judge",
    "observer",
    "prompts.actor",
    "prompts.target",
    "prompts.judge",
    "prompts.opening",
    "prompts.trajectory",
    "fixture",
    "tasks",
    "harness",
    "execution",
    "limits",
)
ModelT = TypeVar("ModelT", bound=StrictModel)


class CorpusSpec(StrictModel):
    version: Literal[1]
    name: Identifier
    revision: Identifier
    cases: tuple[Scenario, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_cases(self) -> CorpusSpec:
        identifiers = [case.id for case in self.cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("corpus case ids must be unique")
        return self


class ActorComponent(StrictModel):
    model: Identifier
    model_request_limit: int = Field(default=2, ge=1, le=5)


class JudgeComponent(StrictModel):
    model: Identifier


class ObserverComponent(StrictModel):
    model: Identifier


class TargetComponent(StrictModel):
    model: Identifier


class PromptComponent(StrictModel):
    role: PromptRole
    text: Text


class FixtureComponent(StrictModel):
    values: dict[Identifier, JsonValue] = Field(default_factory=dict)


class TaskSelector(StrictModel):
    kind: Literal["all", "ids", "tags"]
    ids: tuple[Identifier, ...] = ()
    include_all: tuple[Identifier, ...] = ()
    include_any: tuple[Identifier, ...] = ()
    exclude: tuple[Identifier, ...] = ()


class PydanticAIHarness(StrictModel):
    kind: Literal["pydantic_ai"]
    capture_target_evidence: bool = True


class CommandHarness(StrictModel):
    kind: Literal["command"]
    protocol_version: Literal[2] = 2
    path: Path
    environment: tuple[EnvironmentName, ...] = ("PATH",)


class AgentEnvHarness(StrictModel):
    kind: Literal["agentenv"]
    protocol_version: Literal[1] = 1
    path: Path
    guest_environment: tuple[EnvironmentName, ...] = ()


HarnessSpec: TypeAlias = Annotated[
    PydanticAIHarness | CommandHarness | AgentEnvHarness,
    Field(discriminator="kind"),
]


class HostExecution(StrictModel):
    kind: Literal["host"]
    workspace_mode: Literal["shared", "fresh"] = "shared"


class ContainerExecution(StrictModel):
    kind: Literal["container"]
    image: Text


class AgentEnvExecution(StrictModel):
    kind: Literal["agentenv"]
    region: Text | None = None


ExecutionSpec: TypeAlias = Annotated[
    HostExecution | ContainerExecution | AgentEnvExecution,
    Field(discriminator="kind"),
]


class LimitComponent(StrictModel):
    max_target_turns: int = Field(default=4, ge=1, le=50)
    timeout_seconds: float = Field(default=120, gt=0, le=1800)
    gate: GatePolicy = GatePolicy()

    @property
    def scenario_limits(self) -> ScenarioLimits:
        return ScenarioLimits(
            max_target_turns=self.max_target_turns,
            timeout_seconds=self.timeout_seconds,
        )


class ComponentCatalog(StrictModel):
    version: Literal[1]
    models: dict[Identifier, ModelSpec]
    actors: dict[Identifier, ActorComponent]
    targets: dict[Identifier, TargetComponent]
    judges: dict[Identifier, JudgeComponent]
    observers: dict[Identifier, ObserverComponent]
    prompts: dict[Identifier, PromptComponent]
    fixtures: dict[Identifier, FixtureComponent]
    task_selectors: dict[Identifier, TaskSelector]
    harnesses: dict[Identifier, HarnessSpec]
    executions: dict[Identifier, ExecutionSpec]
    limits: dict[Identifier, LimitComponent]


class PromptChoices(StrictModel):
    actor: Identifier
    target: Identifier
    judge: Identifier
    opening: Identifier
    trajectory: Identifier | None = None


class PromptSelection(StrictModel):
    actor: Identifier | None = None
    target: Identifier | None = None
    judge: Identifier | None = None
    opening: Identifier | None = None
    trajectory: Identifier | None = None


class ArmChoices(StrictModel):
    actor: Identifier
    target: Identifier
    judge: Identifier
    observer: Identifier
    prompts: PromptChoices
    fixture: Identifier
    tasks: Identifier
    harness: Identifier
    execution: Identifier
    limits: Identifier


class ArmSelection(StrictModel):
    actor: Identifier | None = None
    target: Identifier | None = None
    judge: Identifier | None = None
    observer: Identifier | None = None
    prompts: PromptSelection | None = None
    fixture: Identifier | None = None
    tasks: Identifier | None = None
    harness: Identifier | None = None
    execution: Identifier | None = None
    limits: Identifier | None = None


class ArmSpec(StrictModel):
    name: Identifier
    select: ArmSelection = ArmSelection()


class ComparisonSpec(StrictModel):
    name: Identifier
    baseline: Identifier
    candidate: Identifier
    pairing: Literal["exact"] = "exact"


class ArmsDesign(StrictModel):
    kind: Literal["arms"]
    arms: tuple[ArmSpec, ...] = Field(min_length=1)
    comparisons: tuple[ComparisonSpec, ...] = ()

    @model_validator(mode="after")
    def reject_duplicate_names(self) -> ArmsDesign:
        names = [arm.name for arm in self.arms]
        if len(names) != len(set(names)):
            raise ValueError("arm names must be unique")
        return self


class MatrixDesign(StrictModel):
    kind: Literal["matrix"]
    axes: dict[AxisName, tuple[Identifier, ...]] = Field(min_length=1)
    comparisons: tuple[ComparisonSpec, ...] = ()


ExperimentDesign: TypeAlias = Annotated[
    ArmsDesign | MatrixDesign,
    Field(discriminator="kind"),
]


class ExperimentBounds(StrictModel):
    max_arms: int = Field(default=32, ge=1, le=10_000)
    max_cases: int = Field(ge=1, le=1_000_000)
    max_concurrency: int = Field(ge=1, le=1000)
    max_model_requests: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    run_seconds: float = Field(gt=0, le=604_800)


class TrajectoryDiagnostics(StrictModel):
    enabled: bool = False
    max_prefixes_per_case: int = Field(default=4, ge=1, le=50)


class ExperimentDiagnostics(StrictModel):
    trajectory: TrajectoryDiagnostics = TrajectoryDiagnostics()


class ExperimentSpec(StrictModel):
    version: Literal[1]
    name: Identifier
    corpus: Path
    catalog: Path
    defaults: ArmChoices
    design: ExperimentDesign
    repeats: int = Field(default=1, ge=1, le=1000)
    diagnostics: ExperimentDiagnostics = ExperimentDiagnostics()
    bounds: ExperimentBounds


@dataclass(frozen=True, slots=True)
class LoadedExperiment:
    spec: ExperimentSpec
    corpus: CorpusSpec
    catalog: ComponentCatalog
    experiment_source: Path
    corpus_source: Path
    catalog_source: Path


class CorpusCaseKey(StrictModel):
    scenario_id: Identifier
    repeat_index: int = Field(ge=1)


class ArmCaseKey(StrictModel):
    arm: Identifier
    corpus: CorpusCaseKey


class PlannedExperimentCase(StrictModel):
    key: ArmCaseKey
    scenario: Scenario


class ResolvedPrompt(StrictModel):
    name: Identifier
    role: PromptRole
    text: Text


class ResolvedPrompts(StrictModel):
    actor: ResolvedPrompt
    target: ResolvedPrompt
    judge: ResolvedPrompt
    opening: ResolvedPrompt
    trajectory: ResolvedPrompt | None = None


class FixtureValue(StrictModel):
    name: Identifier
    json_value: str


class ResolvedFixture(StrictModel):
    name: Identifier
    values: tuple[FixtureValue, ...]


class SessionArmPlan(StrictModel):
    kind: Literal["session"] = "session"
    name: Identifier
    choices: ArmChoices
    suite: SuiteSpec
    observer_model: ModelSpec
    target_model: ModelSpec
    target: TargetSpec
    target_max_output_tokens_per_turn: int = Field(ge=1)
    prompts: ResolvedPrompts
    fixture: ResolvedFixture
    harness: HarnessSpec
    execution: ExecutionSpec
    cases: tuple[PlannedExperimentCase, ...]


class ComparisonPair(StrictModel):
    corpus: CorpusCaseKey
    baseline: ArmCaseKey
    candidate: ArmCaseKey


class ComparisonPlan(StrictModel):
    name: Identifier
    baseline: Identifier
    candidate: Identifier
    pairing: Literal["exact"] = "exact"
    pairs: tuple[ComparisonPair, ...]


class UsageReservation(StrictModel):
    arm_count: int = Field(ge=0)
    case_count: int = Field(ge=0)
    model_requests: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class ExperimentPlan(StrictModel):
    schema_version: Literal[1] = 1
    experiment_name: Identifier
    corpus: CorpusSpec
    arms: tuple[SessionArmPlan, ...]
    comparisons: tuple[ComparisonPlan, ...]
    diagnostics: ExperimentDiagnostics
    bounds: ExperimentBounds
    reservation: UsageReservation


@dataclass(frozen=True, slots=True)
class _Issue:
    code: str
    path: str
    message: str


def load_experiment(path: str | Path) -> LoadedExperiment:
    """Load strict authored inputs and resolve their source paths once."""

    experiment_source = Path(path).resolve()
    spec = _load_yaml_model(experiment_source, ExperimentSpec)
    corpus_source = _resolve_path(experiment_source.parent, spec.corpus)
    catalog_source = _resolve_path(experiment_source.parent, spec.catalog)
    return LoadedExperiment(
        spec=spec,
        corpus=_load_yaml_model(corpus_source, CorpusSpec),
        catalog=_load_yaml_model(catalog_source, ComponentCatalog),
        experiment_source=experiment_source,
        corpus_source=corpus_source,
        catalog_source=catalog_source,
    )


def compile_experiment(source: str | Path | LoadedExperiment) -> ExperimentPlan:
    """Resolve every authored choice without reading credentials or starting runtimes."""

    loaded = load_experiment(source) if isinstance(source, (str, Path)) else source
    issues: list[_Issue] = []
    _validate_catalog(loaded.catalog, issues)
    _validate_choice_refs(loaded.spec.defaults, loaded.catalog, "defaults", issues)
    _validate_diagnostics(loaded.spec, issues)

    authored_arms = _expand_arms(loaded.spec, issues)
    arm_plans: list[SessionArmPlan] = []
    arm_names = {arm.name for arm in authored_arms}
    for index, arm in enumerate(authored_arms):
        path = f"design.arms[{index}].select"
        _validate_selection_refs(arm.select, loaded.catalog, path, issues)
        choices = _apply_selection(loaded.spec.defaults, arm.select)
        plan = _compile_arm(loaded, arm.name, choices, path, issues)
        if plan is not None:
            arm_plans.append(plan)

    comparisons = _compile_comparisons(
        loaded.spec.design.comparisons,
        tuple(arm_plans),
        arm_names,
        issues,
    )
    reservation = _reserve_usage(tuple(arm_plans), loaded.spec.diagnostics)
    _validate_bounds(loaded.spec.bounds, reservation, issues)
    _raise_issues(issues)
    return ExperimentPlan(
        experiment_name=loaded.spec.name,
        corpus=loaded.corpus,
        arms=tuple(arm_plans),
        comparisons=comparisons,
        diagnostics=loaded.spec.diagnostics,
        bounds=loaded.spec.bounds,
        reservation=reservation,
    )


def _compile_arm(
    loaded: LoadedExperiment,
    arm_name: str,
    choices: ArmChoices,
    path: str,
    issues: list[_Issue],
) -> SessionArmPlan | None:
    catalog = loaded.catalog
    resolved = _resolve_arm_components(catalog, choices)
    if resolved is None:
        return None
    (
        actor,
        judge,
        observer,
        target_component,
        prompts,
        fixture,
        selector,
        harness,
        execution,
        limits,
    ) = resolved
    _validate_prompt_roles(choices, prompts, path, issues)
    _validate_compatibility(harness, execution, path, issues)
    scenarios = _select_cases(loaded.corpus, selector, choices.tasks, issues)
    if not scenarios:
        return None

    target = _resolve_target(
        choices.target,
        target_component,
        harness,
        prompts.target,
        catalog,
        loaded.catalog_source.parent,
        path,
        issues,
    )
    if target is None:
        return None
    if any(issue.path.startswith(path) for issue in issues):
        return None

    rendered_scenarios = tuple(
        scenario.model_copy(
            update={
                "first_prompt": prompts.opening.text.replace(
                    "{{first_prompt}}", scenario.first_prompt
                ),
                "judge_rubric": f"{prompts.judge.text}\n\nCase rubric:\n{scenario.judge_rubric}",
                "limits": limits.scenario_limits,
            }
        )
        for scenario in scenarios
    )
    suite = SuiteSpec(
        version=1,
        name=f"{loaded.spec.name}-{arm_name}",
        actor=ActorSpec(
            model=catalog.models[actor.model],
            instructions=prompts.actor.text,
            model_request_limit=actor.model_request_limit,
        ),
        judge=JudgeSpec(model=catalog.models[judge.model]),
        limits=limits.scenario_limits,
        gate=limits.gate,
        scenarios=rendered_scenarios,
    )
    cases = tuple(
        PlannedExperimentCase(
            key=ArmCaseKey(
                arm=arm_name,
                corpus=CorpusCaseKey(scenario_id=scenario.id, repeat_index=repeat_index),
            ),
            scenario=scenario,
        )
        for scenario in rendered_scenarios
        for repeat_index in range(1, loaded.spec.repeats + 1)
    )
    return SessionArmPlan(
        name=arm_name,
        choices=choices,
        suite=suite,
        observer_model=catalog.models[observer.model],
        target_model=catalog.models[target_component.model],
        target=target,
        target_max_output_tokens_per_turn=catalog.models[target_component.model].options.max_tokens,
        prompts=prompts,
        fixture=ResolvedFixture(
            name=choices.fixture,
            values=tuple(
                FixtureValue(
                    name=name,
                    json_value=json.dumps(value, sort_keys=True, separators=(",", ":")),
                )
                for name, value in sorted(fixture.values.items())
            ),
        ),
        harness=harness,
        execution=execution,
        cases=cases,
    )


def _resolve_arm_components(
    catalog: ComponentCatalog,
    choices: ArmChoices,
) -> (
    tuple[
        ActorComponent,
        JudgeComponent,
        ObserverComponent,
        TargetComponent,
        ResolvedPrompts,
        FixtureComponent,
        TaskSelector,
        HarnessSpec,
        ExecutionSpec,
        LimitComponent,
    ]
    | None
):
    prompt_names = choices.prompts
    required = (
        choices.actor in catalog.actors,
        choices.target in catalog.targets,
        choices.judge in catalog.judges,
        choices.observer in catalog.observers,
        prompt_names.actor in catalog.prompts,
        prompt_names.target in catalog.prompts,
        prompt_names.judge in catalog.prompts,
        prompt_names.opening in catalog.prompts,
        prompt_names.trajectory is None or prompt_names.trajectory in catalog.prompts,
        choices.fixture in catalog.fixtures,
        choices.tasks in catalog.task_selectors,
        choices.harness in catalog.harnesses,
        choices.execution in catalog.executions,
        choices.limits in catalog.limits,
    )
    if not all(required):
        return None
    actor = catalog.actors[choices.actor]
    judge = catalog.judges[choices.judge]
    observer = catalog.observers[choices.observer]
    target = catalog.targets[choices.target]
    if (
        actor.model not in catalog.models
        or judge.model not in catalog.models
        or observer.model not in catalog.models
    ):
        return None
    if target.model not in catalog.models:
        return None
    return (
        actor,
        judge,
        observer,
        target,
        ResolvedPrompts(
            actor=_resolved_prompt(prompt_names.actor, catalog.prompts[prompt_names.actor]),
            target=_resolved_prompt(prompt_names.target, catalog.prompts[prompt_names.target]),
            judge=_resolved_prompt(prompt_names.judge, catalog.prompts[prompt_names.judge]),
            opening=_resolved_prompt(prompt_names.opening, catalog.prompts[prompt_names.opening]),
            trajectory=(
                _resolved_prompt(
                    prompt_names.trajectory,
                    catalog.prompts[prompt_names.trajectory],
                )
                if prompt_names.trajectory is not None
                else None
            ),
        ),
        catalog.fixtures[choices.fixture],
        catalog.task_selectors[choices.tasks],
        catalog.harnesses[choices.harness],
        catalog.executions[choices.execution],
        catalog.limits[choices.limits],
    )


def _resolved_prompt(name: str, prompt: PromptComponent) -> ResolvedPrompt:
    return ResolvedPrompt(name=name, role=prompt.role, text=prompt.text)


def _resolve_target(
    name: str,
    component: TargetComponent,
    harness: HarnessSpec,
    target_prompt: ResolvedPrompt,
    catalog: ComponentCatalog,
    catalog_directory: Path,
    path: str,
    issues: list[_Issue],
) -> TargetSpec | None:
    model = catalog.models.get(component.model)
    if model is None:
        _issue(
            issues,
            "REF_UNKNOWN",
            f"catalog.targets.{name}.model",
            f"unknown model {component.model!r}",
        )
        return None
    if isinstance(harness, PydanticAIHarness):
        return PydanticAITargetSpec(
            version=1,
            name=name,
            kind="pydantic_ai",
            model=model,
            instructions=target_prompt.text,
        )

    source = _resolve_path(catalog_directory, harness.path)
    try:
        target = load_target(source).spec
    except ValueError as error:
        _issue(issues, "TARGET_LOAD", f"{path}.target", str(error).splitlines()[0])
        return None
    if target.kind != harness.kind:
        _issue(
            issues,
            "COMPAT_TARGET_FILE",
            f"{path}.target",
            f"harness declares {harness.kind!r} but {source} contains {target.kind!r}",
        )
        return None
    credential_names = required_environment(model)
    if isinstance(target, CommandTargetSpec):
        if not isinstance(harness, CommandHarness):
            raise AssertionError("command target requires a command harness")
        return target.model_copy(
            update={
                "name": name,
                "inherit_env": tuple(dict.fromkeys((*harness.environment, *credential_names))),
            }
        )
    if isinstance(target, AgentEnvTargetSpec):
        if not isinstance(harness, AgentEnvHarness):
            raise AssertionError("AgentENV target requires an AgentENV harness")
        return target.model_copy(
            update={
                "name": name,
                "guest_env": tuple(dict.fromkeys((*harness.guest_environment, *credential_names))),
            }
        )
    raise AssertionError(f"unhandled harness target: {target}")


def _validate_catalog(catalog: ComponentCatalog, issues: list[_Issue]) -> None:
    for name, actor in catalog.actors.items():
        if actor.model not in catalog.models:
            _issue(
                issues,
                "REF_UNKNOWN",
                f"catalog.actors.{name}.model",
                f"unknown model {actor.model!r}",
            )
    for name, judge in catalog.judges.items():
        if judge.model not in catalog.models:
            _issue(
                issues,
                "REF_UNKNOWN",
                f"catalog.judges.{name}.model",
                f"unknown model {judge.model!r}",
            )
    for name, observer in catalog.observers.items():
        if observer.model not in catalog.models:
            _issue(
                issues,
                "REF_UNKNOWN",
                f"catalog.observers.{name}.model",
                f"unknown model {observer.model!r}",
            )
    for name, target in catalog.targets.items():
        if target.model not in catalog.models:
            _issue(
                issues,
                "REF_UNKNOWN",
                f"catalog.targets.{name}.model",
                f"unknown model {target.model!r}",
            )
    for name, selector in catalog.task_selectors.items():
        if selector.kind == "ids" and not selector.ids:
            _issue(
                issues,
                "SELECTION_EMPTY",
                f"catalog.task_selectors.{name}",
                "an ids selector needs at least one id",
            )
        if selector.kind == "tags" and not (selector.include_all or selector.include_any):
            _issue(
                issues,
                "SELECTION_EMPTY",
                f"catalog.task_selectors.{name}",
                "a tags selector needs include_all or include_any",
            )


def _validate_choice_refs(
    choices: ArmChoices,
    catalog: ComponentCatalog,
    path: str,
    issues: list[_Issue],
) -> None:
    _check_ref(choices.actor, catalog.actors, f"{path}.actor", issues)
    _check_ref(choices.target, catalog.targets, f"{path}.target", issues)
    _check_ref(choices.judge, catalog.judges, f"{path}.judge", issues)
    _check_ref(choices.observer, catalog.observers, f"{path}.observer", issues)
    _check_ref(choices.fixture, catalog.fixtures, f"{path}.fixture", issues)
    _check_ref(choices.tasks, catalog.task_selectors, f"{path}.tasks", issues)
    _check_ref(choices.harness, catalog.harnesses, f"{path}.harness", issues)
    _check_ref(choices.execution, catalog.executions, f"{path}.execution", issues)
    _check_ref(choices.limits, catalog.limits, f"{path}.limits", issues)
    for slot in ("actor", "target", "judge", "opening", "trajectory"):
        value = getattr(choices.prompts, slot)
        if value is not None:
            _check_ref(value, catalog.prompts, f"{path}.prompts.{slot}", issues)


def _validate_diagnostics(
    spec: ExperimentSpec,
    issues: list[_Issue],
) -> None:
    if spec.diagnostics.trajectory.enabled and spec.defaults.prompts.trajectory is None:
        _issue(
            issues,
            "DIAGNOSTIC_PROMPT_REQUIRED",
            "defaults.prompts.trajectory",
            "trajectory diagnostics need a trajectory prompt",
        )


def _validate_selection_refs(
    selection: ArmSelection,
    catalog: ComponentCatalog,
    path: str,
    issues: list[_Issue],
) -> None:
    mappings = {
        "actor": catalog.actors,
        "target": catalog.targets,
        "judge": catalog.judges,
        "observer": catalog.observers,
        "fixture": catalog.fixtures,
        "tasks": catalog.task_selectors,
        "harness": catalog.harnesses,
        "execution": catalog.executions,
        "limits": catalog.limits,
    }
    for field, mapping in mappings.items():
        value = getattr(selection, field)
        if value is not None:
            _check_ref(value, mapping, f"{path}.{field}", issues)
    if selection.prompts is not None:
        for slot in ("actor", "target", "judge", "opening", "trajectory"):
            value = getattr(selection.prompts, slot)
            if value is not None:
                _check_ref(value, catalog.prompts, f"{path}.prompts.{slot}", issues)


def _check_ref(value: str, mapping: Mapping[str, object], path: str, issues: list[_Issue]) -> None:
    if value not in mapping:
        _issue(issues, "REF_UNKNOWN", path, f"unknown component {value!r}")


def _validate_prompt_roles(
    choices: ArmChoices,
    prompts: ResolvedPrompts,
    path: str,
    issues: list[_Issue],
) -> None:
    for slot in ("actor", "target", "judge", "opening", "trajectory"):
        prompt = getattr(prompts, slot)
        if prompt is not None and prompt.role != slot:
            _issue(
                issues,
                "COMPAT_PROMPT_ROLE",
                f"{path}.prompts.{slot}",
                f"prompt {prompt.name!r} has role {prompt.role!r}, expected {slot!r}",
            )
    if "{{first_prompt}}" not in prompts.opening.text:
        _issue(
            issues,
            "COMPAT_OPENING_TEMPLATE",
            f"{path}.prompts.opening",
            f"prompt {choices.prompts.opening!r} must contain {{{{first_prompt}}}}",
        )


def _validate_compatibility(
    harness: HarnessSpec,
    execution: ExecutionSpec,
    path: str,
    issues: list[_Issue],
) -> None:
    allowed_execution = {
        "pydantic_ai": {"host"},
        "command": {"host", "container"},
        "agentenv": {"agentenv"},
    }[harness.kind]
    if execution.kind not in allowed_execution:
        _issue(
            issues,
            "COMPAT_HARNESS_EXECUTION",
            f"{path}.execution",
            f"harness kind {harness.kind!r} cannot use execution kind {execution.kind!r}",
        )
    if (
        isinstance(harness, PydanticAIHarness)
        and isinstance(execution, HostExecution)
        and execution.workspace_mode != "shared"
    ):
        _issue(
            issues,
            "COMPAT_WORKSPACE_MODE",
            f"{path}.execution",
            "the in-process harness has no workspace to isolate",
        )


def _select_cases(
    corpus: CorpusSpec,
    selector: TaskSelector,
    selector_name: str,
    issues: list[_Issue],
) -> tuple[Scenario, ...]:
    if selector.kind == "all":
        selected = list(corpus.cases)
    elif selector.kind == "ids":
        known = {case.id for case in corpus.cases}
        for index, identifier in enumerate(selector.ids):
            if identifier not in known:
                _issue(
                    issues,
                    "SELECTION_UNKNOWN_CASE",
                    f"catalog.task_selectors.{selector_name}.ids[{index}]",
                    f"corpus has no case {identifier!r}",
                )
        requested = set(selector.ids)
        selected = [case for case in corpus.cases if case.id in requested]
    else:
        selected = [
            case
            for case in corpus.cases
            if set(selector.include_all).issubset(case.tags)
            and (not selector.include_any or not set(selector.include_any).isdisjoint(case.tags))
        ]
    if selector.exclude:
        excluded = set(selector.exclude)
        selected = [case for case in selected if excluded.isdisjoint(case.tags)]
    if (
        not selected
        and not (selector.kind == "ids" and not selector.ids)
        and not (selector.kind == "tags" and not (selector.include_all or selector.include_any))
    ):
        _issue(
            issues,
            "SELECTION_NO_MATCH",
            f"catalog.task_selectors.{selector_name}",
            "selector matched no corpus cases",
        )
    return tuple(selected)


def _expand_arms(spec: ExperimentSpec, issues: list[_Issue]) -> tuple[ArmSpec, ...]:
    if isinstance(spec.design, ArmsDesign):
        if len(spec.design.arms) > spec.bounds.max_arms:
            _issue(
                issues,
                "BOUND_ARMS",
                "design.arms",
                f"planned {len(spec.design.arms)} arms above max_arms={spec.bounds.max_arms}",
            )
        return spec.design.arms

    empty_axes = [axis for axis, values in spec.design.axes.items() if not values]
    for axis in empty_axes:
        _issue(
            issues,
            "SELECTION_EMPTY",
            f"design.axes.{axis}",
            "a matrix axis needs at least one value",
        )
    ordered_axes: list[AxisName] = [axis for axis in _AXIS_ORDER if axis in spec.design.axes]
    arm_count = 0 if empty_axes else _product(len(spec.design.axes[axis]) for axis in ordered_axes)
    if arm_count > spec.bounds.max_arms:
        _issue(
            issues,
            "BOUND_ARMS",
            "design.axes",
            f"planned {arm_count} arms above max_arms={spec.bounds.max_arms}",
        )
        return ()
    arms: list[ArmSpec] = []
    for values in itertools.product(*(spec.design.axes[axis] for axis in ordered_axes)):
        selection: dict[str, object] = {}
        prompt_selection: dict[str, str] = {}
        name_parts: list[str] = []
        for axis, value in zip(ordered_axes, values, strict=True):
            name_parts.append(f"{axis.replace('.', '-')}-{value}")
            if axis.startswith("prompts."):
                prompt_selection[axis.removeprefix("prompts.")] = value
            else:
                selection[axis] = value
        if prompt_selection:
            selection["prompts"] = prompt_selection
        arms.append(
            ArmSpec(
                name="--".join(name_parts),
                select=ArmSelection.model_validate(selection),
            )
        )
    return tuple(arms)


def _apply_selection(defaults: ArmChoices, selection: ArmSelection) -> ArmChoices:
    updates = {
        field: getattr(selection, field)
        for field in (
            "actor",
            "target",
            "judge",
            "observer",
            "fixture",
            "tasks",
            "harness",
            "execution",
            "limits",
        )
        if getattr(selection, field) is not None
    }
    if selection.prompts is not None:
        prompt_updates = {
            field: getattr(selection.prompts, field)
            for field in ("actor", "target", "judge", "opening", "trajectory")
            if getattr(selection.prompts, field) is not None
        }
        updates["prompts"] = defaults.prompts.model_copy(update=prompt_updates)
    return defaults.model_copy(update=updates)


def _compile_comparisons(
    authored: tuple[ComparisonSpec, ...],
    arms: tuple[SessionArmPlan, ...],
    authored_arm_names: set[str],
    issues: list[_Issue],
) -> tuple[ComparisonPlan, ...]:
    arm_map = {arm.name: arm for arm in arms}
    plans: list[ComparisonPlan] = []
    for index, comparison in enumerate(authored):
        path = f"design.comparisons[{index}]"
        for field in ("baseline", "candidate"):
            name = getattr(comparison, field)
            if name not in authored_arm_names:
                _issue(
                    issues,
                    "COMPARISON_UNKNOWN_ARM",
                    f"{path}.{field}",
                    f"unknown arm {name!r}",
                )
        baseline = arm_map.get(comparison.baseline)
        candidate = arm_map.get(comparison.candidate)
        if baseline is None or candidate is None:
            continue
        baseline_keys = {
            (case.key.corpus.scenario_id, case.key.corpus.repeat_index) for case in baseline.cases
        }
        candidate_keys = {
            (case.key.corpus.scenario_id, case.key.corpus.repeat_index) for case in candidate.cases
        }
        if baseline_keys != candidate_keys:
            _issue(
                issues,
                "COMPARISON_CASE_MISMATCH",
                path,
                "exact pairing requires the same corpus case keys in both arms",
            )
            continue
        ordered_keys = tuple(
            CorpusCaseKey(scenario_id=scenario_id, repeat_index=repeat_index)
            for scenario_id, repeat_index in sorted(baseline_keys)
        )
        plans.append(
            ComparisonPlan(
                name=comparison.name,
                baseline=comparison.baseline,
                candidate=comparison.candidate,
                pairs=tuple(
                    ComparisonPair(
                        corpus=key,
                        baseline=ArmCaseKey(arm=comparison.baseline, corpus=key),
                        candidate=ArmCaseKey(arm=comparison.candidate, corpus=key),
                    )
                    for key in ordered_keys
                ),
            )
        )
    return tuple(plans)


def _reserve_usage(
    arms: tuple[SessionArmPlan, ...],
    diagnostics: ExperimentDiagnostics | None = None,
) -> UsageReservation:
    requests = 0
    output_tokens = 0
    for arm in arms:
        actor = arm.suite.actor
        judge = arm.suite.judge
        target_tokens = _target_output_tokens(arm)
        for case in arm.cases:
            turns = arm.suite.limits_for(case.scenario).max_target_turns
            requests += turns * (1 + actor.model_request_limit) + 1
            output_tokens += (
                turns * (target_tokens + actor.model.options.max_tokens * actor.model_request_limit)
                + judge.model.options.max_tokens
            )
            if diagnostics is not None and diagnostics.trajectory.enabled:
                prefixes = min(
                    turns,
                    diagnostics.trajectory.max_prefixes_per_case,
                )
                requests += prefixes
                output_tokens += prefixes * arm.observer_model.options.max_tokens
    return UsageReservation(
        arm_count=len(arms),
        case_count=sum(len(arm.cases) for arm in arms),
        model_requests=requests,
        output_tokens=output_tokens,
    )


def _target_output_tokens(arm: SessionArmPlan) -> int:
    return arm.target_max_output_tokens_per_turn


def _validate_bounds(
    bounds: ExperimentBounds,
    reservation: UsageReservation,
    issues: list[_Issue],
) -> None:
    checks = (
        ("BOUND_CASES", "max_cases", reservation.case_count, bounds.max_cases),
        (
            "BOUND_REQUESTS",
            "max_model_requests",
            reservation.model_requests,
            bounds.max_model_requests,
        ),
        (
            "BOUND_OUTPUT_TOKENS",
            "max_output_tokens",
            reservation.output_tokens,
            bounds.max_output_tokens,
        ),
    )
    for code, field, planned, maximum in checks:
        if planned > maximum:
            _issue(
                issues,
                code,
                f"bounds.{field}",
                f"planned {planned} above {field}={maximum}",
            )


def _issue(issues: list[_Issue], code: str, path: str, message: str) -> None:
    issue = _Issue(code=code, path=path, message=message)
    if issue not in issues:
        issues.append(issue)


def _raise_issues(issues: list[_Issue]) -> None:
    if not issues:
        return
    rendered = "\n".join(
        f"[{issue.code}] {issue.path}: {issue.message}"
        for issue in sorted(issues, key=lambda item: (item.path, item.code, item.message))
    )
    raise ValueError(f"experiment compilation failed:\n{rendered}")


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _resolve_path(directory: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (directory / path).resolve()


def _load_yaml_model(path: Path, model: type[ModelT]) -> ModelT:
    try:
        payload = cast(object, yaml.safe_load(path.read_text(encoding="utf-8")))
    except OSError as error:
        raise ValueError(f"could not read {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {path}: {error}") from error
    try:
        return model.model_validate(payload)
    except ValidationError as error:
        raise ValueError(f"invalid {model.__name__} in {path}:\n{error}") from error
