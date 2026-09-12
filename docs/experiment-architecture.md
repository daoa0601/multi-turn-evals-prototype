# Experiment architecture

## Problem

The evaluator can run adaptive conversations, but its authored suite fixes the actor and judge, its direct target fixes the model and prompt, and its campaign varies target files rather than independent experimental choices. Model construction also assumes Z.AI. We need one scenario corpus that can support A/B tests and bounded ablations of the actor, target, judge, prompts, fixtures, task selection, harness, and execution without changing unrelated inputs.

## Usage

An experiment names reusable components and binds them into arms. A matrix is shorthand for generating explicit arms. The compiler expands every arm before execution and writes one complete non-secret plan.

```yaml
version: 1
name: support-prompt-ab
corpus: ../corpora/support.yaml
catalog: ../components/support.yaml
defaults:
  actor: patient-user
  target: support-model
  judge: strict-judge
  prompts:
    actor: adaptive-user
    target: support-current
    judge: final-quality
    opening: task-opening
    trajectory: turn-progress
  fixture: empty
  tasks: smoke
  harness: adaptive
  execution: local
  limits: short
design:
  kind: arms
  arms:
    - name: baseline
      select: {}
    - name: candidate
      select:
        prompts:
          target: support-concise
  comparisons:
    - name: concise-minus-current
      baseline: baseline
      candidate: candidate
repeats: 3
diagnostics:
  trajectory:
    enabled: true
    max_prefixes_per_case: 4
bounds:
  max_cases: 60
  max_concurrency: 4
  max_model_requests: 900
  max_output_tokens: 650000
  run_seconds: 3600
```

```console
uv run multiturn-evals plan experiments/support.yaml --out outputs/support
uv run multiturn-evals experiment-run outputs/support
uv run multiturn-evals resume outputs/support
```

The same compiler accepts a bounded matrix:

```yaml
design:
  kind: matrix
  axes:
    actor: [patient-user, terse-user]
    target: [support-model, support-command]
    prompts.target: [support-current, support-concise]
```

## Shape

The authored inputs have three owners.

- A `CorpusSpec` owns task facts: the opening message, private actor brief, task-specific rubrics, fixture references, limits, and typed dimensions.
- A `ComponentCatalog` owns reusable model bindings, actors, targets, judges, prompt slots, fixtures, task selectors, harnesses, executions, and limit profiles.
- An `ExperimentSpec` owns arm selection, matrix expansion, comparisons, repeats, diagnostics, and resource bounds.

The compiler resolves those inputs into a `RunPlan`. The plan embeds rendered scenarios, selected component settings, fixture content, comparison membership, execution configuration, and source provenance. It stores credential environment names but never credential values. Planning copies command workspaces into the output directory, and execution reads the saved plan rather than the authored YAML. Provider services, interpreters, and container images remain external resources and need immutable versioning when reproducibility depends on them.

```python
class CorpusCaseKey:
    scenario_id: str
    repeat_index: int


class ArmCaseKey:
    arm: str
    corpus: CorpusCaseKey
    actor: str
    target: str
    judge: str
    prompts: PromptSelection
    fixture: str
    harness: str
    execution: str


class SessionArmPlan:
    kind: Literal["session"]
    name: str
    cases: tuple[CasePlan, ...]
    actor: ResolvedActor
    target: ResolvedTarget
    judge: ResolvedJudge
    prompts: ResolvedPrompts
    fixture: ResolvedFixture
    harness: ResolvedHarness
    execution: ResolvedSessionExecution


class HarborArmPlan:
    kind: Literal["harbor-job"]
    name: str
    cases: tuple[CasePlan, ...]
    actor: ResolvedActor
    judge: ResolvedJudge
    job: ResolvedHarborJob


ArmPlan = SessionArmPlan | HarborArmPlan


class RunPlan:
    version: Literal[1]
    experiment: str
    corpus: ResolvedCorpus
    arms: tuple[ArmPlan, ...]
    comparisons: tuple[ComparisonPlan, ...]
    diagnostics: ResolvedDiagnostics
    bounds: ExperimentBounds
    reservation: UsageReservation
```

`Target`, `TargetSession`, `AdaptiveActor`, and the state-owning conversation runner remain unchanged. Harbor remains a whole-job owner rather than a turn-level target.

Model and request settings resolve together. Pydantic AI's `Model` remains the runtime interface.

```python
class BoundModel:
    model: Model
    settings: ModelSettings
    identity: ResolvedModel
    required_environment: tuple[str, ...]


def bind_model(spec: ModelBindingSpec, environment: Mapping[str, str]) -> BoundModel: ...
```

Each provider variant owns its endpoint, credential reference, accepted options, and Pydantic model class. The first supported variants are Z.AI, OpenAI, Anthropic, and custom OpenAI-compatible chat. An unsupported option fails before provider I/O. There is no ambient fallback to another route.

Trajectory assessment runs after the primary result is complete. It judges bounded transcript prefixes with a progress rubric. It receives no later turns, private actor brief, actor decisions, target evidence, final score, or environment verdict. Its failures are stored as diagnostic failures and cannot change the primary gate.

Static compilation collects unknown references, empty selections, duplicate keys, incompatible component combinations, comparison case mismatches, and work above configured bounds. Runtime preflight checks credentials, commands, fixture materialization, and external prerequisites for every arm before any arm starts. A missing credential never removes an arm silently.

## Synthesis decision

Three designs were compared. The typed catalog and explicit matrix scored best because each experimental choice has one owner and no caller must reconstruct inheritance or plugin registration order. The chosen design adopts separate corpus and arm keys, stable preflight issue codes, usage coverage, and requested-versus-reported model identity from the resolved-graph design. It also adopts bounded prefix selection and an expanded-work preview from the layered-profile design.

The design rejects a Cordis-style runtime plugin container. [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) uses that structure for a long-lived application with dynamic composition. This evaluator resolves one immutable run plan, so closed provider and execution unions are smaller and easier to audit. The local reference checkout used for this decision was commit `c291e79`; no source was copied.

## Tradeoffs accepted

- We accept more named component definitions in exchange for one-factor changes that reviewers can identify directly.
- We accept larger plan files in exchange for execution that does not depend on edited source YAML.
- We accept closed provider and execution variants in exchange for strict validation and readable ownership.
- We record unavailable usage as unknown rather than estimating it from transcript length or token limits.
- We keep interrupted work terminal by default because replaying a state-changing case may repeat side effects.

## Alternatives considered

A generic component registry was rejected because it exposes registration, capability, and runtime graph rules to an experiment that only needs a frozen plan. Profile inheritance was rejected because a reviewer would have to reconstruct precedence to learn what changed. Keeping separate suite, comparison, campaign, and Harbor authoring systems was rejected because no file would own the effective experiment.

## Open questions and risks

- Which fixture forms can each target and execution type apply without pretending that local files reset external state?
- Which providers report reliable model identity and reasoning-token usage through the pinned Pydantic AI version?
- Should a judge ablation be allowed to gate a target comparison, or should it produce a grading-policy comparison only?
- What command protocol fields are required to apply target prompts and fixtures without exposing private actor or judge inputs?

## Implemented boundary

The provider binding, corpus and catalog compiler, exact arm pairing, frozen plan, session-arm runner,
durable case receipts, runtime preflight, prompt and fixture transport, and bounded non-gating
trajectory assessment are implemented. The former scale-campaign compiler was deleted after its
60-case GLM workload moved to this path.

Whole-job Harbor arms and provider-reported actual usage remain separate follow-up boundaries. The
plan does not call Harbor a session harness, and it records conservative reservations rather than
presenting token-limit estimates as actual use.

The implementation migrates affected callers with each unit and removes superseded APIs after their last caller moves. It does not add schema fallbacks.
