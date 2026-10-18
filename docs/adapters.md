# Adapters and scale

## Problem

One campaign must keep the same cases, evaluators, receipts, comparisons, and traces while its
harness changes. AgentENV, Harbor, Templar, SWE-bench, and tau3-bench have different dependency
graphs and lifecycle rules, so the shared abstraction stops at one operation:

```python
evidence = await harness.run(case, trial, trace)
```

The campaign remains the only scheduler and the only writer of `TrialResult`. An adapter owns one
trial's external execution, native verification, artifacts, cancellation, and cleanup.

## Usage

Resource claims describe real shared capacity without naming a backend in the runner. Two variants
that use the same Docker host claim the same resource.

```python
campaign = Campaign(
    name="repairs",
    revision="1",
    cases=cases,
    variants=(
        Variant(
            name="baseline",
            harness=baseline,
            evaluators=evaluators,
            resources=(ResourceClaim(name="docker", slots=1),),
        ),
        Variant(
            name="candidate",
            harness=candidate,
            evaluators=evaluators,
            resources=(ResourceClaim(name="docker", slots=1),),
        ),
    ),
)

report = await run(
    campaign,
    directory=Path("runs/repairs"),
    workers=16,
    capacities={"docker": 4},
)
```

Admission happens before an attempt starts, so waiting for capacity does not consume the case's
execution budget. A permit remains held until the harness has completed cleanup and the trial has a
terminal receipt.

Static shards allow several hosts to own separate roots. Every variant for one `(case, sample)` pair
stays on one shard.

```python
await run(
    campaign,
    directory=Path("runs/repairs/shard-1"),
    workers=8,
    capacities={"docker": 2},
    shard=Shard(index=1, count=4),
)

combined = merge_reports(
    campaign,
    tuple(Path(f"runs/repairs/shard-{index}") for index in range(4)),
)
```

Each shard has its own filesystem lock. This is static work allocation, not a distributed lease or
a global quota service. Operators must divide external account capacity across hosts.

## Shape

`ResourceClaim` and `Shard` are the only new core concepts. Claims live on `Variant`; capacities are
per invocation. Shard ownership is a pure function of authored case order and sample number. The
runner scans pending work for the first trial whose complete set of claims fits, which prevents a
full backend pool from idling unrelated capacity.

Named adapters are vertical modules. Each implements the existing `Harness` protocol and keeps its
native request, result, and checkpoint schemas private. There is no backend enum, backend dispatch,
or adapter scheduler in core. External CLI dependencies run in their own installed environment.
Remote adapters persist a safe locator in the attempt directory immediately after creation and do
not retry an uncertain state-changing request.

A Harbor dataset binding may name one exact registry task or request one bounded task without
prelisting the dataset. Harbor still runs with `--n-attempts 1`, `--n-concurrent 1`, and no retries;
the campaign runner owns outer concurrency. `docker_platform` is an explicit adapter setting for
datasets such as CooperBench whose published images do not include every host architecture. It is
passed as `DOCKER_DEFAULT_PLATFORM` only to Harbor and stored in the component definition.
Use explicit `HarborDatasetTask.task` bindings for comparisons. The adapter checks the returned task
identity. The benchmark pack pins task IDs; an omitted task remains available for dataset canaries.

Harbor 0.1.44 is the verified CLI version. The adapter imports text from ATIF v1.0 through v1.6,
including tool observations and bounded continuation chains. Images and subagent trajectories are
not read. Copied-context steps do not duplicate dialogue, tool actions, or model calls.

Imported transcripts are trace-only by default. Set `public_trajectory=True` only when all user,
agent, and tool text is safe for the configured judge. This opt-in is not a sanitizer. System steps,
separate reasoning fields, native rewards, model identity, and usage stay out of judge input.
Reported per-step usage becomes a model-call receipt with `provider="harbor"` and
`source="imported-receipt"`. Missing input or output token counts leave usage absent. Unreported
per-call duration remains `null`.
Native errors remain primary when an auxiliary trajectory is damaged; valid partial evidence is
retained.

Native result and trajectory reads reject symbolic links, including linked parent directories.
`output_bytes` caps both captured stdout and `result.json`; `stderr_bytes` caps stderr;
`trajectory_bytes` caps the entire ATIF chain. These checks run before native data is imported.
Custom agents use `agent_import_path` and JSON-valued `agent_arguments`. Credentials belong in
explicit `inherit_environment` entries, not agent arguments or a saved campaign manifest.

Artifacts are regular files below the attempt directory. The runner rejects path escapes, symlinks,
too many files, oversized files, and excessive total bytes before evaluators see them. Limits are
part of the campaign manifest because changing them can change a result.

```text
campaign.py             scheduling, admission, receipts, exact report
shards.py               pair ownership and exact read-only merge
adapters/agentenv.py    one sandbox, turns, verifier, destroy
adapters/harbor.py      one Harbor task and saved native result
adapters/templar.py     stage, submit, incremental events, accepted result, cancel
adapters/swebench.py    solver patch and official fresh verification
adapters/tau3.py        one text simulation, official reward, trajectory import
```

## Synthesis decision

The base design uses vertically owned adapters and preserves the current result path. From the
competing designs it takes work-conserving atomic admission and stronger artifact validation. It
does not take a public worker SDK, a second recovery result family, SQLite indexing, or a breaking
streaming `Report`; those changes widen ordinary APIs before measured scale requires them.

## Tradeoffs accepted

- We accept process startup for dependency-conflicting CLIs in exchange for isolation and killable
  local work.
- We accept static shard imbalance in exchange for separate writers and exact ownership without a
  lease service.
- We accept an in-memory final report for now; shard execution and traces remain bounded enough for
  the intended first scale tier.
- We accept adapter-specific native schemas in exchange for real verifier semantics instead of a
  fake universal status.
- We accept explicit retry of interrupted work in exchange for never duplicating uncertain external
  side effects silently.

## Alternatives rejected

- Delegating the campaign to Harbor or another native batch runner creates a second scheduler and a
  second definition of attempts, receipts, and pairing.
- A universal `create/poll/verify/cleanup` interface is temporal and shallow. It makes core learn
  every backend lifecycle instead of hiding it behind `Harness.run`.
- A shared run directory across hosts is not safe distributed execution. Separate shard roots make
  ownership explicit.
- A database lease and global quota service are deferred until static shards and per-host capacity
  are measured and shown insufficient.
- A SQLite report rewrite is unrelated to proving the first real adapters and would break the
  current `Report.results` API.

## Proof standard

Contract tests may use small local processes or HTTP servers, but they are described as boundary
tests, not proof that a named service works. A named adapter becomes validated only after a real run
preserves its native verifier output, common receipt, artifacts, trace, and cleanup evidence.

- Harbor: `make verify-harbor` checks native pass/failure and cancellation through Harbor 0.1.44
  and Docker. `make verify-harbor-multiturn` adds a model-driven multi-turn fixture, Pydantic judging,
  imported ATIF messages/tools/usage, exact sample comparisons, resume, and export-error isolation.
  Its deterministic model fixtures prove integration, not remote-model quality.
  `examples.harbor_zai_campaign` is an opt-in, bounded real-provider check.
- AgentENV: a real sandbox with multiple turns, verifier-before-destroy, and cancellation. This
  requires service credentials.
- Templar: a real accepted result plus confirmed cancellation. Its current API has no artifact or
  completed-run deletion endpoint, so the adapter must not claim that cleanup.
- SWE-bench: one pinned official instance whose saved per-instance report supplies `resolved`, with
  solver and verifier environments kept separate.
- Tau3: one pinned external Python 3.12 checkout whose official result supplies the native reward,
  participant messages, tool calls, source revision, and model usage. The checked-in worker uses
  tau3-bench's public text-runner API and writes directly into the common attempt directory.

## Red-flag screen

The public surface stays small; native wire types do not escape their adapter; modules own backend
knowledge rather than lifecycle phases; and no wrapper exists merely to forward arguments. One
trial still has one `Evidence` and one terminal result.
