# Harness-agnostic evaluation design

## Goal

One campaign must be able to run against a conversational agent, an in-process function, an
external benchmark runner, a sandbox service, or a future harness without replacing its cases,
evaluators, scheduler, receipts, comparison, or trace schema.

The design therefore starts from evidence, not from the lifecycle of one existing harness:

```text
                         +------------------+
Case + Trial ----------> | Harness          |
                         | owns execution   |
                         +--------+---------+
                                  |
                               Evidence
                                  |
                         +--------v---------+
                         | Evaluator(s)     |
                         | own judgment     |
                         +--------+---------+
                                  |
                              TrialResult

TraceRecorder observes the path. Exporters run only after TrialResult is stored.
```

There is no harness-kind switch in the runner. A named backend does not get a second scheduler,
result family, gate, or report.

## Domain values

`Case` contains the task, private component data, evaluator criteria, required native check names,
and a trial budget. `Case.data` is available to a trusted harness or simulated user. A conversation
target receives only trial metadata and the messages chosen by the actor; it never receives the
case, rubric, required check names, or private fixture. `Case.data` is not copied into
`Evidence.public` and PydanticJudge does not receive it.

`Harness.run(case, trial, trace)` returns `Evidence`:

- `PublicOutcome` contains messages and JSON output suitable for an evaluator.
- `NativeCheck` reports an authoritative fact from the environment that executed the task.
- `Artifact` names a file inside the attempt directory.

An evaluator receives `EvaluationInput`: the task, criteria, public outcome, and validated resolved
artifact handles. It cannot read `Case.data` or native checks. PydanticJudge deliberately uses only
the task, criteria, and public outcome; a deterministic artifact evaluator may open the provided
files. The runner requires exactly one judgment for every authored criterion and every named native
check from the harness. A false check or judgment is a completed quality failure. Missing evidence
is an execution failure.

`TrialKey(campaign, revision, case, variant, sample)` is the only trial identity. `attempt` records
an explicit retry after interruption and is not a new sample.

Pydantic actor, target, and judge calls record `ModelCall` values through the same attempt trace.
`TrialResult.model_calls` retains those values even when the call fails or is cancelled. A model
call names its role and operation, requested binding, reported provider/model, sampling settings,
duration, usage, response ID, and terminal status. This is operational provenance for the trial;
evaluators receive none of it.

The terminal family is exhaustive:

- `Completed` retains evidence and judgments. It may pass or fail quality.
- `Failed` retains a phase, actionable error, and any evidence already produced.
- `Interrupted` records uncertain work and is not retried by normal resume.

## Ownership

| Owner | Owns | Does not own |
| --- | --- | --- |
| Campaign runner | trial enumeration, bounded concurrency, attempts, receipts, resume, exact pairing, exit status | target lifecycle, model judgment, backend dispatch |
| Harness | environment startup, execution, native verification, cleanup, artifacts | quality policy, campaign receipts |
| Evaluator | authored criteria over public evidence | private case data, environment side effects, scheduling |
| Trace recorder | local ordered facts for one attempt | quality or execution decisions |
| Trace exporter | best-effort copy to Langfuse or another sink | the durable result |

`Variant` binds one harness and a tuple of evaluators. This makes target changes, harness changes,
and evaluator ablations explicit values. Reusing the same evaluator object across variants avoids
configurations that merely share the same string name while resolving to different code.

## Included harness shapes

The repository proves the contract across different ownership boundaries.

`ConversationHarness` owns an adaptive loop. Its actor sees the case and public message history. Its
target remains open while turns run, performs native verification in `finish()`, and closes after
verification. The harness returns evidence; it does not contain a judge.

`CommandTarget` implements that target contract over a persistent JSONL process. The protocol has
start, turn, finish, and close requests. The start request contains trial metadata, not the case.
Replies are bounded and validated. Timeout cleanup kills the process group, not merely the immediate
Python object.

`FunctionHarness` is the direct boundary for an SDK or application that already exposes the whole
task. The function receives the same case, isolated trial directory, seed, and trace recorder as any
other harness.

`OneShotCommandHarness` is the dependency and process boundary for benchmark runners. It passes one
JSON request, requires explicit `Evidence` in the response, bounds time and output, and preserves
declared artifacts. A zero process exit is never treated as a quality pass by itself.

Command processes receive only `MTE_TRIAL_DIRECTORY` plus explicitly named inherited environment
variables. Their manifest definitions record variable names and do not contain values. Both command
boundaries kill the whole process group during timeout, cancellation, or cleanup, including children
left behind after the immediate process exits.

`AgentEnvTarget`, `HarborHarness`, `TemplarHarness`, `SweBenchHarness`, and `Tau3TextHarness`
implement these boundaries without backend dispatch in core. AgentENV composes with
`ConversationHarness`; the other four own a complete external job. Harbor, SWE-bench, and
tau3-bench stay in their own installed environments. Whole-job target requests omit criteria,
required-check names, and private case data. See [the adapter design](adapters.md) for lifecycle and
proof status.

## Receipts and resume

The campaign manifest stores authored cases plus each harness and evaluator definition. Reusing a
directory with a different definition is rejected.

Each attempt owns this directory:

```text
trials/<case>/<variant>/sample-0001/attempt-0001/
  started.json
  events.jsonl
  result.json
  trace-exports.json        # stable export identity and completed exporters
  trace-export-errors.json  # only when an exporter fails
  ... harness artifacts ...
```

The runner uses one work-conserving local scheduler and one filesystem lock. A terminal receipt is
reused on resume. If
`started.json` exists without `result.json`, the next run writes `Interrupted`. Only
`--retry-interrupted` creates the next attempt. This avoids silently repeating a remote side effect.

Artifacts must resolve to regular non-symlink files inside the attempt directory and stay within the
manifest's file-count and byte limits. Invalid evidence never reaches an evaluator.

Variants may declare generic resource claims. Invocation capacity is shared across variants, and
admission occurs before attempt creation and the case deadline. Static shards keep all variants for
one case/sample pair in one separately locked root. Exact merge rejects gaps and duplicates. These
are local admission and static distribution, not a distributed lease or global quota.

Comparisons join results by the exact `(case, sample)` pair. They do not zip lists, substitute a
different sample, or drop missing arms.

## Tracing

Harnesses and model-backed components record typed facts: lifecycle, message, tool, check,
artifact, resource, model call, and log. The local JSONL trace is part of the attempt record and is
written in order with sequence numbers and elapsed time.

External exporters receive the completed local events and durable result. `LangfuseExporter` maps
one attempt to a root evaluator observation, model-call facts to generation observations, remaining
facts to child spans, and judgments to criterion scores. Generations carry model, provider,
sampling, token usage, known cost, status, and response-ID metadata; request and response bodies are
not duplicated into the operational receipt. The runner writes `report.json` before starting a
separate, bounded export phase. The synchronous Langfuse SDK runs in a child process, so timeout or
cancellation can terminate it. The runner persists a random export identity before the first call
and records each successful exporter. A resume retries pending exports without replaying execution.
An unavailable collector can add an export diagnostic but cannot turn a pass into a failure,
prevent the report from being written, or occupy an execution worker indefinitely.

## Pydantic Evals boundary

Pydantic models define all serialized values, and Pydantic AI supplies concrete actor, target, and
judge components. The package does not put `pydantic_evals.Dataset` in charge of execution because
that would create a second trial lifecycle and result family beside interruption-safe receipts.

A future Pydantic Evals adapter may consume completed evidence as an evaluator or export completed
results. It must not become another scheduler inside a campaign run.

## Test policy

Tests exercise public behavior:

- a persistent child process carries state across adaptive turns and is reaped on timeout;
- native verification finishes before target cleanup;
- one case and evaluator run through conversation, direct, and one-shot harnesses;
- native checks and evaluator criteria fail independently;
- completed and failed Pydantic calls survive in trial receipts with usage and failure details;
- artifacts exist inside real attempt directories;
- terminal work is not replayed on resume;
- interrupted work requires an explicit retry;
- comparisons use exact case/sample identities;
- trace export failure cannot change a durable result;
- the CLI preserves quality and execution exit status.

The tests do not assert file counts, class counts, coverage thresholds, legacy API shape, scripted
organization statistics, or names of nonexistent integrations. A deterministic local fixture proves
only the boundary it actually runs.

## Removed design

The rebuild removes the overlapping scenario, experiment, whole-job, Harbor, and scale plan
families; component catalogs; phase-named scheduler files; legacy storage and comparison layers;
pass@k claims without a sampling contract; syntax-policing checks; and backend modules whose tests
never reached the named service.

The API and artifact format are intentionally breaking. Preserving compatibility would keep the
same competing ownership models that caused the rewrite.

## Acceptance rule for a named adapter

A backend adapter is ready to claim support only when a test or saved trial demonstrates all of the
following against the real backend:

1. the backend starts from an isolated task state;
2. a multi-turn or whole-job action reaches the real target;
3. the backend's authoritative verifier produces native evidence;
4. artifacts and partial failure evidence survive cleanup;
5. cancellation or timeout reaches the backend-owned resource;
6. the result uses the common receipt, report, comparison, and trace path.

Until then, an adapter may exist as experimental code, but its local fixture must not be described
as proof of the external system.
