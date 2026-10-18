# Pydantic multi-turn evals

This package runs the same cases and evaluators against different execution harnesses. A harness
may hold a conversation, call an SDK directly, run a benchmark job, manage a sandbox, or delegate
to a service. The campaign runner does not know which one it received.

The shared contract is small:

```text
Case -> Harness -> Evidence -> Evaluator(s) -> TrialResult
                     |
                     +-> local typed trace -> optional Langfuse export
```

`Evidence` keeps three things distinct:

- `public` messages and output that an evaluator may inspect;
- native checks produced by an authoritative environment verifier;
- artifact paths inside the isolated trial directory. Non-model evaluators receive resolved,
  validated handles to those files.

A harness executes. An evaluator decides. A trace exporter observes after the result is durable.
None of those roles may claim the work of another.

## What is implemented

The included harnesses cover four materially different execution shapes:

- `ConversationHarness` composes an adaptive actor with a stateful target.
- `FunctionHarness` calls an in-process SDK or application directly.
- `OneShotCommandHarness` exchanges one bounded JSON request and response with an external job.
- `CommandTarget` keeps a JSONL child process alive for an entire conversation.

Five vertical adapters use that same contract:

- `AgentEnvTarget` keeps one E2B-compatible sandbox across adaptive turns, verifies inside the
  sandbox, downloads bounded artifacts, then destroys it.
- `HarborHarness` runs one task and one native attempt with retries and inner concurrency disabled.
  Its saved verifier reward, not the CLI exit code, decides the native check. It imports public ATIF
  dialogue, tool activity, and reported model usage, and supports custom agent import paths.
- `TemplarHarness` stages an artifact, submits one workflow, persists remote IDs, reads incremental
  events, accepts only an applied result, and cancels on timeout.
- `SweBenchHarness` accepts a bounded external solver or the official gold patch, then reads
  `resolved` from the official saved instance report.
- `Tau3TextHarness` runs one text simulation in a pinned tau3-bench environment, reads its official
  reward and trajectory, and imports participant generations and tool calls into the common trace.

`providers.py` supplies Pydantic AI actor, target, and evaluator components plus explicit OpenAI,
Anthropic, OpenAI-compatible, and Z.AI model binding. The Z.AI coding endpoint never falls through
to an ambient provider default. Every Pydantic model request writes a typed fact to the local trace
and is copied into `TrialResult.model_calls`, including its component role, requested binding,
reported model, provider interface, sampling settings, duration, usage, provider response ID, and
completed, failed, or cancelled status. These operational receipts are not evaluator evidence.

In a conversation, the actor may use private case data but the target receives only messages and
trial metadata. Criteria, required checks, and private fixtures are not sent to the system being
evaluated. A whole-job adapter receives the case as trusted orchestration input and must keep its
rubric away from the target it launches.

None of these names appears in the campaign runner. Harbor 0.1.44, SWE-bench 5.0.2, and tau3-bench
1.0.1 have live verification scripts. AgentENV has no live claim without service credentials.
Templar has a real HTTP contract test but no model-quality claim.

## Install and prove the local system

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required.

```console
make install
make check
```

`make verify` runs one case and one evaluator through persistent conversation processes, a direct
function, and one-shot command jobs. It inspects the artifacts, compares exact case/sample pairs,
then resumes the campaign and proves that no terminal trial was replayed. The fixture is evidence
for these local boundaries only; it is not evidence that an external benchmark works.

Run the opt-in Docker integrations separately:

```console
make verify-harbor
make verify-harbor-multiturn
make verify-swebench
make install-tau3
make verify-tau3
```

The Harbor proof runs an oracle, a no-op agent, and cancellation through the installed Harbor CLI
and Docker. The SWE-bench proof evaluates the official gold patch and an irrelevant nonempty patch
for one pinned verified instance. These prove adapter plumbing and native verifier authority, not
model quality.

The Tau3 proof installs the pinned upstream checkout in its own Python 3.12 environment. It runs
the official mock state-change task and airline policy-refusal task against a deterministic local
OpenAI-compatible model server, then requires both native rewards to pass. This proves the real
tau3-bench orchestration, environment, evaluator, result import, and concurrent process boundary;
it does not claim remote-model quality.

## Run a campaign

Campaigns are Python values, so executable components are bound directly instead of recovered from
strings spread across catalogs and YAML files.

```console
uv run multiturn-evals run examples.offline_campaign:campaign \
  --directory outputs/offline \
  --workers 3 \
  --compare baseline candidate
```

The example exits with status 1 on purpose because the baseline fails quality. Read the saved report
without running any trial:

```console
uv run multiturn-evals report outputs/offline --compare baseline candidate
```

Run the same campaign and directory again to resume. Completed, failed, and interrupted trials are
not replayed. `--retry-interrupted` is explicit because the first attempt may have changed a remote
system before the process stopped.

Exit status 0 means all trials completed and passed. Status 1 means execution completed but at least
one trial failed a native check or evaluator criterion. Status 2 means a trial failed to execute or
was interrupted.

## Run at scale

Attach the same resource name to variants that compete for the same capacity:

```python
Variant(
    name="candidate",
    harness=harbor,
    evaluators=evaluators,
    resources=(ResourceClaim(name="docker", slots=1),),
)
```

Allocate that resource for one invocation and, when needed, select a static shard:

```console
uv run multiturn-evals run my_campaigns:campaign \
  --directory outputs/part-0 --workers 16 --capacity docker=4 --shard 0/4
```

The scheduler starts the oldest trial whose complete resource claim fits. Capacity waiting occurs
before the attempt directory and case deadline, and a permit remains held through harness cleanup.
Claims are shared across variants, so Harbor and SWE-bench can draw from one local Docker pool.

Run every shard in a separate directory, then merge after all roots are closed:

```console
uv run multiturn-evals merge my_campaigns:campaign \
  outputs/part-0 outputs/part-1 outputs/part-2 outputs/part-3 \
  --compare baseline candidate
```

All variants for one `(case, sample)` pair stay on one shard. Merge rejects missing, duplicate,
foreign, or incompatible results. Separate hosts must use separate roots and divide any
account-wide quota themselves; this is static distribution, not a lease service.

### Pydantic AI concurrency run

Run 1,000 trials through `PydanticActor`, `PydanticTarget`, `ConversationHarness`, and
`PydanticJudge` with 128 concurrent workers:

```console
uv run python -m scripts.run_pydantic_scale \
  --cases 100 --samples 5 --workers 128 --model-delay-ms 20
```

The runner uses two distinct target profiles, executes four Pydantic model requests per trial,
checks every authored task reaches the target response, and requires each saved trial receipt to
contain two actor calls, one target call, one judge call, and their reported usage. It then resumes
the same campaign and requires zero model calls. The retained `scale-summary.json` records
throughput, peak model concurrency, in-memory and durable model-call counts, aggregate usage, pass
count, receipt count, trace count, Python version, Pydantic AI version, and source-tree state.

The default models are Pydantic AI `FunctionModel` instances with structured actor and judge output.
This measures the complete local Pydantic AI orchestration and persistence path. It does not claim
throughput for OpenAI, Anthropic, Z.AI, or another remote provider. Swap the explicit `BoundModel`
objects for provider bindings when measuring a configured provider.

### Harbor benchmark canaries

Run one registry task from five different Harbor domains under a shared two-container capacity:

```console
uv run python -m scripts.run_harbor_benchmark_pack
```

The default pack covers Terminal-Bench Sample, BFCL, DABstep, CooperBench, and GAIA. It pins one
exact task ID per dataset, preserves each native Harbor result in the common attempt directory, and writes a
`pack-summary.json` under `outputs/harbor-benchmark-pack/`. CooperBench's published images require
`linux/amd64`, so the pack declares that Docker platform and records it in the campaign manifest.

The default `oracle` run is an adapter and verifier canary, not a model score. Use `--agent`,
`--model`, and explicit `--inherit NAME` arguments for a configured model run. Environment variable
values are passed only to the Harbor process; the manifest stores their names.

### Harbor multi-turn integration

Harbor remains a whole-job harness. It owns the agent, sandbox, and native verifier. The campaign
owns samples, comparisons, evaluator criteria, receipts, and resume. The adapter imports ATIF text,
tool observations, and reported model usage into the common trace. Transcript exposure to judges
requires `public_trajectory=True`. Set it only for tasks whose user, agent, and tool text is safe to
send to the configured judge. It is an explicit trust choice, not a secret scrubber. ATIF system
steps, separate reasoning fields, and native metadata stay out of judge input. Per-call durations
are `null` when the native trajectory does not report them.

Run the complete local integration with Harbor 0.1.44 and Docker:

```console
make verify-harbor-multiturn
```

The default run executes six trials through a custom Harbor agent. Each trial makes four HTTP
requests to a deterministic local model fixture and uses real Docker commands to read state, write
an answer, and read it back. `PydanticJudge` uses deterministic model responses to check the imported
transcript. The script proves native-verifier failure and evaluator failure independently, exact
sample pairing, resume without new model calls, export-error isolation, and container cleanup.
It saves common receipts, native Harbor artifacts, traces, and `summary.json` under
`outputs/harbor-multiturn/`. Use `--directory`, `--samples`, `--workers`, and `--docker-capacity` to
configure a fresh verification run. The fixture script checks resume while its local servers remain
alive; rerunning that script requires a fresh directory because server endpoints are part of the
campaign identity. The real-provider CLI example below also supports resume across process restarts.
The separate `make verify-harbor` command tests cancellation.
These are integration checks, not model-quality scores. The manual **Harbor integration** GitHub
Actions workflow runs the Docker checks and retains the multi-turn artifacts.

For a bounded real-model check, set `ZAI_API_KEY` and run:

```console
uv run multiturn-evals run examples.harbor_zai_campaign:campaign \
  --directory outputs/harbor-zai --workers 2 --capacity docker=1 --compare noop candidate
```

This runs Harbor's `terminus-2` with `zai/glm-4.7` and a Pydantic judge against the Z.AI coding
endpoint. It makes real provider requests. The `noop` baseline is expected to fail quality, so a
successful comparison normally exits with status 1. Reuse the same directory to resume without
replaying completed trials. This small stateful task is a wiring check, not a general model benchmark.

Custom agents use `HarborHarness(agent_import_path="module:AgentClass", agent_arguments={...})`.
The module must be importable in Harbor's separate environment. Pass credentials through named
`inherit_environment` entries, never agent arguments. Harbor 0.1.44 is the verified version;
`harbor_version` records the declared dependency and does not install or select it.

### Tau3 text runs

`Tau3TextHarness` treats tau3-bench as a whole-job harness because tau3-bench owns the simulated
user, agent, domain environment, tools, and native evaluation. Install its pinned dependency
environment once, then run the two-domain verification:

```console
make install-tau3
make verify-tau3
```

Exercise the same real boundary concurrently:

```console
uv run python -m scripts.verify_tau3 \
  --directory outputs/tau3/two-domain \
  --samples 10 --workers 8
```

The adapter itself accepts arbitrary text domains, task sets, splits, task IDs, agent/user
implementations, models, and model arguments. Each core trial launches one external tau3-bench
simulation with its inner concurrency and retries disabled. Provider credentials must be named in
`inherit_environment`; putting credentials in persisted model arguments is rejected.

## Add a harness

Implement one method and give the component a stable manifest definition:

```python
class MyHarness:
    component = ComponentDefinition(kind="my-harness", version="1")

    async def run(self, case, trial, trace) -> Evidence: ...
```

The harness owns its external lifecycle, including cleanup and native verification. It returns
public output, native checks, and relative artifact paths. It does not run quality evaluators or
write campaign receipts. The same `Case`, evaluator tuple, runner, report, comparison, and trace
schema then work without a branch in core code.

External commands inherit no ambient environment variables. Set `inherit_environment=("NAME",)` on
the target or job for each variable it needs. The manifest records the names, never their values.
Do not put credentials in command arguments.

For a separate dependency environment, use `OneShotCommandHarness` as the bounded process boundary.
For an interactive program, implement a target for `ConversationHarness`. Use `FunctionHarness`
when an SDK already exposes the whole task as one call.

## Langfuse

Every attempt writes `events.jsonl` before any external export. Enable Langfuse with its normal
environment variables, the non-secret `LANGFUSE_PROJECT_ID` from the project's settings or URL,
and `--langfuse`:

```console
uv run multiturn-evals run examples.offline_campaign:campaign \
  --directory outputs/offline-langfuse \
  --langfuse
```

Langfuse receives one trial observation, generation observations for model-call facts, span
observations for the remaining typed events, and evaluator scores. Generations carry model,
provider, sampling, usage, cost, status, and response-ID metadata from the durable receipt. The
synchronous SDK runs in a child process that the export timeout can terminate. An export failure is
written to `trace-export-errors.json`; it cannot change the already-written trial result. The runner
writes `report.json` before bounded exports begin. It stores one stable export identity and
successful exporter definitions per attempt, scoped to the base URL and project ID. A resume retries
pending exports without replaying the harness. Observation and score IDs stay stable across retries
so previously delivered records are updated. Keep `LANGFUSE_PROJECT_ID` consistent with the project
selected by the credentials; rotating keys for that project does not require changing its ID.

See [the core design record](docs/design.md) and [the adapter design](docs/adapters.md) for ownership,
receipts, scale boundaries, and adapter acceptance rules.
