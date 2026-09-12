# Pydantic multi-turn evals

This package runs adaptive conversation scenarios through Pydantic Evals. Each scenario begins with
an authored prompt. A simulated user reads each target reply and decides whether to continue,
accept, or stop. A separate final judge grades only the visible conversation. An optional trajectory
observer grades bounded transcript prefixes after completion and cannot steer or fail the run.

The packaged runner supports three local target kinds:

- `pydantic_ai` runs a configured Pydantic AI agent.
- `command` runs any CLI adapter that implements the versioned JSONL protocol below.
- `agentenv` runs fixed turn and verifier commands in one AgentENV sandbox per case.

Harbor is supported separately as a whole-arm runner because a Harbor job owns complete trials,
verification, artifacts, concurrency, and sandbox teardown. It is not invoked once per chat turn.

The target, simulated user, and judge examples use `glm-5.3-flash` through the Z.AI Coding Plan
endpoint. Credentials come from environment variables, never YAML.

## Choose model providers

Every framework-owned model uses the same `ModelSpec`. The actor, target, and judge may select
different providers. The provider and its request settings resolve together before a model call.

```yaml
model:
  name: gpt-5.4
  provider:
    kind: openai
    interface: responses
    api_key_env: OPENAI_API_KEY
  options:
    thinking: high
    max_tokens: 1200
    timeout_seconds: 90
```

Supported provider kinds are `zai`, `openai`, `anthropic`, and `openai-compatible`. Z.AI keeps an
explicit `endpoint_plan` because a Coding Plan credential must use its Coding Plan URL. A custom
OpenAI-compatible route requires `base_url`, accepts either the `chat` or `responses` interface, and
may omit `api_key_env` for a keyless local server.

Common options are `temperature`, `top_p`, `thinking`, `max_tokens`, and `timeout_seconds`. Omitted
sampling options stay absent from the provider request. The resolver does not send Z.AI-specific
fields to other providers.

## Compile and run an experiment

An experiment selects its corpus, actor, target, final judge, trajectory observer, prompts, fixture,
task subset, harness, execution environment, and limits by name. Explicit arms change one or more
choices; a matrix expands several axes deterministically. Compilation validates and embeds every
choice without reading credentials or starting a provider.

```console
uv run multiturn-evals plan experiments/support-ab.yaml --out outputs/support-ab-plan
uv run multiturn-evals experiment-run outputs/support-ab-plan
uv run multiturn-evals resume outputs/support-ab-plan
```

The checked-in support experiment is intentionally a portability manifest: some arms require
OpenAI, Anthropic, and DeepSeek credentials. Runtime preflight reports every missing requirement
before any arm starts. Edit the arm list or supply the named credentials before running it.

Each case gets a durable directory with the frozen input, normal evaluation artifacts, trajectory
diagnostics, and one terminal receipt. Resume skips completed and failed work. A case left with only
`started.json` is marked interrupted instead of replaying a possibly state-changing interaction.
Comparisons pair exact `(scenario_id, repeat_index)` corpus keys across arbitrary named arms.

## Run the low-level two-target API

```console
uv sync --extra tracing
uv run multiturn-evals validate scenarios/support.yaml --target targets/support.yaml
uv run multiturn-evals validate scenarios/support.yaml --target targets/support-candidate.yaml
uv run multiturn-evals compare scenarios/support.yaml \
  --baseline targets/support.yaml \
  --candidate targets/support-candidate.yaml \
  --repeat 3 \
  --out outputs/support-ab
```

The baseline and candidate run sequentially with fresh target, simulated-user, judge, and in-memory
state objects. A failing baseline gate does not skip the candidate. The two arms begin from the same
authored scenario, but later user turns adapt independently to each target's replies.

Cases are paired by the typed `(scenario_id, repeat_index)` created before evaluation. Score and pass
rate deltas are reported as candidate minus baseline; they are descriptive rather than claims of
statistical significance. The compare command exits successfully only when both arm gates pass.

```text
outputs/support-ab/
  gate.json
  comparison.json
  comparison.txt
  baseline/
    gate.json
    report.json
    report.txt
    transcripts.jsonl
    evidence.jsonl
  candidate/
    ...
```

Use `run` for one target:

```console
uv run multiturn-evals run scenarios/support.yaml \
  --target targets/support.yaml \
  --out outputs/support
```

## Run the broad GLM experiment

[`experiments/glm-wide.yaml`](experiments/glm-wide.yaml) expands 20 adaptive scenarios across web chat,
email, ticket, CLI, and API response modes. The scenarios cover 19 task types and nine operating
contexts. The checked-in lanes run `glm-5.3-flash` directly through Pydantic AI, through one local
JSONL child process per case, and through the same JSONL contract in a Docker container. Build the
container harness before running the experiment:

```console
scripts/build_glm_harness_image.sh
```

Inspect the resolved plan before making model calls:

```console
uv run multiturn-evals plan experiments/glm-wide.yaml --out outputs/glm-wide
```

Run or resume its saved plan:

```console
uv run multiturn-evals experiment-run outputs/glm-wide
uv run multiturn-evals resume outputs/glm-wide
```

Each case writes its evaluation artifacts and a terminal receipt before the coordinator marks it
complete. Resume skips completed and failed cases. It marks a previously started case as interrupted
instead of replaying possible external side effects. See [the scale-run design](docs/scale-campaign.md)
for the data and recovery contract.

The manual [`Scale evals`](.github/workflows/scale-evals.yml) workflow runs the same experiment and
uploads the complete run directory even when the quality gate fails.

## Plug in another CLI harness

[`targets/command-example.yaml`](targets/command-example.yaml) shows the command target format. `argv`
is passed directly to `asyncio.create_subprocess_exec`; shell strings are not supported. `cwd`
resolves relative to the target YAML. The child receives only names listed in `inherit_env`, and the
configuration cannot contain environment values.

One process is opened for each scenario and repeat. The runner writes one JSON object per line. The
process first receives `start` and must answer `ready`:

```json
{"protocol":2,"type":"start","session":{"suite":"support-smoke","target":"command-example","scenario_id":"refund-needs-order-number","repeat_index":1,"run_id":"...","comparison_id":null,"arm":null}}
{"protocol":2,"type":"ready"}
```

Each turn contains the complete visible history. Reply IDs must match:

```json
{"protocol":2,"type":"turn","id":1,"messages":[{"role":"user","content":"I need a refund."}]}
{"protocol":2,"type":"reply","id":1,"assistant_text":"What is your order number?","session_id":"optional","evidence":{"route":"refund-intake"}}
```

At the end of the conversation, protocol 2 requires a final exchange before teardown:

```json
{"protocol":2,"type":"finish","outcome":{"transcript":{"exchanges":[]},"decisions":[],"termination":{"kind":"turn_limit_reached","limit":1}}}
{"protocol":2,"type":"finished","completion":{}}
```

The runner then sends `{"protocol":2,"type":"close"}` and closes stdin. There is no protocol 1
fallback. Startup, each reply, stdout line size, retained stderr, and shutdown are bounded. Timeout
and cancellation terminate the whole process group and reap the direct child. The adapter in
[`examples/jsonl_harness.py`](examples/jsonl_harness.py) is a small executable reference.

Only `assistant_text` becomes conversation text. Timing, the optional session ID, the sanitized
executable name, and JSON evidence are written to `evidence.jsonl`. They are excluded from the judge
request and Langfuse metadata. Failed commands add their bounded stderr to the local evidence file;
the traced exception reports only its byte count. Do not put secrets in `argv`; operating-system
process listings may show arguments even though this runner does not trace them.

## Run a target in AgentENV

[`targets/agentenv-example.yaml`](targets/agentenv-example.yaml) defines a native AgentENV target.
The host uses AgentENV's E2B-compatible SDK and reads `E2B_API_URL`, `E2B_SANDBOX_URL`, and
`E2B_API_KEY` from the environment. The YAML stores only those names. Install the optional client
and run it through the normal local A/B coordinator:

```console
uv sync --extra agentenv --extra tracing
uv run multiturn-evals compare scenarios/support.yaml \
  --baseline targets/agentenv-baseline.yaml \
  --candidate targets/agentenv-candidate.yaml \
  --out outputs/agentenv-ab \
  --langfuse
```

Each planned `(scenario_id, repeat_index)` gets one sandbox with a provider TTL. Every turn writes
the complete visible history to a bounded request file and runs the configured `turn` command. The
command writes a bounded response file. After the conversation, the runner writes the terminal
outcome and runs the configured verifier while the sandbox is still alive. It then destroys the
sandbox in a bounded cleanup step. [`examples/agentenv_turn.py`](examples/agentenv_turn.py) and
[`examples/agentenv_verify.py`](examples/agentenv_verify.py) show the two file contracts; bake them
and the evaluated harness into the named AgentENV template.

The transcript judge and environment verifier are independent. A case passes only when the LLM
judge passes and the present environment verifier passes. The verifier reward, reason, sandbox ID,
and structured evidence go to `evidence.jsonl`, never to the judge request. A hard loss of the host
process can still leave a sandbox until its TTL, so this integration does not claim crash recovery.

Run the live SDK check before relying on a new AgentENV deployment:

```console
uv run --extra agentenv python scripts/compat_smoke.py agentenv \
  --template pydantic-multiturn-eval-v1
```

## Run complete arms through Harbor

Harbor configurations live under [`harbor/`](harbor/). Each arm file points to its own Harbor base
job configuration and a task template. The compiler creates one deterministic Harbor task per case,
sets `n_attempts: 1`, and gives Harbor the requested concurrency limit. Baseline and candidate are
always separate sequential jobs:

```console
uv run multiturn-evals harbor-compare scenarios/support.yaml \
  --baseline harbor/baseline.example.yaml \
  --candidate harbor/candidate.example.yaml \
  --repeat 3 \
  --max-concurrency 4 \
  --out outputs/harbor-ab \
  --langfuse
```

The example command pins `harbor[e2b]==0.22.0` in a Python 3.12 `uvx` environment and installs the
dependency-light shim in [`harbor/harbor_adapter`](harbor/harbor_adapter). Harbor 0.22 and Pydantic AI
2.42 require incompatible major versions of the OpenAI Python package, so the shim does not import
the evaluator package. It sends each private actor decision to
`pydantic_multiturn_evals.harbor_actor_worker` in the project's own environment over bounded stdin
and stdout. This preserves the same Pydantic actor without mixing the two dependency graphs.

The private user agent reads the authored scenario, sends the exact first prompt through Harbor's
ACP bridge, adapts later messages, and writes a normalized transcript. The outer process reads only
Harbor's saved `result.json` and that transcript artifact; it never infers success from console
output. It then runs the same transcript-only LLM judge and combines that verdict with Harbor's
named verifier reward.

Harbor's documented simulated-user path currently uses ACP. Use a target agent supported by that
bridge; the checked-in example uses Claude Code. A non-ACP CLI belongs in the direct `command` or
`agentenv` path. Do not configure a native `agentenv` target inside Harbor: Harbor's E2B environment
owns the AgentENV sandbox in this mode, so nesting another sandbox would make teardown and verifier
ownership ambiguous.

Copy the two base job examples and change their target agent or model settings independently for a
real A/B comparison. Keep credentials out of those files. Harbor arm configs explicitly allowlist
which host environment variables reach the Harbor process, and literal secret-looking fields in a
base config are rejected.

## Scenarios and judging

[`scenarios/support.yaml`](scenarios/support.yaml) is the working suite. A scenario owns its first
prompt, a private simulated-user brief, a final judge rubric, and optional turn and timeout limits.
Suite-level settings provide defaults. Unknown YAML fields fail validation and scenario IDs must be
unique.

The whole adaptive conversation is one Pydantic Evals case. The `TranscriptJudge` explicitly removes
the case input and projects the result to visible user and assistant exchanges before calling
Pydantic's `LLMJudge`. Private personas, actor decisions, termination reasons, command evidence, and
stderr do not enter the judge request.

## Langfuse tracing

Tracing is optional:

```console
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...
export LANGFUSE_BASE_URL=https://cloud.langfuse.com
uv run multiturn-evals compare scenarios/support.yaml \
  --baseline targets/support.yaml \
  --candidate targets/support-candidate.yaml \
  --out outputs/support-ab \
  --langfuse
```

Comparison, arm, and scenario executions are separate Langfuse traces correlated by
`comparison_id`; local target turns and instrumented model calls are children of their scenario
trace. Allowlisted metadata identifies the arm, target, execution kind, scenario, repeat, run, and
turn. Harbor normalization also records its job and trial IDs. Judge score, judge pass, environment
reward and pass, arm gate, mean score, pass rate, and comparison deltas are explicit scores. Pydantic
AI instrumentation also exports model prompts and responses when tracing is enabled; configure
Langfuse according to your data policy. GitHub Actions sets `LANGFUSE_RELEASE` to the evaluated
commit. The runner calls Langfuse shutdown in a `finally` block.

Checked-in YAML remains the source of truth. Langfuse dataset sync is deliberately not part of a CI
run, so a remote dataset edit cannot silently change a pull-request evaluation. Two hosted Langfuse
dataset experiment runs can be added later if the Langfuse comparison UI is required.

## State and automation

Conversation state is saved after every complete transition in an `InMemoryStateStore`. Redis is not
needed for one local process or GitHub Actions job and would not make external target effects
exactly-once. A target that changes an external database, queue, browser profile, or service remains
responsible for isolating or resetting that state between arms.

`make check` runs formatting, lint, type checking, and behavior tests. `make compat` recompiles both
Harbor arms twice with no credentials or remote services. The normal
[`ci.yml`](.github/workflows/ci.yml) path needs no credentials. The manual
[`live-evals.yml`](.github/workflows/live-evals.yml) workflow runs both checked-in target configs with
Langfuse tracing and always uploads the complete comparison directory. Configure its `evals`
environment with `ZAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY`.

The manual [`environment-evals.yml`](.github/workflows/environment-evals.yml) workflow selects either
the native AgentENV comparison or the Harbor comparison. It connects to a remote AgentENV deployment;
GitHub-hosted runners do not host the KVM service themselves. Add `E2B_API_URL`, `E2B_SANDBOX_URL`,
and `E2B_API_KEY` to the same environment. The checked-in AgentENV template names must already exist.
The Harbor example also needs `ANTHROPIC_API_KEY` for its Claude Code target. Both paths upload their
artifacts even when the quality gate fails.

The model integration is pinned to Pydantic Evals and Pydantic AI 2.42, the AgentENV client to E2B
2.46.4, and the external Harbor command to Harbor 0.22.0. See the official
[Pydantic Evals overview](https://pydantic.dev/docs/ai/evals/evals/),
[LLMJudge guide](https://pydantic.dev/docs/ai/evals/evaluators/llm-judge/), and
[Langfuse SDK experiments](https://langfuse.com/docs/evaluation/experiments/experiments-via-sdk),
plus [Harbor core concepts](https://www.harborframework.com/docs/core-concepts),
[Harbor simulated users](https://www.harborframework.com/docs/run-jobs/simulated-user), and the
[AgentENV E2B integration](https://kvcache-ai.github.io/AgentENV/latest/integration/e2b.html).
