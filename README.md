# Pydantic multi-turn evals

This package runs adaptive conversation scenarios through Pydantic Evals. Each scenario begins with
an authored prompt. A simulated user reads each target reply and decides whether to continue,
accept, or stop. A separate LLM judge grades only the visible user and assistant text.

The packaged runner supports two target kinds:

- `pydantic_ai` runs a configured Pydantic AI agent.
- `command` runs any CLI adapter that implements the versioned JSONL protocol below.

The target, simulated user, and judge examples use `glm-5.3-flash` through the Z.AI Coding Plan
endpoint. Credentials come from environment variables, never YAML.

## Run the checked-in A/B comparison

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

## Plug in another CLI harness

[`targets/command-example.yaml`](targets/command-example.yaml) shows the command target format. `argv`
is passed directly to `asyncio.create_subprocess_exec`; shell strings are not supported. `cwd`
resolves relative to the target YAML. The child receives only names listed in `inherit_env`, and the
configuration cannot contain environment values.

One process is opened for each scenario and repeat. The runner writes one JSON object per line. The
process first receives `start` and must answer `ready`:

```json
{"protocol":1,"type":"start","session":{"suite":"support-smoke","target":"command-example","scenario_id":"refund-needs-order-number","repeat_index":1,"run_id":"...","comparison_id":null,"arm":null}}
{"protocol":1,"type":"ready"}
```

Each turn contains the complete visible history. Reply IDs must match:

```json
{"protocol":1,"type":"turn","id":1,"messages":[{"role":"user","content":"I need a refund."}]}
{"protocol":1,"type":"reply","id":1,"assistant_text":"What is your order number?","session_id":"optional","evidence":{"route":"refund-intake"}}
```

At the end, the runner sends `{"protocol":1,"type":"close"}` and closes stdin. Startup, each reply,
stdout line size, retained stderr, and shutdown are bounded. Timeout and cancellation terminate the
whole process group and reap the direct child. The adapter in
[`examples/jsonl_harness.py`](examples/jsonl_harness.py) is a small executable reference.

Only `assistant_text` becomes conversation text. Timing, the optional session ID, the sanitized
executable name, and JSON evidence are written to `evidence.jsonl`. They are excluded from the judge
request and Langfuse metadata. Failed commands add their bounded stderr to the local evidence file;
the traced exception reports only its byte count. Do not put secrets in `argv`; operating-system
process listings may show arguments even though this runner does not trace them.

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

The trace hierarchy is comparison, arm, scenario, and target turn. Allowlisted metadata identifies
the comparison, arm, target, harness kind, scenario, repeat, run, and turn. Judge score, judge pass,
arm gate, mean score, pass rate, and comparison deltas are attached as explicit scores. Pydantic AI
instrumentation also exports model prompts and responses when tracing is enabled; configure Langfuse
according to your data policy. GitHub Actions sets `LANGFUSE_RELEASE` to the evaluated commit. The
runner calls Langfuse shutdown in a `finally` block.

Checked-in YAML remains the source of truth. Langfuse dataset sync is deliberately not part of a CI
run, so a remote dataset edit cannot silently change a pull-request evaluation. Two hosted Langfuse
dataset experiment runs can be added later if the Langfuse comparison UI is required.

## State and automation

Conversation state is saved after every complete transition in an `InMemoryStateStore`. Redis is not
needed for one local process or GitHub Actions job and would not make external target effects
exactly-once. A target that changes an external database, queue, browser profile, or service remains
responsible for isolating or resetting that state between arms.

`make check` runs formatting, lint, type checking, and behavior tests. The normal
[`ci.yml`](.github/workflows/ci.yml) path needs no credentials. The manual
[`live-evals.yml`](.github/workflows/live-evals.yml) workflow runs both checked-in target configs with
Langfuse tracing and always uploads the complete comparison directory. Configure its `evals`
environment with `ZAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY`.

The model integration is pinned to Pydantic Evals and Pydantic AI 2.42. See the official
[Pydantic Evals overview](https://pydantic.dev/docs/ai/evals/evals/),
[LLMJudge guide](https://pydantic.dev/docs/ai/evals/evaluators/llm-judge/), and
[Langfuse SDK experiments](https://langfuse.com/docs/evaluation/experiments/experiments-via-sdk).
