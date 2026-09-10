# Pydantic multi-turn evals

This package runs adaptive conversation scenarios through Pydantic Evals. Each scenario starts with
an authored user prompt. A simulated user reads every target response and chooses the next message,
accepts the interaction, or stops it. A separate LLM judge grades the frozen transcript.

The target, simulated user, and judge default to `glm-5.3-flash`. The checked-in examples use the
Z.AI Coding Plan endpoint and read `ZAI_API_KEY` from the process environment. The program does not
read `.zshrc`; a local shell may export the variable from there, while GitHub Actions must provide it
as a repository or environment secret.

## Run the example

```console
uv sync
uv run multiturn-evals validate scenarios/support.yaml --target targets/support.yaml
uv run multiturn-evals run scenarios/support.yaml \
  --target targets/support.yaml \
  --out outputs/support
```

The run writes four reviewable artifacts:

- `report.txt` is the human-readable Pydantic Evals report.
- `report.json` contains case inputs, full results, judge values, failures, and trace IDs.
- `transcripts.jsonl` contains one complete scenario result per successful case.
- `gate.json` is the small CI contract. The command exits with status 1 when its gate fails.

The example suite allows at most three target replies per scenario, runs cases serially, and disables
task retries. Those limits matter because replaying a state-changing conversation is unsafe unless
the external system resets or accepts an idempotency key.

## Use a Python target

The library target is an async callable. It receives complete user/assistant exchanges plus the one
pending user turn. It does not need to use Pydantic AI.

```python
from pydantic_multiturn_evals import ConversationView, evaluate_suite


async def target(view: ConversationView) -> str:
    messages = [{"role": turn.role, "content": turn.content} for turn in view.messages]
    return await existing_chat_application.reply(messages)


result = await evaluate_suite("scenarios/support.yaml", target=target)
result.report.print(include_reasons=True)
result.write_artifacts("outputs/support")
raise SystemExit(0 if result.gate.passed else 1)
```

`evaluate_suite()` compiles every scenario into a Pydantic `Case`. The task for that case is the
entire adaptive conversation, not one model turn. Each case has its own rubric and its own Pydantic
`LLMJudge`, configured to return both `judge_score` and `judge_pass`.

The judge receives only the visible exchanges. It does not receive the simulated user's private
brief, its accept/stop decision, or the reason for that decision.

## Scenario format

[`scenarios/support.yaml`](scenarios/support.yaml) is the working example. A scenario owns:

- `first_prompt`, the first user message;
- an actor persona, goal, and private rules used to choose later user turns;
- a judge rubric used only after the conversation ends;
- optional turn and timeout limits.

Suite-level model and limit settings provide defaults. Unknown YAML fields fail validation, scenario
IDs must be unique, and endpoint plans select one fixed URL:

- `general` uses `https://api.z.ai/api/paas/v4`.
- `coding` uses `https://api.z.ai/api/coding/paas/v4`.

The configuration can name an environment variable but cannot contain a base URL or API key value.

## State and tracing

Conversation state lives in an `InMemoryStateStore` and is saved after every complete transition.
That is enough for local runs and GitHub Actions because a scenario never moves between processes.
Redis would not make external target effects exactly-once, so this first version does not pretend to
offer crash resume. The `StateStore` protocol is available when a real cross-process use case exists.

Every successful result contains its full typed transcript and controller decisions. Pydantic Evals
adds report and case trace IDs. Langfuse tracing is opt-in:

```console
uv sync --extra tracing
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...
export LANGFUSE_BASE_URL=https://cloud.langfuse.com
uv run multiturn-evals run scenarios/support.yaml \
  --target targets/support.yaml \
  --out outputs/support \
  --langfuse
```

Local YAML remains the scenario source of truth. Langfuse dataset sync is intentionally absent from
the first version, so a remote dataset edit cannot silently change a pull-request evaluation.

## Automation

`make check` runs formatting, lint, type checking, and tests with fake model behavior. The normal
[`ci.yml`](.github/workflows/ci.yml) workflow runs that path without credentials.

[`live-evals.yml`](.github/workflows/live-evals.yml) is manual and runs the entire checked-in support
suite with one case at a time. Configure an `evals` GitHub environment with `ZAI_API_KEY`. The job
always uploads artifacts before it reports a failed gate.

The model adapter follows the current Pydantic Evals and Pydantic AI 2.42 APIs. See the official
[Pydantic Evals overview](https://pydantic.dev/docs/ai/evals/evals/),
[LLMJudge guide](https://pydantic.dev/docs/ai/evals/evaluators/llm-judge/), and
[GLM-5.3-Flash model page](https://docs.z.ai/guides/vlm/glm-5.3-flash).
