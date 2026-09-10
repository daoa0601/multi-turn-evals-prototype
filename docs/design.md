# Design record

## Problem

Pydantic Evals runs one task callable per case, but it does not supply an adaptive conversational
user. This package must turn one case into a bounded conversation, keep the simulated user separate
from the final judge, and remain runnable in CI without Redis, Langfuse, or AgentTrace.

## Usage

Library callers provide one async chat target and call `evaluate_suite()`. The CLI builds that target
from a separate Pydantic AI target file. In both paths, suite YAML supplies scenarios, actor and judge
models, hard limits, and gate policy.

## Shape

The Pydantic `Dataset` is the outer runner. Each `Case` input is a `Scenario`; its task executes the
whole conversation and returns a `ScenarioResult`. Complete `Exchange` values prevent the actor and
judge from seeing a half-written turn. Tagged actor decisions and terminal states keep transition
logic explicit. The runner, not the actor prompt, enforces the target-turn and wall-clock limits.

Each case gets a `TranscriptJudge`, a small adapter around Pydantic's `LLMJudge`. The adapter replaces
the evaluator output with the public transcript before judging, so actor decisions cannot bias the
verdict. The gate reads the resulting report dataclass and fails closed on task errors, evaluator
errors, missing results, or non-finite scores.

Provider construction has one fixed URL per Z.AI endpoint plan. The same OpenAI-compatible Pydantic
model adapter covers both official chat-completion endpoints. API keys come only from the named
environment variable.

## Synthesis decision

Candidate 2 was the base because its complete-exchange representation and fail-closed gate made the
legal states clearer. Candidate 1 contributed its explicit per-case gate records, atomic artifact
writes, a dedicated YAML module, and the decision to treat Redis as short-lived observation rather
than reliable resume.

Three ideas were rejected. The actor does not import Agent Blocks because that code is an unpublished
TypeScript coding orchestrator, not a Python simulated-user runtime. The judge does not receive the
actor's termination reason. Redis, AgentTrace, and Langfuse do not own execution or scenario data.

## Tradeoffs

- The target receives its full visible history on each call. This keeps the target boundary stateless
  and provider-neutral at the cost of resending messages.
- State remains process-local. This avoids false exactly-once claims, but a killed job reruns a case
  from the beginning.
- The first transcript contains text exchanges only. Tool and environment evidence should become
  explicit domain fields when a concrete target needs them.
- A shared model may play target, actor, and judge in the example. Their prompts and calls remain
  separate so another model can replace any role during calibration.

## Alternatives

A public start/advance/judge state machine lost because callers would have to coordinate legal
ordering. A bespoke batch runner followed by Pydantic reporting lost because it would create two
schedulers and hide task failures from Pydantic. Redis-first workers lost because queue leases and
idempotency would become required even though CI cases fit in one process.

## Verification

`make check` passes 14 behavior tests with 87 percent branch-aware coverage, Ruff, and BasedPyright.
The checked-in YAML validates. A live `glm-5.3-flash` run through the Coding Plan endpoint completed
both scenarios. The refund scenario used two target turns and supplied its order number only after the
target asked for it. Both independent Pydantic judges returned a passing assertion and score 1.0.
