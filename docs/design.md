# Design record

## Problem

The original Python target was one async text call. That was enough for Pydantic AI, but it could not
own a persistent CLI process or guarantee cleanup on timeout. The packaged CLI also knew only one
target configuration, so it could not produce an isolated two-arm comparison.

## Shape

Target YAML is a strict union selected by `kind`. The exhaustive factory returns a Pydantic AI,
command, or AgentENV target. All three expose the same lifecycle:

```text
Target.session(SessionContext) -> async TargetSession
TargetSession.reply(ConversationView) -> TargetReply
TargetSession.finish(SessionOutcome) -> TargetCompletion
```

The scenario runner creates the run ID and owns one target session around the complete adaptive
conversation. Pydantic AI has a zero-resource session. A command session owns one subprocess,
versioned JSONL, byte and time limits, stderr draining, process-group termination, and reaping.
AgentENV owns one remote sandbox per case. It sends full-history turns through bounded files, runs a
fixed verifier before teardown, and destroys the sandbox under bounded, cancellation-shielded
cleanup. Its provider TTL limits leaks after a hard host-process loss.

`TargetReply.assistant_text` becomes an `AssistantTurn`. Structured evidence becomes immutable
`TargetTurnEvidence` on the scenario result and is written separately. The judge gets an explicit
`Transcript` projection after the evaluator input is cleared, so it sees only public user and
assistant text. `TargetCompletion` can add environment verifier evidence, which stays outside the
judge input. A present environment failure makes the case fail even when the text judge passes.

The command wire protocol is version 2. After the last turn, the runner sends the terminal outcome
and requires a `finished` response before `close`. This makes verification and artifact references
available while the child still owns its resources. There is no protocol 1 fallback.

Repeats are expanded into `PlannedCase` values before Pydantic Evals runs. Each owns a typed
`CaseKey(scenario_id, repeat_index)`. Pydantic display names are derived from that key, while A/B
pairing uses the key itself. This keeps failed and repeated cases stable without parsing report-name
formatting.

The A/B coordinator accepts exactly one baseline and one candidate target. It validates and builds
both first, then runs them sequentially with fresh targets, actors, judges, stores, and reports. It
does not inspect the baseline gate before starting the candidate. Per-arm artifacts are written under
fixed `baseline/` and `candidate/` directories, followed by one paired comparison and root gate.

Harbor is an arm runner, not a target. It compiles one task per planned case, fixes Harbor attempts
at one, and submits baseline and candidate as separate sequential jobs. A custom Harbor user agent
uses ACP to run the same adaptive-user loop inside one trial. Harbor owns trial concurrency,
environment creation, verification, artifacts, and teardown. The outer process reads saved trial
results and normalized transcript artifacts, runs the text judge, and returns the same gate and
comparison shape as local execution. Direct AgentENV targets are rejected from this conceptual
nesting: when Harbor uses its E2B environment against AgentENV, Harbor is the sole sandbox owner.

The Harbor user agent is a small separate package with no evaluator dependencies. Harbor 0.22's
LiteLLM dependency requires OpenAI 2.x while Pydantic AI 2.42 requires OpenAI 3.x. Installing both in
one environment is unsatisfiable. The Harbor shim therefore sends bounded actor requests to a
one-shot worker in the evaluator environment. Conversation state stays in the Harbor trial; only
the private next-turn decision crosses that process boundary.

Langfuse sits behind a small tracing protocol. With tracing disabled, no Langfuse package or network
is needed. With tracing enabled, comparison, arm, and scenario roots are separate traces correlated
by comparison ID. Target turns and model calls stay beneath their scenario root. Each observation
carries only typed, allowlisted identity fields. Scenario, arm, and comparison scores are explicit.
Target configuration, argv, environment values, stderr, and command evidence are never trace
metadata. Harbor traces add only saved job and trial identities. Environment pass and reward are
explicit scores. Pydantic AI model content remains subject to its normal opt-in instrumentation.

## Synthesis decision

The AgentENV and Harbor extension also used two design passes. Native `kind: agentenv` won over a
JSONL child bridge because the evaluator remains the visible owner of the remote sandbox and can
destroy it when a turn fails. Harbor stayed above targets because one Harbor invocation represents a
whole trial rather than one assistant reply. The command protocol's terminal exchange and Harbor's
deterministic per-case tasks were retained from the other design.

The public CLI uses direct `--baseline` and `--candidate` paths instead of adding a third comparison
file. Two target files are already the reproducible configurations, and fixed option names encode the
requested cardinality.

## Tradeoffs

- One process per case isolates concurrent scenarios and arms, at the cost of process startup.
- Every command turn carries the complete visible history. Harness adapters can map that snapshot to
  their own session model without relying on hidden runner state.
- AgentENV uses one command invocation per turn rather than requiring persistent streamed stdin. The
  sandbox itself preserves files, services, and browser state across those invocations.
- Arms run sequentially. This avoids doubling the configured concurrency and reduces cross-arm rate
  limit interference.
- Harbor is a separate CLI path because it requires Python 3.12 and owns its own concurrency pool.
  The checked-in command pins Harbor in `uvx`; the local package remains usable on Python 3.11.
- The combined CI gate requires both arms to pass. Score deltas are descriptive and do not imply
  statistical significance.
- Local objects are isolated, but external databases, browser profiles, queues, and APIs are outside
  the runner's reset boundary.
- State stays process-local. Redis would not provide crash-safe replay or exactly-once external
  effects, so it is not included.

## Alternatives rejected

A new subprocess per turn lost because it cannot preserve a CLI browser or agent session. One
subprocess for the whole suite lost because concurrent cases would share mutable state. A generic
list of experiment arms lost because it adds baseline-selection and scheduling policy that a fixed
A/B comparison does not need. Passing Pydantic's `repeat` through and parsing report labels lost
because display names are not a domain identity. Sending full scenario results to the judge lost
because private actor state and command evidence could bias the verdict.

Running `harbor run` inside `TargetSession.reply()` lost because it would create one trial per chat
turn. Running an AgentENV target inside Harbor lost because two layers would both believe they own
the sandbox lifecycle. Adding Redis lost because neither a stored sandbox ID nor a saved transcript
can reconstruct remote process memory or make external side effects exactly once.

## Verification contract

Behavior tests run a real JSONL subprocess and cover two turns, malformed JSON, nonzero exit,
oversized output, timeout cancellation, and child reaping. Comparison tests prove both configurations
run, a failed baseline gate does not skip the candidate, every repeat pairs by `CaseKey`, and artifact
directories do not overwrite. Fake tracing tests check identity fields and scores without depending
on a Langfuse server. Fake AgentENV tests prove one sandbox per case, verify-before-destroy ordering,
bounded responses, cancellation cleanup, and separate judge evidence. Fake Harbor tests prove stable
task replacement, one job per arm, Harbor-owned concurrency, saved-result normalization, and
combined environment and judge gates. `scripts/compat_smoke.py` provides an offline compiler check
plus opt-in live AgentENV and full Harbor comparison modes.
