# Design record

## Problem

The original Python target was one async text call. That was enough for Pydantic AI, but it could not
own a persistent CLI process or guarantee cleanup on timeout. The packaged CLI also knew only one
target configuration, so it could not produce an isolated two-arm comparison.

## Shape

Target YAML is a strict union selected by `kind`. The exhaustive factory returns either a Pydantic AI
target or a command target. Both expose the same lifecycle:

```text
Target.session(SessionContext) -> async TargetSession
TargetSession.reply(ConversationView) -> TargetReply
```

The scenario runner creates the run ID and owns one target session around the complete adaptive
conversation. Pydantic AI has a zero-resource session. A command session owns one subprocess,
versioned JSONL, byte and time limits, stderr draining, process-group termination, and reaping.

`TargetReply.assistant_text` becomes an `AssistantTurn`. Structured evidence becomes immutable
`TargetTurnEvidence` on the scenario result and is written separately. The judge gets an explicit
`Transcript` projection after the evaluator input is cleared, so it sees only public user and
assistant text.

Repeats are expanded into `PlannedCase` values before Pydantic Evals runs. Each owns a typed
`CaseKey(scenario_id, repeat_index)`. Pydantic display names are derived from that key, while A/B
pairing uses the key itself. This keeps failed and repeated cases stable without parsing report-name
formatting.

The A/B coordinator accepts exactly one baseline and one candidate target. It validates and builds
both first, then runs them sequentially with fresh targets, actors, judges, stores, and reports. It
does not inspect the baseline gate before starting the candidate. Per-arm artifacts are written under
fixed `baseline/` and `candidate/` directories, followed by one paired comparison and root gate.

Langfuse sits behind a small tracing protocol. With tracing disabled, no Langfuse package or network
is needed. With tracing enabled, comparison, arm, and scenario roots are separate traces correlated
by comparison ID. Target turns and model calls stay beneath their scenario root. Each observation
carries only typed, allowlisted identity fields. Scenario, arm, and comparison scores are explicit.
Target configuration, argv, environment values, stderr, and command evidence are never trace
metadata. Pydantic AI model content remains subject to its normal opt-in instrumentation.

## Synthesis decision

Two designs were compared. The session-and-case-plan design won because JSONL, cancellation cleanup,
and typed repeat identity are correctness boundaries rather than optional detail. The final shape
also keeps target-file source provenance, stores evidence on immutable scenario results, and judges a
visible transcript instead of assistant text alone.

The public CLI uses direct `--baseline` and `--candidate` paths instead of adding a third comparison
file. Two target files are already the reproducible configurations, and fixed option names encode the
requested cardinality.

## Tradeoffs

- One process per case isolates concurrent scenarios and arms, at the cost of process startup.
- Every command turn carries the complete visible history. Harness adapters can map that snapshot to
  their own session model without relying on hidden runner state.
- Arms run sequentially. This avoids doubling the configured concurrency and reduces cross-arm rate
  limit interference.
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

## Verification contract

Behavior tests run a real JSONL subprocess and cover two turns, malformed JSON, nonzero exit,
oversized output, timeout cancellation, and child reaping. Comparison tests prove both configurations
run, a failed baseline gate does not skip the candidate, every repeat pairs by `CaseKey`, and artifact
directories do not overwrite. Fake tracing tests check identity fields and scores without depending
on a Langfuse server.
