# Broad GLM campaign design

## Problem

The ordinary runner keeps a complete arm in memory and writes artifacts after the arm finishes. A
long run can therefore repeat completed model calls after a host failure. The broad campaign also
needs stable channel, task, scenario-environment, and execution-environment identities. Display
names and free-form tags are not sufficient for recovery or coverage reporting.

## Usage

Create an inspectable plan without calling a model:

```console
uv run python scripts/run_scale_campaign.py plan campaigns/glm-wide.yaml \
  --out outputs/glm-wide-plan
```

Run the saved units:

```console
uv run python scripts/run_scale_campaign.py run campaigns/glm-wide.yaml \
  --out outputs/glm-wide
```

Resume a stopped coordinator:

```console
uv run python scripts/run_scale_campaign.py resume outputs/glm-wide
```

## Shape

`CampaignPlan` is the saved contract. It contains the resolved suite, target configurations,
admission bounds, explicit exclusions, and ordered `CampaignUnitPlan` values. Each unit identifies
one scenario in one execution lane. Scenario tags provide exactly one `channel.*`, `task.*`,
`environment.*`, and `revision.*` value. The compiler parses those values into `CampaignUnitKey`
before execution.

The unit key stores `target_kind` separately from `execution_environment`. A Docker-backed JSONL
target still has target kind `command`, but its execution environment is `docker-container` rather
than `local-process`.

The coordinator runs lanes sequentially and applies one concurrency limit within a lane. A unit
writes `started.json` before calling the target. It writes the normal Pydantic evaluation artifacts
and `complete.json` after grading. An exception writes `failure.json`. Cancellation writes
`interrupted.json`.

Resume skips every terminal unit. A prior `started.json` without a terminal receipt becomes
interrupted and does not run again. Stored conversation text cannot restore a command process,
sandbox, database, or external delivery, so the coordinator does not claim mid-case recovery.

The campaign manifest caps planned cases, concurrency, wall time, reserved model requests, and
reserved output tokens. These are admission limits. They are conservative estimates because an
external command can hide provider retries. The current campaign keeps automatic case and Harbor
retries disabled.

## Synthesis decision

Two designs were compared. The selected design uses immutable files and a campaign-specific runner.
The rejected design added a general matrix language, SQLite claims, and singleton Harbor jobs. The
larger design had stronger multi-coordinator semantics, but it delayed the first real GLM run.

The selected design adopted two ideas from the larger design. Stable revisions and explicit
exclusions prevent inflated coverage claims. Request and output-token reservations stop an oversized
plan before paid work starts.

## Tradeoffs accepted

- The runner supports one coordinator per output directory in exchange for a small, inspectable
  implementation.
- Completed units are durable, but active units do not resume automatically.
- Filesystem lookup is adequate for tens or hundreds of units. A database becomes useful only when
  many coordinators need shared claims.
- Simulated email and ticket prompts measure response behavior. They do not prove delivery through a
  real mail or ticket system.

## Environment boundary

The checked-in campaign runs the direct Pydantic lane, a local JSONL process lane, and the same JSONL
contract inside a Docker container. AgentENV and Harbor remain explicit exclusions until their
credentials, templates, target agents, and external-state isolation are configured. A sandbox does
not reset an external database, queue, cache, account, or object store.
