# Broad GLM experiment

The checked-in broad experiment uses the same compiler and runner as smaller A/B tests. It expands
20 corpus cases into three arms: direct Pydantic AI, a local JSONL command harness, and the same
JSONL harness in a container. This yields 60 independently receipted multi-turn cases across web
chat, email, tickets, CLI work, and JSON APIs.

```console
scripts/build_glm_harness_image.sh
uv run multiturn-evals plan experiments/glm-wide.yaml --out outputs/glm-wide
uv run multiturn-evals experiment-run outputs/glm-wide
```

Resume a stopped coordinator without recompiling source YAML:

```console
uv run multiturn-evals resume outputs/glm-wide
```

`plan.json` is the saved contract. It embeds the corpus revision, resolved models and provider
routes, prompts, fixture, selected tasks, harness configuration, execution configuration, limits,
comparison pairs, and conservative request and output-token reservations. It stores credential
environment variable names, never their values.

Runtime preflight checks every arm before work begins. A missing provider key or executable fails the
whole preflight; it never removes an arm and then reports reduced coverage as success.

Each case writes `started.json` before target work. Completion writes the normal evaluation
artifacts, non-gating `trajectory.jsonl`, and `complete.json`. An exception writes `failure.json`.
Cancellation writes `interrupted.json`. Resume skips every terminal case. A stale started case is
marked interrupted rather than replayed because a transcript cannot prove whether an external side
effect already occurred.

The plan caps arms, cases, global concurrency, wall time, model requests, and reserved output tokens.
The 60-case GLM plan reserves 780 model requests and 486,000 possible output tokens, including up to
three post-task trajectory observations per case. These are admission bounds, not claims about actual
usage.

Container execution is a real execution choice: the runner starts the configured image and carries
the resolved target model, prompt, fixture, and case identity through protocol 2. The target model is
not fixed by the command adapter. AgentENV uses the same target-model and fixture fields in its
sandbox request. Harbor remains a whole-job execution type and is not represented as a turn-level
target.

A sandbox or container isolates only the files and processes it owns. It does not reset an external
database, queue, cache, account, or object store. Any experiment that needs those guarantees must put
the reset and verifier in the selected environment contract.
