"""Focused safety tests for the private ATIF trajectory importer.

These tests pin the semantics reviewed for Harbor continuation trajectories:

* steps flagged ``is_copied_context`` never become fresh message, tool, or
  model facts and never duplicate the public transcript, because the adapter
  imports every segment of a ``continued_trajectory_ref`` chain itself;
* imported model-call identities stay distinct across segments even when
  ``session_id`` and ``step_id`` repeat, with the receipt relative path as
  provenance;
* partial ATIF metrics never fabricate usage counts, and arbitrary
  ``metrics.extra`` integers never become usage counters;
* the content-part discriminator matches the pinned native Harbor 0.1.44
  parser, which rejects an image part that also carries ``text``.

The importer is exercised directly because its caller (the Harbor adapter) is
the only consumer and its file/chain handling lives there.
"""

import json

import pytest

from pydantic_multiturn_evals import (
    LogFact,
    ModelCallFact,
    ResourceFact,
)
from pydantic_multiturn_evals.adapters._harbor_trajectory import (
    AtifTrajectory,
    import_trajectory,
    parse_trajectory,
)


def _document(steps: list[dict], **overrides: object) -> dict:
    document: dict = {
        "schema_version": "ATIF-v1.6",
        "session_id": "session-1",
        "agent": {"name": "claude-code", "version": "2", "model_name": "agent-model"},
        "steps": steps,
    }
    document.update(overrides)
    return document


def _imported(document: dict, *, receipt: str = "agent/trajectory.json"):
    trajectory = parse_trajectory(json.dumps(document).encode("utf-8"))
    return import_trajectory(trajectory, receipt=receipt)


def _model_calls(receipt) -> list:
    return [fact.call for fact in receipt.facts if isinstance(fact, ModelCallFact)]


def _resources(receipt) -> list[ResourceFact]:
    return [fact for fact in receipt.facts if isinstance(fact, ResourceFact)]


def test_copied_context_steps_generate_no_facts_and_no_dialogue() -> None:
    document = _document(
        [
            {"step_id": 1, "source": "user", "message": "fresh question"},
            {
                "step_id": 2,
                "source": "agent",
                "message": "copied replay from an earlier segment",
                "is_copied_context": True,
                "tool_calls": [
                    {
                        "tool_call_id": "call-copied",
                        "function_name": "read_firewall",
                        "arguments": {"host": "edge"},
                    }
                ],
                "observation": {
                    "results": [{"source_call_id": "call-copied", "content": "copied observation"}]
                },
                "metrics": {"prompt_tokens": 50, "completion_tokens": 5},
            },
            {
                "step_id": 3,
                "source": "user",
                "message": "copied user turn",
                "is_copied_context": True,
            },
            {
                "step_id": 4,
                "source": "agent",
                "message": "fresh answer",
                "metrics": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        ]
    )

    imported = _imported(document)

    assert [message.content for message in imported.messages] == [
        "fresh question",
        "fresh answer",
    ]
    kinds = [fact.kind for fact in imported.facts]
    assert kinds.count("message") == 2
    assert kinds.count("tool") == 0
    assert kinds.count("model-call") == 1
    dumped = json.dumps([fact.model_dump(mode="json") for fact in imported.facts])
    assert "copied" not in dumped
    assert "read_firewall" not in dumped


def test_model_call_ids_stay_distinct_across_continuation_segments() -> None:
    steps = [
        {"step_id": 1, "source": "user", "message": "continue"},
        {
            "step_id": 2,
            "source": "agent",
            "message": "segment answer",
            "metrics": {"prompt_tokens": 12, "completion_tokens": 3},
        },
    ]
    paths = ["agent/trajectory.json", "agent/trajectory.cont-1.json", "agent_trajectory.json"]
    calls = [
        call for path in paths for call in _model_calls(_imported(_document(steps), receipt=path))
    ]

    assert len({call.operation for call in calls}) == len(paths)
    assert [call.requested.config["receipt"] for call in calls] == paths


def test_missing_token_counts_leave_usage_unknown_instead_of_zero() -> None:
    document = _document(
        [
            {
                "step_id": 1,
                "source": "agent",
                "message": "cost only",
                "metrics": {"cost_usd": 0.02},
            }
        ]
    )

    imported = _imported(document)

    calls = _model_calls(imported)
    assert len(calls) == 1
    assert calls[0].usage is None
    resources = _resources(imported)
    assert [(fact.name, fact.value, fact.unit) for fact in resources] == [
        ("harbor-trajectory-cost", 0.02, "usd")
    ]


def test_reported_zero_counts_are_preserved_as_usage() -> None:
    document = _document(
        [
            {
                "step_id": 1,
                "source": "agent",
                "message": "zero usage",
                "metrics": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                },
            }
        ]
    )

    imported = _imported(document)

    usage = _model_calls(imported)[0].usage
    assert usage is not None
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.cache_read_tokens == 0
    assert _resources(imported) == []


def test_negative_counts_drop_usage_but_keep_the_model_call_receipt() -> None:
    document = _document(
        [
            {
                "step_id": 1,
                "source": "agent",
                "message": "damaged metrics",
                "metrics": {
                    "prompt_tokens": 100,
                    "completion_tokens": -3,
                    "cached_tokens": 20,
                    "cost_usd": 0.02,
                },
            }
        ]
    )

    imported = _imported(document)

    calls = _model_calls(imported)
    assert len(calls) == 1
    assert calls[0].usage is None
    assert [(fact.name, fact.value) for fact in _resources(imported)] == [
        ("harbor-trajectory-prompt-tokens", 100.0),
        ("harbor-trajectory-cached-tokens", 20.0),
        ("harbor-trajectory-cost", 0.02),
    ]
    dumped = json.dumps([fact.model_dump(mode="json") for fact in imported.facts])
    assert "-3" not in dumped


def test_unrelated_extra_metrics_never_become_usage_counters() -> None:
    document = _document(
        [
            {
                "step_id": 1,
                "source": "agent",
                "message": "no tokens reported",
                "metrics": {
                    "cost_usd": 0.01,
                    "extra": {"latency_ms": 250, "requests": 9},
                },
            }
        ]
    )

    imported = _imported(document)

    calls = _model_calls(imported)
    assert len(calls) == 1
    assert calls[0].usage is None
    dumped = json.dumps([fact.model_dump(mode="json") for fact in imported.facts])
    assert "latency_ms" not in dumped
    assert "requests" not in dumped
    assert [(fact.name, fact.value) for fact in _resources(imported)] == [
        ("harbor-trajectory-cost", 0.01)
    ]

    complete = _document(
        [
            {
                "step_id": 1,
                "source": "agent",
                "message": "full tokens reported",
                "metrics": {
                    "prompt_tokens": 5,
                    "completion_tokens": 6,
                    "extra": {"latency_ms": 250},
                },
            }
        ]
    )
    usage = _model_calls(_imported(complete))[0].usage
    assert usage is not None
    assert usage.details == {}


def test_aggregate_final_metrics_never_create_duplicate_model_calls() -> None:
    document = _document(
        [
            {
                "step_id": 1,
                "source": "agent",
                "message": "partial step",
                "metrics": {"cost_usd": 0.02},
            }
        ],
        final_metrics={
            "total_prompt_tokens": 9000,
            "total_completion_tokens": 800,
            "total_cost_usd": 1.5,
            "total_steps": 4,
        },
    )

    imported = _imported(document)

    calls = _model_calls(imported)
    assert len(calls) == 1
    dumped = json.dumps([fact.model_dump(mode="json") for fact in imported.facts])
    assert "total_prompt_tokens" not in dumped
    assert "9000" not in dumped
    assert "1.5" not in dumped


def test_private_and_system_content_stays_out_of_the_import() -> None:
    document = _document(
        [
            {
                "step_id": 1,
                "source": "system",
                "message": "SYSTEM SECRET orchestration note",
            },
            {
                "step_id": 2,
                "source": "agent",
                "message": "public answer",
                "reasoning_content": "internal reasoning must stay hidden",
                "metrics": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "prompt_token_ids": [1, 2, 3],
                    "completion_token_ids": [4],
                    "logprobs": [-0.25],
                },
                "extra": {"hidden": "step-extra-secret"},
            },
        ],
        notes="trajectory notes secret",
        extra={"root": "root-extra-secret"},
    )
    document["agent"]["extra"] = {"agent": "agent-extra-secret"}

    imported = _imported(document)

    dumped = json.dumps(
        {
            "messages": [message.model_dump(mode="json") for message in imported.messages],
            "facts": [fact.model_dump(mode="json") for fact in imported.facts],
        }
    )
    for secret in (
        "SYSTEM SECRET",
        "internal reasoning",
        "token_ids",
        "logprobs",
        "step-extra-secret",
        "agent-extra-secret",
        "trajectory notes secret",
        "root-extra-secret",
    ):
        assert secret not in dumped
    usage = _model_calls(imported)[0].usage
    assert usage is not None
    assert usage.input_tokens == 10
    assert usage.output_tokens == 2


def test_image_content_part_with_text_is_rejected_like_native_harbor() -> None:
    document = _document(
        [
            {
                "step_id": 1,
                "source": "user",
                "message": [
                    {
                        "type": "image",
                        "text": "caption must be rejected",
                        "source": {"media_type": "image/png", "path": "shot.png"},
                    }
                ],
            }
        ]
    )

    with pytest.raises(ValueError, match="'text' field is not allowed when type='image'"):
        _imported(document)


def test_text_content_part_with_source_is_rejected_like_native_harbor() -> None:
    document = _document(
        [
            {
                "step_id": 1,
                "source": "user",
                "message": [
                    {
                        "type": "text",
                        "text": "caption",
                        "source": {"media_type": "image/png", "path": "shot.png"},
                    }
                ],
            }
        ]
    )

    with pytest.raises(ValueError, match="'source' field is not allowed when type='text'"):
        _imported(document)


def test_image_without_source_is_rejected_like_native_harbor() -> None:
    document = _document([{"step_id": 1, "source": "user", "message": [{"type": "image"}]}])

    with pytest.raises(ValueError, match="'source' field is required when type='image'"):
        _imported(document)


def test_mixed_text_and_image_content_keeps_text_and_warns() -> None:
    document = _document(
        [
            {
                "step_id": 1,
                "source": "user",
                "message": [
                    {"type": "text", "text": "look at this"},
                    {
                        "type": "image",
                        "source": {"media_type": "image/png", "path": "shot.png"},
                    },
                ],
            }
        ]
    )

    imported = _imported(document)

    assert [message.content for message in imported.messages] == ["look at this"]
    warnings = [fact for fact in imported.facts if isinstance(fact, LogFact)]
    assert len(warnings) == 1
    assert warnings[0].level == "warning"
    assert "1 image content parts" in warnings[0].message


def test_parse_returns_the_typed_trajectory_model() -> None:
    trajectory = parse_trajectory(
        json.dumps(_document([{"step_id": 1, "source": "user", "message": "hi"}])).encode()
    )
    assert isinstance(trajectory, AtifTrajectory)
