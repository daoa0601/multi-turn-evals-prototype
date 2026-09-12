from __future__ import annotations

import asyncio

from pydantic_multiturn_evals.models import AssistantTurn, Exchange, Transcript, UserTurn
from pydantic_multiturn_evals.trajectory import (
    TrajectoryAssessment,
    assess_trajectory,
    transcript_prefixes,
)


def transcript() -> Transcript:
    return Transcript(
        exchanges=tuple(
            Exchange(
                user=UserTurn(content=f"user-{index}"),
                assistant=AssistantTurn(content=f"assistant-{index}"),
            )
            for index in range(1, 4)
        )
    )


def test_transcript_prefixes_never_include_a_later_exchange() -> None:
    prefixes = transcript_prefixes(transcript(), max_prefixes=3)

    assert [len(prefix.exchanges) for prefix in prefixes] == [1, 2, 3]
    assert "assistant-2" not in prefixes[0].model_dump_json()
    assert "assistant-3" not in prefixes[1].model_dump_json()


def test_trajectory_failures_are_recorded_without_aborting_other_points() -> None:
    seen: list[int] = []

    class Assessor:
        async def assess(
            self,
            *,
            rubric: str,
            prefix: Transcript,
            turn_index: int,
        ) -> TrajectoryAssessment:
            seen.append(turn_index)
            if turn_index == 2:
                raise RuntimeError("observer unavailable")
            return TrajectoryAssessment(
                turn_index=turn_index,
                score=turn_index / 3,
                passed=True,
                reason=f"Assessed {len(prefix.exchanges)} exchange(s) against {rubric}.",
            )

    result = asyncio.run(
        assess_trajectory(
            transcript(),
            rubric="Judge progress and safety at this point.",
            assessor=Assessor(),
            max_prefixes=3,
        )
    )

    assert seen == [1, 2, 3]
    assert result.status == "partial"
    assert [point.status for point in result.points] == ["complete", "failed", "complete"]
    assert result.points[1].error == "RuntimeError: observer unavailable"


def test_prefix_bound_is_deterministic_and_reports_truncation() -> None:
    prefixes = transcript_prefixes(transcript(), max_prefixes=2)

    assert [len(prefix.exchanges) for prefix in prefixes] == [1, 2]
