"""Post-task transcript-prefix assessment that cannot affect conversation control."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import Field
from pydantic_ai import Agent, format_as_xml

from pydantic_multiturn_evals.model_bindings import BoundModel
from pydantic_multiturn_evals.models import Identifier, StrictModel, Text, Transcript


class TrajectoryAssessment(StrictModel):
    turn_index: int = Field(ge=1)
    score: float = Field(ge=0, le=1)
    passed: bool
    reason: Text


class TrajectoryPoint(StrictModel):
    turn_index: int = Field(ge=1)
    status: Literal["complete", "failed"]
    score: float | None = Field(default=None, ge=0, le=1)
    passed: bool | None = None
    reason: str | None = None
    error: str | None = None


class TrajectoryResult(StrictModel):
    status: Literal["complete", "partial", "skipped"]
    total_turns: int = Field(ge=0)
    assessed_turns: int = Field(ge=0)
    truncated: bool = False
    points: tuple[TrajectoryPoint, ...] = ()


class CaseTrajectory(StrictModel):
    case_name: Text
    scenario_id: Identifier
    repeat_index: int = Field(ge=1)
    assessment: TrajectoryResult


class TrajectoryAssessor(Protocol):
    async def assess(
        self,
        *,
        rubric: str,
        prefix: Transcript,
        turn_index: int,
    ) -> TrajectoryAssessment: ...


class _TrajectoryPayload(StrictModel):
    score: float = Field(ge=0, le=1)
    passed: bool
    reason: Text


class PydanticTrajectoryAssessor:
    def __init__(self, model_binding: BoundModel, *, instructions: str) -> None:
        self._settings = model_binding.settings
        self._agent = Agent(
            model_binding.model,
            output_type=_TrajectoryPayload,
            instructions=instructions,
        )

    async def assess(
        self,
        *,
        rubric: str,
        prefix: Transcript,
        turn_index: int,
    ) -> TrajectoryAssessment:
        result = await self._agent.run(
            format_as_xml(
                {
                    "rubric": rubric,
                    "visible_conversation_through_turn": prefix,
                }
            ),
            model_settings=self._settings,
            metadata={"eval_role": "trajectory", "turn_index": turn_index},
        )
        return TrajectoryAssessment(turn_index=turn_index, **result.output.model_dump())


def transcript_prefixes(transcript: Transcript, *, max_prefixes: int) -> tuple[Transcript, ...]:
    if max_prefixes < 1:
        raise ValueError("max_prefixes must be positive")
    count = min(len(transcript.exchanges), max_prefixes)
    return tuple(
        Transcript(exchanges=transcript.exchanges[:turn_index])
        for turn_index in range(1, count + 1)
    )


async def assess_trajectory(
    transcript: Transcript,
    *,
    rubric: str,
    assessor: TrajectoryAssessor,
    max_prefixes: int,
) -> TrajectoryResult:
    prefixes = transcript_prefixes(transcript, max_prefixes=max_prefixes)
    if not prefixes:
        return TrajectoryResult(
            status="skipped",
            total_turns=0,
            assessed_turns=0,
        )

    points: list[TrajectoryPoint] = []
    for turn_index, prefix in enumerate(prefixes, start=1):
        try:
            assessment = await assessor.assess(
                rubric=rubric,
                prefix=prefix,
                turn_index=turn_index,
            )
        except Exception as error:
            points.append(
                TrajectoryPoint(
                    turn_index=turn_index,
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                )
            )
            continue
        points.append(
            TrajectoryPoint(
                turn_index=turn_index,
                status="complete",
                score=assessment.score,
                passed=assessment.passed,
                reason=assessment.reason,
            )
        )

    truncated = len(prefixes) < len(transcript.exchanges)
    complete = not truncated and all(point.status == "complete" for point in points)
    return TrajectoryResult(
        status="complete" if complete else "partial",
        total_turns=len(transcript.exchanges),
        assessed_turns=len(prefixes),
        truncated=truncated,
        points=tuple(points),
    )
