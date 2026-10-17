"""Isolated process entry point for the synchronous Langfuse SDK."""

from __future__ import annotations

import json
import sys

from langfuse import get_client

from pydantic_multiturn_evals.model import TraceEvent, TrialKey, TrialResult
from pydantic_multiturn_evals.tracing import export_with_langfuse_client


def main() -> int:
    payload = json.loads(sys.stdin.readline())
    export_with_langfuse_client(
        get_client(),
        TrialKey.model_validate(payload["key"]),
        tuple(TraceEvent.model_validate(event) for event in payload["events"]),
        TrialResult.model_validate(payload["result"]),
        payload["export_id"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
