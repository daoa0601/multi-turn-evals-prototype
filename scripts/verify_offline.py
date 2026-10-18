"""Run the public offline campaign twice and inspect its real artifacts."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from examples.offline_campaign import campaign
from pydantic_multiturn_evals import read_report, run


async def verify() -> None:
    with tempfile.TemporaryDirectory(prefix="multiturn-evals-") as temporary:
        directory = Path(temporary)
        first = await run(campaign, directory=directory, workers=3)
        if first.exit_code != 1 or first.executed_trials != 8:
            raise RuntimeError("the first run did not expose the intended quality failure")
        comparison = first.compare("baseline", "candidate")
        if comparison.only_right_passed != 2:
            raise RuntimeError("the candidate did not win both exact sample pairs")
        starts = tuple(directory.glob("trials/**/target-started.json"))
        if len(starts) != 4:
            raise RuntimeError("the persistent command target did not start once per trial")
        if len(tuple(directory.glob("trials/**/job-output.txt"))) != 2:
            raise RuntimeError("the one-shot command harness did not preserve both artifacts")
        for event_path in directory.glob("trials/**/events.jsonl"):
            kinds = {
                json.loads(line)["fact"]["kind"]
                for line in event_path.read_text(encoding="utf-8").splitlines()
            }
            if "check" not in kinds or "lifecycle" not in kinds:
                raise RuntimeError(f"the common trace is incomplete: {event_path}")

        second = await run(campaign, directory=directory, workers=3)
        if second.executed_trials != 0 or second.exit_code != 1:
            raise RuntimeError(
                "the second run replayed terminal work or changed the quality result"
            )
        if len(tuple(directory.glob("trials/**/target-started.json"))) != 4:
            raise RuntimeError("the second run started another command target")
        if read_report(directory).results != second.results:
            raise RuntimeError("the read-only report disagrees with the saved receipts")
        print("offline run verified: 3 harness shapes, 8 trials, 2 exact wins, 0 replays")


if __name__ == "__main__":
    asyncio.run(verify())
