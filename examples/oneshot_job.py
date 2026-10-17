"""One-shot job used by the offline end-to-end campaign."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    request = json.loads(sys.stdin.readline())
    workspace = Path(os.environ["MTE_TRIAL_DIRECTORY"])
    artifact = workspace / "job-output.txt"
    artifact.write_text(request["case"]["task"] + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "protocol": 1,
                "evidence": {
                    "public": {
                        "messages": [
                            {
                                "role": "assistant",
                                "content": "resolved: one-shot job",
                                "actor": "offline-job",
                            }
                        ],
                        "output": {"seed": request["trial"]["seed"]},
                    },
                    "checks": [
                        {
                            "name": "completed",
                            "passed": True,
                            "reason": "the one-shot job returned its evidence",
                        }
                    ],
                    "artifacts": [{"name": "job-output", "path": artifact.name}],
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
