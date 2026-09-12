FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY examples/glm_jsonl_harness.py ./examples/glm_jsonl_harness.py
COPY targets/glm-wide.yaml ./targets/glm-wide.yaml

RUN python -m pip install --no-cache-dir .

ENTRYPOINT ["python", "examples/glm_jsonl_harness.py"]
