#!/bin/sh
set -eu

docker build \
  --file environments/glm-harness/Dockerfile \
  --tag pydantic-multiturn-evals-glm:local \
  .
