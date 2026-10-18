#!/bin/sh
set -u

if [ -f /app/answer.txt ]; then
  printf '1\n' > /logs/verifier/reward.txt
else
  printf '0\n' > /logs/verifier/reward.txt
fi
