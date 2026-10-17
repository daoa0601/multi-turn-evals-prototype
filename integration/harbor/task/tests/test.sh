#!/bin/sh
set -u

if [ -f /app/answer.txt ] && [ "$(cat /app/answer.txt)" = "scale-ready" ]; then
  printf '1\n' > /logs/verifier/reward.txt
  printf 'answer is correct\n'
else
  printf '0\n' > /logs/verifier/reward.txt
  printf 'answer is missing or incorrect\n'
fi
