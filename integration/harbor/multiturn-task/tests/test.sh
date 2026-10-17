#!/bin/sh
set -u

seed="$(cat /app/state/seed.txt 2>/dev/null)"
host="$(uname -n)"
expected="${seed}-${host}"
actual="$(cat /app/final/answer.txt 2>/dev/null)"

if [ -n "$seed" ] && [ "$actual" = "$expected" ]; then
  printf '1\n' > /logs/verifier/reward.txt
  printf 'answer matches the seed-hostname state\n'
else
  printf '0\n' > /logs/verifier/reward.txt
  printf 'answer is missing or does not match %s\n' "$expected"
fi
