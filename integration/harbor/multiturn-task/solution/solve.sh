#!/bin/sh
set -eu

mkdir -p /app/final
printf '%s-%s\n' "$(cat /app/state/seed.txt)" "$(uname -n)" > /app/final/answer.txt
