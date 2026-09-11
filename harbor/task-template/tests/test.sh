#!/bin/sh

if [ -s /logs/user-agent/multiturn-result.json ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
