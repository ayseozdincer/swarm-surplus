#!/bin/bash
# resume the paused forbidden_leak run once both solvable processes have exited
while pgrep -f "[r]uns/sweep2_solvable" >/dev/null; do sleep 20; done
echo "solvable done $(date)"; kill -CONT $(pgrep -f "[r]uns/sweep2_forbidden2")
