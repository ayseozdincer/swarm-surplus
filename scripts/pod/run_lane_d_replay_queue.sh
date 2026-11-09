#!/bin/bash
# Lane D replay queue on the replay card: after the sweep2 follow job exits, the rule-off arm the same way.
cd /workspace/swarm
until grep -q '^exit ' /workspace/logs/replay.log; do sleep 30; done
EXP=sweep2_norule PLAN=norule SEEDS=${SEEDS:-20} bash scripts/pod/run_lane_d_replay_follow.sh
