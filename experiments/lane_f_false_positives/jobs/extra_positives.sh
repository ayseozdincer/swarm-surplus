#!/bin/bash
# Lane F, second request, on a fresh pod: lane D's replay setup, then the blind agent's solve turns in sweep2 forbidden_leak,
# sweep2_forbidden2 (forbidden_leak, none) and sweep2_norule (leak, none), traced. Run from /workspace/swarm after the code is
# synced; it waits for /workspace/runs_synced, which the driver machine touches when the run directories are up.
#   STOP_AT  unix time after which no new turn starts
cd /workspace/swarm
mkdir -p /workspace/logs
bash scripts/pod/replay_setup.sh > /workspace/logs/setup.log 2>&1 || { echo "setup failed $(date -u)"; exit 1; }
echo "setup done $(date -u)"
until [ -f /workspace/runs_synced ]; do sleep 5; done
export SWARM_MODEL=qwen3_14b SWARM_ARTIFACTS=/root/artifacts HF_HOME=/root/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
touch /workspace/busy
/root/venv/bin/python -u experiments/lane_f_false_positives/replay_traces.py runs --out-root data/replay_lane_f --units forbidden forbidden2 norule \
  ${STOP_AT:+--stop-at $STOP_AT} > /workspace/logs/lane_f_extra.log 2>&1
echo "extra replay exit $? $(date -u)"
