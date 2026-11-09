#!/bin/bash
# lane D final pass on the replay card: wait for the running decompose, decompose what changed since (stale or new), score
cd /workspace/swarm
export SWARM_MODEL=qwen3_14b SWARM_ARTIFACTS=/root/artifacts HF_HOME=/root/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/venv/bin/python
while pgrep -f "[d]ecompose.py" >/dev/null; do sleep 15; done
$PY -u scripts/decompose.py data/replay_sweep2/sweep2/seed*/* data/replay_sweep2/sweep2_norule/seed*/* \
  --positions prompt_last first_generated first_thought after_think mention_pre planted_pre verdict_pre || exit 1
$PY -u experiments/lane_d_sweep2/sweep2_scores.py || exit 1
$PY -u experiments/lane_d_sweep2/arms_scores.py > /workspace/logs/arms_scores.log 2>&1
echo final done
