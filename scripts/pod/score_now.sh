#!/bin/bash
# decompose what the replay has written so far and score it (shares the card with the replay)
cd /workspace/swarm
export SWARM_MODEL=qwen3_14b SWARM_ARTIFACTS=/root/artifacts HF_HOME=/root/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/venv/bin/python
$PY -u scripts/decompose.py data/replay_sweep2/sweep*/seed*/* --positions prompt_last first_generated first_thought after_think \
  mention_pre planted_pre verdict_pre || exit 1
$PY -u scripts/sweep2_scores.py
