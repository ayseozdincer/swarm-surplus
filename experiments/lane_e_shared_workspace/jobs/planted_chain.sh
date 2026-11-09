#!/bin/bash
# Lane E phase 2c, after generation: wait for the job to exit, stop vLLM to free the card, check the masks on the real model,
# replay every target's post-read solve calls with both references, decompose the scored positions, score.
cd /workspace/swarm
until grep -q '^exit ' /workspace/logs/planted.log; do sleep 30; done
pkill -f '[v]llm serve'; sleep 30
[ -d /usr/local/cuda-13.0/compat ] && export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export SWARM_MODEL=qwen3_14b SWARM_ARTIFACTS=/root/artifacts HF_HOME=/root/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/vllm-venv/bin/python
mkdir -p experiments/lane_e_shared_workspace/results data/replay_planted
$PY experiments/lane_e_shared_workspace/summarize_planted.py runs/planted --notes > experiments/lane_e_shared_workspace/results/summary_planted.txt
$PY scripts/render_parity.py runs/planted/seed*/leak > experiments/lane_e_shared_workspace/results/render_parity_planted.txt 2>&1
first=$(ls -d runs/planted/seed*/leak | head -1)
# a failed check is written down and read by hand; it does not halt a billing card (the tiny-model test covers the logic)
$PY tests/gpu_planted_check.py "$first" experiments/lane_e_shared_workspace/results/planted_check.json || echo "planted check did not pass, see experiments/lane_e_shared_workspace/results/planted_check.json"
$PY -u experiments/lane_e_shared_workspace/replay_planted.py runs/planted --out-root data/replay_planted || exit 1
$PY -u scripts/decompose.py data/replay_planted/planted/seed*/* --positions first_generated first_thought after_think mention_pre \
  planted_pre credential_pre tool_call_pre token_pre answer_pre || exit 1
$PY -u experiments/lane_e_shared_workspace/planted_scores.py runs/planted data/replay_planted/planted experiments/lane_e_shared_workspace/results/planted_scores.json || exit 1
