#!/bin/bash
# Lane D, after generation: wait for the sweep2 job to exit, stop vLLM to free the card, check the shared-prefix replay on
# 14B, replay seed by seed until STOP_AT (unix time; no new seed after it), decompose the scored positions, score.
cd /workspace/swarm
until grep -q '^exit ' /workspace/logs/sweep2.log; do sleep 30; done
pkill -f '[v]llm serve'; sleep 30
export SWARM_MODEL=qwen3_14b SWARM_ARTIFACTS=/root/artifacts HF_HOME=/root/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/vllm-venv/bin/python
mkdir -p results/lane_d data/replay_sweep2
blind=$($PY -c "import json; c=json.load(open('runs/sweep2/seed5001/leak/run.json'))['run_config']; print([a for e, a in c['blind_plan'] if e == 2][0])")
$PY tests/gpu_prefix_check.py runs/sweep2/seed5001/leak "$blind" 2 solve "" results/lane_d/prefix_check_14b.json
$PY -u scripts/pod/replay_sweep2.py runs/sweep2 --out-root data/replay_sweep2 ${STOP_AT:+--stop-at $STOP_AT} || exit 1
$PY -u scripts/decompose.py data/replay_sweep2/sweep2/seed*/* --positions prompt_last first_generated first_thought after_think \
  mention_pre planted_pre verdict_pre || exit 1
$PY -u scripts/sweep2_scores.py || exit 1
