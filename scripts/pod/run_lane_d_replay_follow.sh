#!/bin/bash
# Lane D replay on a second card while generation finishes elsewhere: runs/sweep2 is synced in from the generation pod
# (through the driver machine). Shared-prefix check on 14B, then each seed's core replay as soon as all its branches are
# written, the extra pass, decompose the scored positions, score. STOP_AT (unix time) stops starting new seeds.
cd /workspace/swarm
export SWARM_MODEL=qwen3_14b SWARM_ARTIFACTS=/root/artifacts HF_HOME=/root/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/venv/bin/python
[ -d /usr/local/cuda-13.0/compat ] && export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
mkdir -p results/lane_d data/replay_sweep2
until [ -f runs/sweep2/seed5001/leak/run.json ] || ls runs/sweep2/seed*/leak/run.json >/dev/null 2>&1; do sleep 30; done
first=$(ls -d runs/sweep2/seed*/leak | head -1)
blind=$($PY -c "import json; c=json.load(open('$first/run.json'))['run_config']; print([a for e, a in c['blind_plan'] if e == 2][0])")
$PY tests/gpu_prefix_check.py "$first" "$blind" 2 solve "" results/lane_d/prefix_check_14b.json
$PY -u scripts/pod/replay_sweep2.py runs/sweep2 --out-root data/replay_sweep2 --follow ${SEEDS:-20} ${STOP_AT:+--stop-at $STOP_AT} || exit 1
$PY -u scripts/decompose.py data/replay_sweep2/sweep2/seed*/* --positions prompt_last first_generated first_thought after_think \
  mention_pre planted_pre verdict_pre || exit 1
$PY -u scripts/sweep2_scores.py || exit 1
