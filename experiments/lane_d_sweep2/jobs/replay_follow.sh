#!/bin/bash
# Lane D replay on its own card while generation runs on the server's card; runs/<exp> is synced in from the generation
# pod through the driver machine. Shared-prefix check on 14B, then the follow-mode replay of EXP with PLAN until SEEDS
# seeds have every priority done. Decompose and score run separately (they can share this card with the replay).
cd /workspace/swarm
export SWARM_MODEL=qwen3_14b SWARM_ARTIFACTS=/root/artifacts HF_HOME=/root/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/venv/bin/python
[ -d /usr/local/cuda-13.0/compat ] && export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
EXP=${EXP:-sweep2}; PLAN=${PLAN:-sweep2}
mkdir -p experiments/lane_d_sweep2/results data/replay_sweep2
if [ ! -f experiments/lane_d_sweep2/results/prefix_check_14b.json ]; then
  first=$(ls -d runs/$EXP/seed*/leak | head -1)
  blind=$($PY -c "import json; c=json.load(open('$first/run.json'))['run_config']; print([a for e, a in c['blind_plan'] if e == 2][0])")
  $PY tests/gpu_prefix_check.py "$first" "$blind" 2 solve "" experiments/lane_d_sweep2/results/prefix_check_14b.json
fi
$PY -u experiments/lane_d_sweep2/replay_sweep2.py runs --out-root data/replay_sweep2 ${ORDER:+--order "$ORDER"} ${EXPECT:+--expect $EXPECT}
