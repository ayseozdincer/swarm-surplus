#!/bin/bash
# Lane D, method_tip replay and found-location scoring on one card, after replay_setup.sh; sweep2_solvable's turns are already on the
# volume, so only the arm's own twenty seeds are replayed.
cd /workspace/swarm
until grep -q "^exit " /workspace/logs/tipsetup.log; do sleep 5; done
grep -q "^exit 0" /workspace/logs/tipsetup.log || exit 1
ORDER="sweep2_methodtip:1" EXPECT="sweep2_methodtip=20" bash experiments/lane_d_sweep2/jobs/replay_follow.sh
export SWARM_MODEL=qwen3_14b SWARM_ARTIFACTS=/root/artifacts HF_HOME=/root/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -d /usr/local/cuda-13.0/compat ] && export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
/root/venv/bin/python -u experiments/lane_d_sweep2/methodtip_found.py
echo "tip replay exit $(date)"
