#!/bin/bash
# Lane D, the method_tip length control and lane C's placebo for the two trace seeds, after replay_setup.sh, then the two CSVs again.
cd /workspace/swarm
until grep -q "^exit " /workspace/logs/tipsetup2.log; do sleep 5; done
grep -q "^exit 0" /workspace/logs/tipsetup2.log || exit 1
export SWARM_MODEL=qwen3_14b SWARM_ARTIFACTS=/root/artifacts HF_HOME=/root/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -d /usr/local/cuda-13.0/compat ] && export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
/root/venv/bin/python -u experiments/lane_d_sweep2/methodtip_length.py --csv-seeds seed5034 seed5046
/root/venv/bin/python experiments/lane_d_sweep2/trace_csv.py seed5034
/root/venv/bin/python experiments/lane_d_sweep2/trace_csv.py seed5046
echo "tip length exit $(date)"
