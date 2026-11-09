# Lane C stage 2 on a subset of seeds, so two pods can split it: the free agents' verdict turns of every variant, actual and
# kept reference, then their decomposition.   run_lane_c_verdict.sh seed3015 seed3018 ...
. /workspace/venv/bin/activate
cd /workspace/swarm
export HF_HOME=/workspace/hf SWARM_MODEL=qwen3_14b PYTHONPATH=/workspace/swarm/third_party/maj-lens:/workspace/swarm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
POSITIONS="prompt_last first_generated first_thought after_think answer_pre verdict_pre end_of_response mention_pre mention_0"
RUNS=""; DIRS=""
for s in "$@"; do for v in leak none benign_k1 benign_k3 proposal_k1 proposal_k3; do RUNS="$RUNS runs/sweep14b/$s/$v"; DIRS="$DIRS data/replay_14b/verdict/sweep14b/$s/$v"; done; done
python -u scripts/replay.py $RUNS --episode 2 --phases verdict --free-only --uncompressed --lens-on-gpu --out-root data/replay_14b/verdict --resume || exit 1
python -u scripts/decompose.py $DIRS --positions $POSITIONS || exit 1
echo "verdict part done"
