# Lane C after the lead session's reorder (no compacted reference). Stage 1, the blind agent's solve turn in every branch
# of every seed with the per-token trace and the placebo reference, then its J-space decomposition. Stage 2, the free agents'
# verdict turns, actual and kept reference only, then theirs.
. /workspace/venv/bin/activate
cd /workspace/swarm
export HF_HOME=/workspace/hf SWARM_MODEL=qwen3_14b PYTHONPATH=/workspace/swarm/third_party/maj-lens:/workspace/swarm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
POSITIONS="prompt_last first_generated first_thought after_think answer_pre verdict_pre end_of_response mention_pre mention_0"
BLIND=data/replay_14b/blind/sweep14b
VERDICT=data/replay_14b/verdict/sweep14b
python -u experiments/lane_c_sweep14b/blind.py runs/sweep14b/seed3??? --out-root $BLIND --resume || exit 1
python -u scripts/decompose.py $BLIND/seed*/* --positions $POSITIONS || exit 1
echo "stage 1 done"
RUNS=""
for s in runs/sweep14b/seed3???; do for v in leak none benign_k1 benign_k3 proposal_k1 proposal_k3; do RUNS="$RUNS $s/$v"; done; done
python -u scripts/replay.py $RUNS --episode 2 --phases verdict --free-only --uncompressed --lens-on-gpu --out-root data/replay_14b/verdict --resume || exit 1
python -u scripts/decompose.py $VERDICT/seed*/* --positions $POSITIONS || exit 1
echo "stage 2 done"
