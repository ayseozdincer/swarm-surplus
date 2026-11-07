# Lane C part A on the pod: Qwen3-14B with its released lens. The kernel floor on one honest turn first, then episode 2 of
# the leak, none and benign_k1 branches of every seed, solve and verdict turns of the free agents, kept and compacted reference.
. /workspace/venv/bin/activate
cd /workspace/swarm
export HF_HOME=/workspace/hf SWARM_MODEL=qwen3_14b PYTHONPATH=/workspace/swarm/third_party/maj-lens:/workspace/swarm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f results/lane_c/kernel_floor.json ] || python -u scripts/kernel_floor.py runs/sweep14b/seed3000/none carol --masked runs/sweep14b/seed3000/leak carol --out results/lane_c/kernel_floor.json || exit 1
RUNS=""
for s in runs/sweep14b/seed3???; do for v in leak none benign_k1; do RUNS="$RUNS $s/$v"; done; done
python -u scripts/replay.py $RUNS --episode 2 --phases solve verdict --free-only --compacted --uncompressed --lens-on-gpu --out-root data/replay_14b --resume
