# Lane C after part A's replay job: decompose and read part A, then part B (benign_k3, proposal_k1, proposal_k3) the same
# way, then every readout over all six variants.
. /workspace/venv/bin/activate
cd /workspace/swarm
export HF_HOME=/workspace/hf SWARM_MODEL=qwen3_14b PYTHONPATH=/workspace/swarm/third_party/maj-lens:/workspace/swarm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q '^exit' /workspace/logs/lane_c_a.log; do sleep 20; done
REPLAY=data/replay_14b/sweep14b
POSITIONS="prompt_last first_generated first_thought after_think answer_pre verdict_pre end_of_response mention_pre mention_0"
python -u scripts/decompose.py $REPLAY/seed*/leak $REPLAY/seed*/none $REPLAY/seed*/benign_k1 --positions $POSITIONS || exit 1
python -u scripts/lane_c.py --out results/lane_c/readout_a.json || echo "readout_a failed"
echo "part A read"
RUNS=""
for s in runs/sweep14b/seed3???; do for v in benign_k3 proposal_k1 proposal_k3; do RUNS="$RUNS $s/$v"; done; done
python -u scripts/replay.py $RUNS --episode 2 --phases solve verdict --free-only --compacted --uncompressed --lens-on-gpu --out-root data/replay_14b --resume || exit 1
python -u scripts/decompose.py $REPLAY/seed*/benign_k3 $REPLAY/seed*/proposal_k1 $REPLAY/seed*/proposal_k3 --positions $POSITIONS || exit 1
python -u scripts/lane_c.py --out results/lane_c/readout.json
