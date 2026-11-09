# Lane B, after the Stego replay frees the card: W4's padded placebo group for Transfer (honest agents, own reference
# prompt plus the donor brief), its J-space shares, and the Transfer report again with all three placebo-style groups.
. /workspace/venv/bin/activate
cd /workspace/swarm/third_party/maj-lens
export HF_HOME=/workspace/hf STUDY_MODELS=qwen3_32b PYTHONPATH=.:/workspace/swarm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "^exit" /workspace/logs/lane_b_replay.log; do sleep 20; done
grep -q "^exit 0" /workspace/logs/lane_b_replay.log || exit 1
python -u ../../experiments/lane_b_transfer_stego/run.py transfer_padded || exit 1
python -u ../../experiments/lane_b_transfer_stego/run.py follow report_transfer --dirs ../../data/replay_lane_b/transfer_padded --expect 72
