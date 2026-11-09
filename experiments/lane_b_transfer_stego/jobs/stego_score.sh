# Lane B: J-space shares of each Stego run as its replay lands (one lens load), then the Stego report.
. /workspace/venv/bin/activate
cd /workspace/swarm/third_party/maj-lens
export HF_HOME=/workspace/hf STUDY_MODELS=qwen3_32b PYTHONPATH=.:/workspace/swarm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u ../../experiments/lane_b_transfer_stego/run.py follow report_stego --dirs ../../data/replay_lane_b/stego --expect 40
