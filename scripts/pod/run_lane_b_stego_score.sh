# Lane B, after the Stego replay: J-space shares and the Stego report.
. /workspace/venv/bin/activate
cd /workspace/swarm/third_party/maj-lens
export HF_HOME=/workspace/hf STUDY_MODELS=qwen3_32b PYTHONPATH=.:/workspace/swarm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u ../../scripts/lane_b.py decompose report_stego --dirs ../../data/replay_lane_b/stego/*
