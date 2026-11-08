# Lane C final readout over both stages: every part of scripts/lane_c.py, then the window score again.
. /workspace/venv/bin/activate
cd /workspace/swarm
export HF_HOME=/workspace/hf SWARM_MODEL=qwen3_14b PYTHONPATH=/workspace/swarm/third_party/maj-lens:/workspace/swarm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/lane_c.py --roots data/replay_14b/blind/sweep14b data/replay_14b/verdict/sweep14b --out results/lane_c/readout.json
