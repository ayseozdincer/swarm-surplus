# Lane C final readout over both stages: every part of experiments/lane_c_sweep14b/score.py, then the window score again.
. /workspace/venv/bin/activate
cd /workspace/swarm
export HF_HOME=/workspace/hf SWARM_MODEL=qwen3_14b PYTHONPATH=/workspace/swarm/third_party/maj-lens:/workspace/swarm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u experiments/lane_c_sweep14b/score.py --roots data/replay_14b/blind/sweep14b data/replay_14b/verdict/sweep14b --out experiments/lane_c_sweep14b/results/readout.json
