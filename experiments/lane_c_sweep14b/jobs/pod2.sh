# Lane C, second pod: the found-location composition (needs only the stage-1 trace deltas and indexes), then its share of
# stage 2.
. /workspace/venv/bin/activate
cd /workspace/swarm
export HF_HOME=/workspace/hf SWARM_MODEL=qwen3_14b PYTHONPATH=/workspace/swarm/third_party/maj-lens:/workspace/swarm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u experiments/lane_c_sweep14b/found.py || echo "found failed"
echo "found done"
bash experiments/lane_c_sweep14b/jobs/verdict.sh "$@"
