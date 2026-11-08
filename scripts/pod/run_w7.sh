# W7 on the pod, after W4 follow-up 4: the GPU part of the direction test, part B on the sweep1 replay, then the 1c
# direction edge on the Core arrays (CPU).
. /workspace/venv/bin/activate
cd /workspace/swarm
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/swarm/third_party/maj-lens:/workspace/swarm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/direction_test.py collect || exit 1
python -u scripts/direction_test.py analyze || exit 1
python -u scripts/detection_run.py
python -u scripts/direction_edges.py
python -u scripts/concealment_read.py collect
python -u scripts/concealment_read.py analyze
