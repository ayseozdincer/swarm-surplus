# W3 on the pod: replay episode 2 of the pilot branches, then the GPU checks.
. /workspace/venv/bin/activate
cd /workspace/swarm
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/swarm/third_party/maj-lens:/workspace/swarm
python -u scripts/replay.py runs/pilot/seed1000/leak runs/pilot/seed1000/none runs/pilot/seed1000/benign_k1 runs/pilot/seed1000/proposal_k1 --episode 2
python -u tests/gpu_checks.py
