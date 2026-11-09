# W3 on the pod: replay episode 2 of the pilot branches with the attention kernel pinned, decompose, read the leak.
. /workspace/venv/bin/activate
cd /workspace/swarm
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/swarm/third_party/maj-lens:/workspace/swarm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/replay.py runs/pilot/seed1000/leak runs/pilot/seed1000/none runs/pilot/seed1000/benign_k1 runs/pilot/seed1000/proposal_k1 --episode 2 || exit 1
python -u scripts/decompose.py data/replay/pilot/seed1000/leak data/replay/pilot/seed1000/none data/replay/pilot/seed1000/benign_k1 data/replay/pilot/seed1000/proposal_k1 || exit 1
python -u experiments/phase0_controls/leak_readout.py
