# The rest of W6 after the W4 follow-ups: the sweep1 replay, resumed from any run directory already finished.
. /workspace/venv/bin/activate
cd /workspace/swarm
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/swarm/third_party/maj-lens:/workspace/swarm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUNS=""
for s in 2000 2001 2002; do for v in none benign_k1 benign_k3 leak proposal_k1 proposal_k3; do RUNS="$RUNS runs/sweep1/seed$s/$v"; done; done
python -u scripts/replay.py $RUNS --episode 2 --resume || exit 1
python -u scripts/replay.py $(cat experiments/phase0_controls/jobs/cited_runs.txt) --episode 2 --phases verdict --resume || exit 1
python -u scripts/decompose.py data/replay/sweep1/seed*/* || exit 1
python -u scripts/surplus_table.py experiments/phase0_controls/results/surplus_sweep1.json data/replay/sweep1/seed2000 data/replay/sweep1/seed2001 data/replay/sweep1/seed2002 || exit 1
python -u experiments/phase0_controls/sweep_readout.py
