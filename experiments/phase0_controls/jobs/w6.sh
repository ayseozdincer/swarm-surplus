# W6 on the pod. First the pilot's blind turn in the leak, none and benign_k1 branches for the first-mention position, the
# W3 follow-up, so it reports early. Then the sweep1 replay of episode 2, seeds 2000 to 2002, every variant, and the verdict
# turns of the later runs whose reasoning cites the notes, since seeds 2000 to 2002 have only one such turn.
. /workspace/venv/bin/activate
cd /workspace/swarm
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/swarm/third_party/maj-lens:/workspace/swarm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/replay.py runs/pilot/seed1000/leak runs/pilot/seed1000/none runs/pilot/seed1000/benign_k1 --episode 2 --agents carol --phases solve --out-root data/replay_mention || exit 1
python -u scripts/decompose.py data/replay_mention/pilot/seed1000/leak data/replay_mention/pilot/seed1000/none data/replay_mention/pilot/seed1000/benign_k1 || exit 1
python -u experiments/phase0_controls/mention_readout.py || exit 1
echo "mention done"
python -u experiments/phase0_controls/compact_reference.py || exit 1
echo "compact done"
RUNS=""
for s in 2000 2001 2002; do for v in none benign_k1 benign_k3 leak proposal_k1 proposal_k3; do RUNS="$RUNS runs/sweep1/seed$s/$v"; done; done
python -u scripts/replay.py $RUNS --episode 2 || exit 1
python -u scripts/replay.py $(cat experiments/phase0_controls/jobs/cited_runs.txt) --episode 2 --phases verdict || exit 1
python -u scripts/decompose.py data/replay/sweep1/seed*/* || exit 1
python -u scripts/surplus_table.py experiments/phase0_controls/results/surplus_sweep1.json data/replay/sweep1/seed2000 data/replay/sweep1/seed2001 data/replay/sweep1/seed2002 || exit 1
python -u experiments/phase0_controls/sweep_readout.py
