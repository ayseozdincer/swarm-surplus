#!/bin/bash
# Lane G, after generation: stop vLLM, merge the two output directories into runs/lane_g/sweep2_leak2, replay the blind agents'
# episode-2 solve turns in leak2 and none with one reference per note (experiments/lane_d_sweep2/replay_sweep2.py plan leak2, unchanged
# from lane D), then score: the pre-registered test (experiments/lane_g_two_notes/score_leak2.py), lane D's leak2_uptake.py on the new turns,
# and the behaviour summary.
cd /workspace/swarm
pkill -f '[v]llm serve'; sleep 30
export SWARM_MODEL=qwen3_14b SWARM_ARTIFACTS=/root/artifacts HF_HOME=/root/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -d /usr/local/cuda-13.0/compat ] && export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY=/root/vllm-venv/bin/python
mkdir -p experiments/lane_g_two_notes/results data/replay_lane_g
$PY experiments/lane_g_two_notes/merge_runs.py runs/leak2_name runs/leak2_broad runs/lane_g/sweep2_leak2 || exit 1
N=$(ls -d runs/lane_g/sweep2_leak2/seed*/ 2>/dev/null | wc -l)
[ "$N" -gt 0 ] || { echo "no complete seed"; exit 1; }
echo "replaying $N seeds $(date -u +%FT%TZ)"
$PY -u experiments/lane_d_sweep2/replay_sweep2.py runs/lane_g --out-root data/replay_lane_g --order "sweep2_leak2:1" --expect sweep2_leak2=$N \
  ${STOP_AT:+--stop-at $STOP_AT} || exit 1
echo "replay exit $(date -u +%FT%TZ)"
$PY scripts/summarize.py runs/lane_g/sweep2_leak2 > experiments/lane_g_two_notes/results/summary_leak2.txt 2>&1
$PY experiments/lane_g_two_notes/score_leak2.py --replay data/replay_lane_g --runs runs/lane_g --out experiments/lane_g_two_notes/results/leak2_scores.json > experiments/lane_g_two_notes/results/leak2_scores.txt 2>&1 || echo "experiments/lane_g_two_notes/score_leak2.py failed"
cp experiments/lane_d_sweep2/results/leak2_uptake.json /workspace/lane_d_leak2_uptake.bak.json
$PY experiments/lane_d_sweep2/leak2_uptake.py data/replay_lane_g runs/lane_g > experiments/lane_g_two_notes/results/leak2_uptake.txt 2>&1 && mv experiments/lane_d_sweep2/results/leak2_uptake.json experiments/lane_g_two_notes/results/leak2_uptake.json
cp /workspace/lane_d_leak2_uptake.bak.json experiments/lane_d_sweep2/results/leak2_uptake.json
echo "score exit $(date -u +%FT%TZ)"
