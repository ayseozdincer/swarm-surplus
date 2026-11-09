#!/bin/bash
# Lane G, after generation: stop vLLM, merge the two output directories into runs/sweep2_leak2, replay the blind agents'
# episode-2 solve turns in leak2 and none with one reference per note (scripts/pod/replay_sweep2.py plan leak2, unchanged
# from lane D), then score: the pre-registered test (scripts/lane_g_leak2.py), lane D's leak2_uptake.py on the new turns,
# and the behaviour summary.
cd /workspace/swarm
pkill -f '[v]llm serve'; sleep 30
export SWARM_MODEL=qwen3_14b SWARM_ARTIFACTS=/root/artifacts HF_HOME=/root/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -d /usr/local/cuda-13.0/compat ] && export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY=/root/vllm-venv/bin/python
mkdir -p results/lane_g data/replay_sweep2
$PY scripts/lane_g_merge.py runs/leak2_name runs/leak2_broad runs/sweep2_leak2 || exit 1
N=$(ls -d runs/sweep2_leak2/seed*/ 2>/dev/null | wc -l)
[ "$N" -gt 0 ] || { echo "no complete seed"; exit 1; }
echo "replaying $N seeds $(date -u +%FT%TZ)"
$PY -u scripts/pod/replay_sweep2.py runs --out-root data/replay_sweep2 --order "sweep2_leak2:1" --expect sweep2_leak2=$N \
  ${STOP_AT:+--stop-at $STOP_AT} || exit 1
echo "replay exit $(date -u +%FT%TZ)"
$PY scripts/summarize.py runs/sweep2_leak2 > results/lane_g/summary_leak2.txt 2>&1
$PY scripts/lane_g_leak2.py --replay data/replay_sweep2 --runs runs --out results/lane_g/leak2_scores.json > results/lane_g/leak2_scores.txt 2>&1 || echo "lane_g_leak2.py failed"
cp results/lane_d/leak2_uptake.json /workspace/lane_d_leak2_uptake.bak.json
$PY scripts/leak2_uptake.py data/replay_sweep2 runs > results/lane_g/leak2_uptake.txt 2>&1 && mv results/lane_d/leak2_uptake.json results/lane_g/leak2_uptake.json
cp /workspace/lane_d_leak2_uptake.bak.json results/lane_d/leak2_uptake.json
echo "score exit $(date -u +%FT%TZ)"
