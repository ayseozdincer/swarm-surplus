#!/bin/bash
# Lane D, at 21:55: the 20 new forbidden_leak seeds start beside the solvable arm (the cache was half empty at six solvable
# seeds in flight), then leak_all once both have finished.
cd /workspace/swarm
export VLLM_KEY=local-dummy
COMMON="--warmup 2 --post 1 --max-tokens 12000 --model Qwen/Qwen3-14B --base-url http://127.0.0.1:8000/v1 --api-key-env VLLM_KEY --timeout 7200"
PY=/root/vllm-venv/bin/python
$PY -u scripts/run_plants.py --out runs/sweep2_forbidden2 --seed-list $(seq 5061 3 5118) --rules confidential --variants none forbidden_leak $COMMON --concurrency 4 --max-concurrency 48 > /workspace/logs/q_forbidden2.log 2>&1
echo "forbidden2 exit $(date)"
while pgrep -f "[r]uns/sweep2_solvable" >/dev/null; do sleep 30; done
echo "solvable done $(date)"
$PY -u scripts/run_plants.py --out runs/sweep2_multi --seed-list 5001 5004 5007 5010 5013 5031 5034 5037 5040 5043 --blind-per-episode 3 --variants none leak_all --leak-phrasing varied $COMMON --concurrency 5 --max-concurrency 64 > /workspace/logs/q_multi.log 2>&1
echo "multi exit $(date)"
