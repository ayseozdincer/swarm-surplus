#!/bin/bash
# Lane D, beside the rule-off arm on the same server: leak2 and leak_all at ten seeds each, rules off, varied phrasing.
# leak2 runs half its seeds with the addressee named and half broadcast. Ten seeds, not twenty, to stay inside the budget.
cd /workspace/swarm
export VLLM_KEY=local-dummy
COMMON="--warmup 2 --post 1 --max-tokens 12000 --model Qwen/Qwen3-14B --base-url http://127.0.0.1:8000/v1 --api-key-env VLLM_KEY --timeout 7200"
PY=/root/vllm-venv/bin/python
NAME="5001 5004 5007 5010 5013"; BROAD="5031 5034 5037 5040 5043"; TEN="$NAME $BROAD"
( $PY -u scripts/run_plants.py --out runs/sweep2_leak2 --seed-list $NAME --variants none leak2 --leak-phrasing varied --leak-address name $COMMON --concurrency 5 --max-concurrency 40 > /workspace/logs/sweep2_leak2_a.log 2>&1 ) &
( $PY -u scripts/run_plants.py --out runs/sweep2_leak2 --seed-list $BROAD --variants none leak2 --leak-phrasing varied --leak-address broadcast $COMMON --concurrency 5 --max-concurrency 40 > /workspace/logs/sweep2_leak2_b.log 2>&1 ) &
( $PY -u scripts/run_plants.py --out runs/sweep2_multi --seed-list $TEN --blind-per-episode 3 --variants none leak_all --leak-phrasing varied $COMMON --concurrency 10 --max-concurrency 60 > /workspace/logs/sweep2_multi.log 2>&1 ) &
wait
echo "leak2 and multi exit $(date)"
