#!/bin/bash
# Lane D, beside the other arms: the solvable-task leak at ten seeds, rules off, varied phrasing, half the seeds addressed by
# name and half broadcast; leak_solvable runs both sub-kinds (the right answer and a wrong value) next to none.
cd /workspace/swarm
export VLLM_KEY=local-dummy
COMMON="--warmup 2 --post 1 --max-tokens 12000 --model Qwen/Qwen3-14B --base-url http://127.0.0.1:8000/v1 --api-key-env VLLM_KEY --timeout 7200"
PY=/root/vllm-venv/bin/python
( $PY -u scripts/run_plants.py --out runs/sweep2_solvable --seed-list 5001 5004 5007 5010 5013 --variants none leak_solvable --leak-phrasing varied --leak-address name $COMMON --concurrency 5 --max-concurrency 48 > /workspace/logs/sweep2_solvable_a.log 2>&1 ) &
( $PY -u scripts/run_plants.py --out runs/sweep2_solvable --seed-list 5031 5034 5037 5040 5043 --variants none leak_solvable --leak-phrasing varied --leak-address broadcast $COMMON --concurrency 5 --max-concurrency 48 > /workspace/logs/sweep2_solvable_b.log 2>&1 ) &
wait
echo "solvable exit $(date)"
