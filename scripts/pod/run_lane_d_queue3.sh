#!/bin/bash
# Lane D, second queue as rerun at 21:40 on the running server: the solvable arm moved to seeds whose episode-2 blind agent is
# not a code specialist, so both code agents are free and the harness picks a code-task target (on the sweep2 seeds a code
# agent is always blind and the target falls to records or data). Then 20 new forbidden_leak seeds, then leak_all.
cd /workspace/swarm
export VLLM_KEY=local-dummy
COMMON="--warmup 2 --post 1 --max-tokens 12000 --model Qwen/Qwen3-14B --base-url http://127.0.0.1:8000/v1 --api-key-env VLLM_KEY --timeout 7200"
PY=/root/vllm-venv/bin/python
A="5000 5002 5003 5005 5006 5008 5009 5011 5012 5014"; B="5015 5017 5018 5020 5021 5023 5024 5026 5027 5029"
( $PY -u scripts/run_plants.py --out runs/sweep2_solvable --seed-list $A --variants none leak_solvable --leak-phrasing varied --leak-address name $COMMON --concurrency 3 --max-concurrency 40 > /workspace/logs/q_solvable_a.log 2>&1 ) &
( $PY -u scripts/run_plants.py --out runs/sweep2_solvable --seed-list $B --variants none leak_solvable --leak-phrasing varied --leak-address broadcast $COMMON --concurrency 3 --max-concurrency 40 > /workspace/logs/q_solvable_b.log 2>&1 ) &
wait
echo "solvable exit $(date)"
$PY -u scripts/run_plants.py --out runs/sweep2_forbidden2 --seed-list $(seq 5061 3 5118) --rules confidential --variants none forbidden_leak $COMMON --concurrency 5 --max-concurrency 64 > /workspace/logs/q_forbidden2.log 2>&1
echo "forbidden2 exit $(date)"
$PY -u scripts/run_plants.py --out runs/sweep2_multi --seed-list 5001 5004 5007 5010 5013 5031 5034 5037 5040 5043 --blind-per-episode 3 --variants none leak_all --leak-phrasing varied $COMMON --concurrency 5 --max-concurrency 64 > /workspace/logs/q_multi.log 2>&1
echo "multi exit $(date)"
