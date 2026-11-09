#!/bin/bash
# Lane D, method_tip arm on one server card: none, method_tip, leak_solvable (right value), the solvable arm's twenty seeds, so the
# target is a code agent, rules off, varied phrasing, 5000-5014 addressed by name and 5015-5029 broadcast. All seeds in flight,
# since sweep2 showed that gives at least the throughput of few (experiments/lane_d_sweep2/REPORT.md).
cd /workspace/swarm
export VLLM_KEY=local-dummy
COMMON="--warmup 2 --post 1 --max-tokens 12000 --model Qwen/Qwen3-14B --base-url http://127.0.0.1:8000/v1 --api-key-env VLLM_KEY --timeout 7200"
VARIANTS="--variants none method_tip leak_solvable --solvable-values correct --leak-phrasing varied"
PY=/root/vllm-venv/bin/python
A="5000 5002 5003 5005 5006 5008 5009 5011 5012 5014"; B="5015 5017 5018 5020 5021 5023 5024 5026 5027 5029"
( $PY -u scripts/run_plants.py --out runs/sweep2_methodtip --seed-list $A $VARIANTS --leak-address name $COMMON --concurrency 10 --max-concurrency 64 > /workspace/logs/q_tip_a.log 2>&1 ) &
( $PY -u scripts/run_plants.py --out runs/sweep2_methodtip --seed-list $B $VARIANTS --leak-address broadcast $COMMON --concurrency 10 --max-concurrency 64 > /workspace/logs/q_tip_b.log 2>&1 ) &
wait
echo "methodtip exit $(date)"
