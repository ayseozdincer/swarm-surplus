#!/bin/bash
# Lane D, second evening queue on one server card, one arm at a time with few seeds in flight so the prefix cache holds:
#   1 solvable, 20 seeds, none + leak_solvable (right and wrong values, three branches a seed), rules off, varied phrasing,
#     5001-5028 addressed by name and 5031-5058 broadcast
#   2 forbidden_leak, 20 new code-task seeds 5061-5118, rules confidential, none + forbidden_leak (for the concealer count)
#   3 leak_all, 10 seeds, three blind targets an episode, rules off, varied phrasing
# The partial runs of the first attempt at solvable and leak_all are moved aside, not reused.
cd /workspace/swarm
export VLLM_KEY=local-dummy
COMMON="--warmup 2 --post 1 --max-tokens 12000 --model Qwen/Qwen3-14B --base-url http://127.0.0.1:8000/v1 --api-key-env VLLM_KEY --timeout 7200"
PY=/root/vllm-venv/bin/python
healthy() { for i in $(seq 1 40); do curl -sf -H "Authorization: Bearer local-dummy" http://127.0.0.1:8000/v1/models >/dev/null && return 0; sleep 10; done; return 1; }
for d in sweep2_solvable sweep2_multi; do [ -d runs/$d ] && [ ! -d runs/${d}_partial1 ] && mv runs/$d runs/${d}_partial1; done
KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-100} nohup setsid bash scripts/pod/vllm_serve.sh > /workspace/logs/vllm4.log 2>&1 < /dev/null &
healthy || { pkill -f '[v]llm serve'; sleep 20; nohup setsid bash scripts/pod/vllm_serve.sh > /workspace/logs/vllm4.log 2>&1 < /dev/null & healthy || exit 1; }
echo "server up $(date)"
( $PY -u scripts/run_plants.py --out runs/sweep2_solvable --seed-list $(seq 5001 3 5028) --variants none leak_solvable --leak-phrasing varied --leak-address name $COMMON --concurrency 3 --max-concurrency 40 > /workspace/logs/q_solvable_a.log 2>&1 ) &
( $PY -u scripts/run_plants.py --out runs/sweep2_solvable --seed-list $(seq 5031 3 5058) --variants none leak_solvable --leak-phrasing varied --leak-address broadcast $COMMON --concurrency 3 --max-concurrency 40 > /workspace/logs/q_solvable_b.log 2>&1 ) &
wait
echo "solvable exit $(date)"
$PY -u scripts/run_plants.py --out runs/sweep2_forbidden2 --seed-list $(seq 5061 3 5118) --rules confidential --variants none forbidden_leak $COMMON --concurrency 5 --max-concurrency 64 > /workspace/logs/q_forbidden2.log 2>&1
echo "forbidden2 exit $(date)"
$PY -u scripts/run_plants.py --out runs/sweep2_multi --seed-list 5001 5004 5007 5010 5013 5031 5034 5037 5040 5043 --blind-per-episode 3 --variants none leak_all --leak-phrasing varied $COMMON --concurrency 5 --max-concurrency 64 > /workspace/logs/q_multi.log 2>&1
echo "multi exit $(date)"
