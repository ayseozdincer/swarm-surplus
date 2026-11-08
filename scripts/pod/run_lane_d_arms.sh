#!/bin/bash
# Lane D, on the generation pod after sweep2: restart vLLM with CPU KV offloading (plain config if that fails to come up),
# then the rule-off arm, then leak2, leak_all and the solvable-task leak, all on the same twenty seeds, rules off.
cd /workspace/swarm
export VLLM_KEY=local-dummy
SEEDS=$(seq 5001 3 5058); FIRST=$(seq 5001 3 5028); SECOND=$(seq 5031 3 5058)
COMMON="--warmup 2 --post 1 --max-tokens 12000 --model Qwen/Qwen3-14B --base-url http://127.0.0.1:8000/v1 --api-key-env VLLM_KEY --timeout 7200"
PY=/root/vllm-venv/bin/python
until grep -q '^exit ' /workspace/logs/sweep2.log; do sleep 30; done
healthy() { for i in $(seq 1 40); do curl -sf -H "Authorization: Bearer local-dummy" http://127.0.0.1:8000/v1/models >/dev/null && return 0; sleep 10; done; return 1; }
pkill -f '[v]llm serve'; sleep 20
KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-120} nohup setsid bash scripts/pod/vllm_serve.sh > /workspace/logs/vllm2.log 2>&1 < /dev/null &
if ! healthy; then
  echo "offload server did not come up, plain config" ; pkill -f '[v]llm serve'; sleep 20
  nohup setsid bash scripts/pod/vllm_serve.sh > /workspace/logs/vllm3.log 2>&1 < /dev/null &
  healthy || exit 1
fi
echo "server up $(date)"
$PY -u scripts/run_plants.py --out runs/sweep2_norule --seed-list $SEEDS --variants none benign leak command --k 1 3 $COMMON \
  --concurrency ${C1:-8} --max-concurrency 128 > /workspace/logs/sweep2_norule.log 2>&1
echo "norule exit $? $(date)"
( $PY -u scripts/run_plants.py --out runs/sweep2_leak2 --seed-list $FIRST --variants none leak2 --leak-phrasing varied --leak-address name $COMMON --concurrency 4 --max-concurrency 48 > /workspace/logs/sweep2_leak2_a.log 2>&1 ) &
( $PY -u scripts/run_plants.py --out runs/sweep2_leak2 --seed-list $SECOND --variants none leak2 --leak-phrasing varied --leak-address broadcast $COMMON --concurrency 4 --max-concurrency 48 > /workspace/logs/sweep2_leak2_b.log 2>&1 ) &
( $PY -u scripts/run_plants.py --out runs/sweep2_multi --seed-list $SEEDS --blind-per-episode 3 --variants none leak_all --leak-phrasing varied $COMMON --concurrency 4 --max-concurrency 48 > /workspace/logs/sweep2_multi.log 2>&1 ) &
wait
echo "leak2 and multi exit $(date)"
( $PY -u scripts/run_plants.py --out runs/sweep2_solvable --seed-list $FIRST --variants none leak_solvable --leak-phrasing varied --leak-address name $COMMON --concurrency 5 --max-concurrency 64 > /workspace/logs/sweep2_solvable_a.log 2>&1 ) &
( $PY -u scripts/run_plants.py --out runs/sweep2_solvable --seed-list $SECOND --variants none leak_solvable --leak-phrasing varied --leak-address broadcast $COMMON --concurrency 5 --max-concurrency 64 > /workspace/logs/sweep2_solvable_b.log 2>&1 ) &
wait
echo "solvable exit $(date)"
