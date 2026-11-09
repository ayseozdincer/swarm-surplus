#!/bin/bash
# Lane G, on the generation pod: vLLM with CPU KV offloading (plain config if that fails to come up), then twenty new leak2
# seeds, rules off, no post episode, ten with the target named and ten broadcast, one run_plants.py process per output
# directory (runs/leak2_name, runs/leak2_broad; merged by scripts/lane_g_merge.py before the replay). If GEN_DEADLINE (unix
# time) passes while generation runs, both processes are killed and the replay takes the seeds with both branches complete.
cd /workspace/swarm
export VLLM_KEY=local-dummy
PY=/root/vllm-venv/bin/python
COMMON="--variants none leak2 --leak-phrasing varied --rules none --warmup 2 --post 0 --max-tokens 12000 --model Qwen/Qwen3-14B --base-url http://127.0.0.1:8000/v1 --api-key-env VLLM_KEY --timeout 7200"
NAME=$(seq 6000 3 6027); BROAD=$(seq 6030 3 6057)
healthy() { for i in $(seq 1 60); do curl -sf -H "Authorization: Bearer local-dummy" http://127.0.0.1:8000/v1/models >/dev/null && return 0; sleep 10; done; return 1; }
mkdir -p /workspace/logs
KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-100} nohup setsid bash scripts/pod/vllm_serve.sh > /workspace/logs/vllm.log 2>&1 < /dev/null &
if ! healthy; then
  echo "offload server did not come up, plain config $(date -u +%FT%TZ)"; pkill -f '[v]llm serve'; sleep 20
  nohup setsid bash scripts/pod/vllm_serve.sh > /workspace/logs/vllm2.log 2>&1 < /dev/null &
  healthy || { echo "server failed $(date -u +%FT%TZ)"; exit 1; }
fi
echo "server up $(date -u +%FT%TZ)"
$PY -u scripts/run_plants.py --out runs/leak2_name --seed-list $NAME --leak-address name $COMMON \
  --concurrency ${C:-10} --max-concurrency ${MC:-64} > /workspace/logs/leak2_name.log 2>&1 &
A=$!
$PY -u scripts/run_plants.py --out runs/leak2_broad --seed-list $BROAD --leak-address broadcast $COMMON \
  --concurrency ${C:-10} --max-concurrency ${MC:-64} > /workspace/logs/leak2_broad.log 2>&1 &
B=$!
while kill -0 $A 2>/dev/null || kill -0 $B 2>/dev/null; do
  if [ -n "$GEN_DEADLINE" ] && [ "$(date +%s)" -ge "$GEN_DEADLINE" ]; then
    echo "generation deadline reached $(date -u +%FT%TZ), killing run_plants"
    kill $A $B 2>/dev/null; sleep 5; pkill -f '[r]un_plants.py'; break
  fi
  sleep 20
done
wait
echo "generation exit $(date -u +%FT%TZ)"
