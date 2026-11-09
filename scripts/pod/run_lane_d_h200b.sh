#!/bin/bash
# Lane D, second server card (H200 y5ggibbnlcv2c0) from 22:45: the 20 new forbidden_leak seeds, rules confidential, code-task
# blind agents, six seeds in flight. Moved here from the first H200, where it was paused behind the solvable arm.
cd /workspace/swarm
export VLLM_KEY=local-dummy
PY=/root/vllm-venv/bin/python
healthy() { for i in $(seq 1 40); do curl -sf -H "Authorization: Bearer local-dummy" http://127.0.0.1:8000/v1/models >/dev/null && return 0; sleep 10; done; return 1; }
KV_OFFLOAD_GB=100 nohup setsid bash scripts/pod/vllm_serve.sh > /workspace/logs/vllm.log 2>&1 < /dev/null &
healthy || { pkill -f '[v]llm serve'; sleep 20; nohup setsid bash scripts/pod/vllm_serve.sh > /workspace/logs/vllm.log 2>&1 < /dev/null & healthy || exit 1; }
echo "server up $(date)"
$PY -u scripts/run_plants.py --out runs/sweep2_forbidden2 --seed-list $(seq 5061 3 5118) --rules confidential --variants none forbidden_leak \
  --warmup 2 --post 1 --max-tokens 12000 --model Qwen/Qwen3-14B --base-url http://127.0.0.1:8000/v1 --api-key-env VLLM_KEY --timeout 7200 \
  --concurrency 6 --max-concurrency 72 > /workspace/logs/q_forbidden2.log 2>&1
echo "forbidden2 exit $(date)"
