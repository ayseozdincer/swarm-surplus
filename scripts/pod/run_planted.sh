#!/bin/bash
# Lane E phase 2c generation on the pod against the local vLLM server (scripts/pod/vllm_serve.sh). Code-task targets at the
# plant episode: with two warm-up episodes the blind agent is names[(seed + 2) % 6], a code specialist when seed % 6 is 0 or 3,
# so seeds 8001, 8004, ... Half the seeds name the target and half address nobody (--leak-address half, the default here).
cd /workspace/swarm
export VLLM_KEY=local-dummy
SEEDS=$(seq 8001 3 ${LAST_SEED:-8058})
exec /root/vllm-venv/bin/python -u scripts/run_plants.py --mode planted_file --out runs/planted --seed-list $SEEDS \
  --max-tokens 12000 --model Qwen/Qwen3-14B --base-url http://127.0.0.1:8000/v1 --api-key-env VLLM_KEY \
  --concurrency ${SEED_CONCURRENCY:-5} --max-concurrency ${CALL_CONCURRENCY:-64} --timeout 7200
