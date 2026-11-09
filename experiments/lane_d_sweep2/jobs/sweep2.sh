#!/bin/bash
# Lane D sweep2 on the pod against the local vLLM server (scripts/pod/vllm_serve.sh). Code-task blind agents at the
# plant episode: with two warm-up episodes the blind agent is names[(seed + 2) % 6], a code specialist when seed % 6 is
# 0 or 3, so seeds 5001, 5004, ..., 5058.
cd /workspace/swarm
export VLLM_KEY=local-dummy
SEEDS=$(seq 5001 3 5058)
exec /root/vllm-venv/bin/python -u scripts/run_plants.py --out runs/sweep2 --seed-list $SEEDS \
  --variants none benign leak forbidden_leak command --k 1 3 --rules confidential --warmup 2 --post 1 \
  --max-tokens 12000 --model Qwen/Qwen3-14B --base-url http://127.0.0.1:8000/v1 --api-key-env VLLM_KEY \
  --concurrency ${SEED_CONCURRENCY:-10} --max-concurrency ${CALL_CONCURRENCY:-96} --timeout 7200
