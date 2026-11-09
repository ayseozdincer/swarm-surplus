#!/bin/bash
# Lane D: leak_all, 10 seeds, three blind targets an episode, rules off, varied phrasing; started beside the solvable tail
cd /workspace/swarm
export VLLM_KEY=local-dummy
/root/vllm-venv/bin/python -u scripts/run_plants.py --out runs/sweep2_multi --seed-list 5001 5004 5007 5010 5013 5031 5034 5037 5040 5043 --blind-per-episode 3 --variants none leak_all --leak-phrasing varied --warmup 2 --post 1 --max-tokens 12000 --model Qwen/Qwen3-14B --base-url http://127.0.0.1:8000/v1 --api-key-env VLLM_KEY --timeout 7200 --concurrency 6 --max-concurrency 72 > /workspace/logs/q_multi.log 2>&1
echo "multi exit $(date)"
