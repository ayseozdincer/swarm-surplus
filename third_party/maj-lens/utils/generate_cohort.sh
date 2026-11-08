#!/bin/bash
# DeepSeek-V4-Flash's own NARCBench transcripts: NARCBench's unmodified Core and Stego generation code, with
# DeepSeek-V4-Flash as every agent, served by vLLM (OpenAI-compatible API, as NARCBench expects). NARCBench's
# defaults otherwise: temperature 0.7, thinking off, 50 Core scenarios x (collusion, control), 20 Stego shoes x
# (collusion, control). Needs the GPU free (no other model loaded). vLLM runs from its own environment:
#   python3.12 -m venv .vllm-venv && .vllm-venv/bin/pip install uv &&
#   .vllm-venv/bin/uv pip install --python .vllm-venv/bin/python vllm==0.30.0 --torch-backend=auto
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
VLLM=${VLLM:-$PWD/.vllm-venv/bin/vllm}
OUT=$PWD/artifacts/narcbench/scenarios/deepseek_v4_flash
URL=http://127.0.0.1:8020/v1
M=deepseek-v4-flash
$VLLM serve artifacts/deepseek_v4_flash --served-model-name $M --port 8020 --max-model-len 16384 \
  --gpu-memory-utilization 0.93 --trust-remote-code > logs/vllm_serve.log 2>&1 &
SERVER=$!
trap 'kill $SERVER' EXIT
until curl -sf $URL/models > /dev/null; do sleep 10; kill -0 $SERVER || exit 1; done
cd narcbench/generation
$PY -u core.py --scenarios 50 --mode both --parallel 4 --model $M --base-url $URL --output-dir $OUT/core &
for k in 0 1 2 3; do   # four Stego streams of five shoes; stego.py runs one shoe (collusion + control) per call
  ( for s in $(seq $k 4 19); do $PY -u stego.py --shoe $s --model $M --base-url $URL --output-dir $OUT/stego; done ) &
done
wait
