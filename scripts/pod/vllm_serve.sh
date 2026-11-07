#!/bin/bash
# Qwen3-14B on vLLM, bf16, OpenAI-compatible on 127.0.0.1:8000. Tool calls through the hermes parser, thinking split
# out by the qwen3 reasoning parser. The harness runs on the pod against this endpoint, so nothing is exposed.
#   GPU_UTIL   fraction of the card vLLM takes (default 0.92; lower it to share the card with the replay)
# The FlashInfer sampler JIT-builds a kernel at startup and the image has no matching toolchain, so sampling uses
# vLLM's PyTorch sampler instead (same distribution).
export HF_HOME=/root/hf VLLM_API_KEY=${VLLM_API_KEY:-local-dummy} VLLM_USE_FLASHINFER_SAMPLER=0 PATH=/root/vllm-venv/bin:$PATH
exec /root/vllm-venv/bin/vllm serve Qwen/Qwen3-14B --served-model-name Qwen/Qwen3-14B \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 40960 \
  --gpu-memory-utilization ${GPU_UTIL:-0.92} --enable-prefix-caching --max-num-seqs ${MAX_SEQS:-256} \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --generation-config vllm --disable-uvicorn-access-log
