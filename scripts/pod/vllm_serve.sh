#!/bin/bash
# Qwen3-14B on vLLM, bf16, OpenAI-compatible on 127.0.0.1:8000. Tool calls through the hermes parser, thinking split
# out by the qwen3 reasoning parser. The harness runs on the pod against this endpoint, so nothing is exposed.
#   GPU_UTIL       fraction of the card vLLM takes (default 0.92; lower it to share the card with the replay)
#   KV_OFFLOAD_GB  CPU buffer (GB, under the free /dev/shm, 119 GB on the H200 pod) for vLLM's native KV offloading, so prefixes evicted from the card reload instead of being
#                  recomputed (sweep2 spent most of its time recomputing prompts with the GPU cache full)
# The FlashInfer sampler JIT-builds a kernel at startup and the image has no matching toolchain, so sampling uses
# vLLM's PyTorch sampler instead (same distribution).
export HF_HOME=/root/hf VLLM_API_KEY=${VLLM_API_KEY:-local-dummy} VLLM_USE_FLASHINFER_SAMPLER=0 PATH=/root/vllm-venv/bin:$PATH
# A host driver older than CUDA 13 (the H200 pod had 575, CUDA 12.9) runs the cu130 build through NVIDIA's forward
# compatibility libraries (apt install cuda-compat-13-0).
[ -d /usr/local/cuda-13.0/compat ] && export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec /root/vllm-venv/bin/vllm serve Qwen/Qwen3-14B --served-model-name Qwen/Qwen3-14B \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 40960 \
  --gpu-memory-utilization ${GPU_UTIL:-0.92} --enable-prefix-caching --max-num-seqs ${MAX_SEQS:-256} \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --generation-config vllm --disable-uvicorn-access-log \
  ${KV_OFFLOAD_GB:+--kv-offloading-size $KV_OFFLOAD_GB --kv-offloading-backend native}
