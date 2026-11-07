#!/bin/bash
# Lane D pod setup, run from /workspace/swarm, as run on 2026-11-07 (vllm 0.33.0, torch 2.14.0+cu130, transformers
# 5.20.2). One venv on the container disk serves both vLLM and the replay; /workspace is a network filesystem and slow
# to import or load weights from. The image's system Python is externally managed, hence --break-system-packages for
# the downloader only.
set -euo pipefail
mkdir -p /workspace/logs /root/artifacts
pip install -q --break-system-packages "huggingface_hub[hf_xet]" uv
HF_HOME=/root/hf python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-14B'))"
ln -sfn "$(ls -d /root/hf/hub/models--Qwen--Qwen3-14B/snapshots/*)" /root/artifacts/qwen3_14b
# The converged Qwen3-14B lens, finished format (a `J` key, layers 0 to 38, float16).
python3 -c "
from huggingface_hub import hf_hub_download
for f in ['Qwen3-14B_jacobian_lens.pt', 'config.yaml', 'Qwen3-14B_convergence.csv']:
    print(hf_hub_download('neuronpedia/jacobian-lens', 'qwen3-14b/jlens/Salesforce-wikitext/' + f,
                          revision='b87d2a42d715f50e3a8359d9c29ccdd5514509d2', local_dir='/root/artifacts/lens_candidate'))"
# vLLM 0.33 needs torch 2.14, built for CUDA 12.6 or 13.0 only; on a host driver older than CUDA 13 the cu130 build
# runs through the forward compatibility libraries (datacenter GPUs), see vllm_serve.sh
nvidia-smi | grep -q "CUDA Version: 13" || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-compat-13-0
[ -x /root/vllm-venv/bin/python ] || uv venv -q /root/vllm-venv --python 3.12
VIRTUAL_ENV=/root/vllm-venv uv pip install -q vllm==0.33.0 openai ninja pyyaml scipy scikit-learn accelerate==1.15.0
# jlens at the J-space repo's pin, without letting it touch vLLM's torch
[ -d /root/jlens-src ] || git clone -q https://github.com/anthropics/jacobian-lens.git /root/jlens-src
git -C /root/jlens-src checkout -q 581d398613e5602a5af361e1c34d3a92ea82ba8e
VIRTUAL_ENV=/root/vllm-venv uv pip install -q --no-deps /root/jlens-src
/root/vllm-venv/bin/python -c "import vllm, torch, jlens; print('vllm', vllm.__version__, 'torch', torch.__version__)"
echo setup done
