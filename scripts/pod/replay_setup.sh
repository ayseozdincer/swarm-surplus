#!/bin/bash
# Lane D replay-only pod: torch, transformers and jlens in /root/venv (no vLLM), Qwen3-14B and its lens under /root.
# The versions match the generation pod's venv (torch 2.14.0+cu130, transformers 5.20.2).
set -euo pipefail
mkdir -p /workspace/logs /root/artifacts
pip install -q --break-system-packages "huggingface_hub[hf_xet]" uv
nvidia-smi | grep -q "CUDA Version: 13" || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-compat-13-0
HF_HOME=/root/hf python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-14B'))"
ln -sfn "$(ls -d /root/hf/hub/models--Qwen--Qwen3-14B/snapshots/*)" /root/artifacts/qwen3_14b
python3 -c "
from huggingface_hub import hf_hub_download
for f in ['Qwen3-14B_jacobian_lens.pt', 'config.yaml', 'Qwen3-14B_convergence.csv']:
    print(hf_hub_download('neuronpedia/jacobian-lens', 'qwen3-14b/jlens/Salesforce-wikitext/' + f,
                          revision='b87d2a42d715f50e3a8359d9c29ccdd5514509d2', local_dir='/root/artifacts/lens_candidate'))"
[ -x /root/venv/bin/python ] || uv venv -q /root/venv --python 3.12
VIRTUAL_ENV=/root/venv uv pip install -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
VIRTUAL_ENV=/root/venv uv pip install -q transformers==5.20.2 accelerate==1.15.0 safetensors numpy scipy pyyaml
[ -d /root/jlens-src ] || git clone -q https://github.com/anthropics/jacobian-lens.git /root/jlens-src
git -C /root/jlens-src checkout -q 581d398613e5602a5af361e1c34d3a92ea82ba8e
VIRTUAL_ENV=/root/venv uv pip install -q --no-deps /root/jlens-src
[ -d /usr/local/cuda-13.0/compat ] && export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
/root/venv/bin/python -c "import torch, transformers, jlens; print('torch', torch.__version__, torch.cuda.get_device_name(0), 'transformers', transformers.__version__)"
echo setup done
