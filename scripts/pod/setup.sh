#!/bin/bash
# Pod setup, run from /workspace/swarm. Image runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404
# (torch 2.8.0+cu128, Python 3.12). The venv lives on the volume so it survives a stop.
# Versions follow third_party/maj-lens/env/requirements.txt; kernels and playwright are left
# out (GPT-OSS and figure rendering only), torch stays the image's build.
set -euo pipefail
python3 -m venv --system-site-packages /workspace/venv
. /workspace/venv/bin/activate
pip install -q --upgrade pip
pip install -q numpy==2.1.0 scipy==1.18.0 scikit-learn==1.7.2 transformers==5.17.0 transformer-lens==3.9.0 \
  huggingface-hub==1.28.0 accelerate==1.15.0 datasets==4.8.5 pyarrow==25.0.1 safetensors==0.8.0 PyYAML==6.0.1 \
  matplotlib==3.11.1 tqdm==4.70.0 threadpoolctl jinja2
python - <<'P'
import torch, transformers, transformer_lens, numpy
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, 'numpy', numpy.__version__)
P
