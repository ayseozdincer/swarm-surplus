#!/bin/bash
# Lane D pod setup, run from /workspace/swarm: a separate venv for vLLM (it pins its own torch), the replay venv from
# setup.sh, and the Qwen3-14B weights, both on the container disk (/workspace is a network filesystem, slow to load from).
set -euo pipefail
export HF_HOME=/root/hf
pip install -q uv
[ -x /root/vllm-venv/bin/python ] || uv venv -q /root/vllm-venv --python 3.12
VIRTUAL_ENV=/root/vllm-venv uv pip install -q vllm openai
/root/vllm-venv/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda)"
/root/vllm-venv/bin/hf download Qwen/Qwen3-14B > /workspace/logs/download.log 2>&1
echo setup done
