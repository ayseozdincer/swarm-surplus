# Lane C inputs: the pinned jacobian-lens checkout (for utils.release.load_released_lens), Qwen3-14B and its released lens,
# at the lens revision config.py pins. Idempotent. Run after setup.sh.
set -e
mkdir -p /workspace/artifacts /workspace/hf
cd /workspace/swarm/third_party/maj-lens
ln -sfn /workspace/artifacts artifacts
. /workspace/venv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=0
[ -d jacobian-lens/.git ] || git clone -q https://github.com/anthropics/jacobian-lens.git jacobian-lens
git -C jacobian-lens checkout -q 581d398613e5602a5af361e1c34d3a92ea82ba8e
pip install -q --no-deps -e jacobian-lens
python - <<'P'
from huggingface_hub import hf_hub_download, snapshot_download
REV = "8aa8152c99710a9d7a2f8889319502d46443ff10"
for name in ("Qwen3-14B_jacobian_lens.pt", "config.yaml", "Qwen3-14B_convergence.csv"):
    hf_hub_download("neuronpedia/jacobian-lens", f"qwen3-14b/jlens/Salesforce-wikitext/{name}", revision=REV,
                    local_dir="/workspace/artifacts/lens_candidate")
snapshot_download("Qwen/Qwen3-14B", local_dir="/workspace/artifacts/qwen3_14b",
                  allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.jinja"])
print("inputs ready for qwen3_14b", flush=True)
P
