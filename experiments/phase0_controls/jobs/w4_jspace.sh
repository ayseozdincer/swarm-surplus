# W4 part one: the J-space repo's pipeline from public inputs, into /workspace/artifacts beside the pinned inputs.
set -e
. /workspace/venv/bin/activate
cd /workspace/swarm/third_party/maj-lens
export HF_HOME=/workspace/hf STUDY_MODELS=qwen3_32b PYTHONPATH=.:/workspace/swarm
python -u -m utils.build.setup replay --model qwen3_32b --placement auto
python -u -m utils.build.setup prepare --model qwen3_32b
python -u -m utils.build.setup stage --model qwen3_32b
python -u -m utils.build.setup primary --model qwen3_32b
NARC_PHASE=discussion python -u -m utils.build.setup replay --model qwen3_32b --placement auto --output artifacts/rounds/qwen3_32b
python -u /workspace/swarm/experiments/phase0_controls/sweep_fast.py qwen3_32b
