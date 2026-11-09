# W4 part two: our reference pass on Core and the noise floor.
set -e
. /workspace/venv/bin/activate
cd /workspace/swarm/third_party/maj-lens
export HF_HOME=/workspace/hf STUDY_MODELS=qwen3_32b PYTHONPATH=.:/workspace/swarm
python -u ../../experiments/phase0_controls/phase0.py noise --out ../../data/phase0/noise --n 20
python -u ../../experiments/phase0_controls/phase0.py replay --out ../../data/phase0/core
python -u ../../scripts/decompose.py ../../data/phase0/noise/repeat ../../data/phase0/noise/kernel ../../data/phase0/core/*
