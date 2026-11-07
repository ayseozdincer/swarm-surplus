# W4 follow-up 4, after the sweep1 replay: honest references with a size-matched removal, then the comparison.
. /workspace/venv/bin/activate
cd /workspace/swarm/third_party/maj-lens
export HF_HOME=/workspace/hf STUDY_MODELS=qwen3_32b PYTHONPATH=.:/workspace/swarm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u ../../scripts/phase0_followups.py collect_matched || exit 1
python -u ../../scripts/phase0_followups.py decompose --kinds removed swap_removed || exit 1
python -u ../../scripts/phase0_followups.py report_matched
