# W4 follow-ups: placebo references, ontology on the surplus, the edge test. Uses the Core replay on the pod.
. /workspace/venv/bin/activate
cd /workspace/swarm/third_party/maj-lens
export HF_HOME=/workspace/hf STUDY_MODELS=qwen3_32b PYTHONPATH=.:/workspace/swarm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u ../../experiments/phase0_controls/phase0_followups.py collect || exit 1
python -u ../../experiments/phase0_controls/phase0_followups.py decompose || exit 1
python -u ../../experiments/phase0_controls/phase0_followups.py report
