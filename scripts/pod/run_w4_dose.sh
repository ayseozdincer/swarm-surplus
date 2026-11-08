# W4 follow-ups 5 and 6 after W7: key-sentence dose references, their decomposition, token edits, then the report.
. /workspace/venv/bin/activate
cd /workspace/swarm/third_party/maj-lens
export HF_HOME=/workspace/hf STUDY_MODELS=qwen3_32b PYTHONPATH=.:/workspace/swarm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u ../../scripts/phase0_dose.py collect || exit 1
python -u ../../scripts/phase0_followups.py decompose --kinds dose_key dose_neutral dose_full dose_honest_neutral || exit 1
python -u ../../scripts/phase0_dose.py edits || exit 1
python -u ../../scripts/phase0_dose.py report
