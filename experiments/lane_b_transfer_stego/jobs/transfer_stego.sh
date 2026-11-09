# Lane B on its own pod. One model load replays the noise turns, Transfer, then Stego. A second job decomposes and scores
# each Transfer run as soon as its replay lands (one lens load), then scores Transfer, beside the Stego replay on the same card (model 61 GiB, lens about 7 GiB).
. /workspace/venv/bin/activate
cd /workspace/swarm/third_party/maj-lens
export HF_HOME=/workspace/hf STUDY_MODELS=qwen3_32b PYTHONPATH=.:/workspace/swarm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u ../../experiments/lane_b_transfer_stego/run.py check || exit 1
/workspace/swarm/scripts/pod/job.sh lane_b_transfer_score python -u ../../experiments/lane_b_transfer_stego/run.py follow report_transfer --dirs ../../data/replay_lane_b/transfer --expect 72
python -u ../../experiments/lane_b_transfer_stego/run.py noise transfer stego
