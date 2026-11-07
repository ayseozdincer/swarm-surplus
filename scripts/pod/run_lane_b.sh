# Lane B on its own pod. One model load replays the noise turns, Transfer, then Stego. A second job decomposes and scores
# Transfer as soon as its replay is done, beside the Stego replay on the same card (model 61 GiB, lens about 7 GiB).
. /workspace/venv/bin/activate
cd /workspace/swarm/third_party/maj-lens
export HF_HOME=/workspace/hf STUDY_MODELS=qwen3_32b PYTHONPATH=.:/workspace/swarm PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u ../../scripts/lane_b.py check || exit 1
/workspace/swarm/scripts/pod/job.sh lane_b_transfer_score bash -c '
  until grep -q "^TRANSFER DONE" /workspace/logs/lane_b_replay.log; do grep -q "^exit" /workspace/logs/lane_b_replay.log && exit 1; sleep 20; done
  D=../../data/replay_lane_b
  python -u ../../scripts/lane_b.py decompose report_transfer --dirs $D/noise $D/transfer/*'
python -u ../../scripts/lane_b.py noise transfer stego
