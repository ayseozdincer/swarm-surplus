#!/bin/bash
# Pack runs/ and data/ into the bundle scripts/fetch_data.sh unpacks, and record its checksum.
# Lane G's slim traces come from experiments/lane_g_two_notes/slim_traces.py.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:-swarm-surplus-data-v1.tar.gz}
RUNS="runs/pilot runs/sweep1 runs/sweep14b runs/sweep2 runs/sweep2_norule runs/sweep2_leak2 runs/sweep2_multi runs/sweep2_solvable
      runs/sweep2_forbidden2 runs/sweep2_methodtip runs/planted runs/shared runs/shared_c2 runs/lane_g"
DATA="data/replay_lane_f data/replay_lane_g data/replay_tasktracker data/replay_insider data/external_tt"
COPYFILE_DISABLE=1 tar --exclude='.DS_Store' --exclude='__pycache__' -czf "$NAME" $RUNS $DATA
echo "$(shasum -a 256 "$NAME" | cut -d' ' -f1)  $(basename "$NAME")" > scripts/data_bundle.sha256
ls -lh "$NAME"; cat scripts/data_bundle.sha256
