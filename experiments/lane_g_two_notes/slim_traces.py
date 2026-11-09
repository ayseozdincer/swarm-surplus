#!/usr/bin/env python3
"""Keep only `trace_delta_reference` of each replayed turn, the array paper/build/score_newdata.py reads. A full turn's .npz is
about 170 MB, its slim copy under 1 MB, so the data bundle can carry the out-of-sample test.

  python experiments/lane_g_two_notes/slim_traces.py <replay>/sweep2_leak2 data/replay_lane_g/sweep2_leak2
"""
import shutil
import sys
from pathlib import Path

import numpy as np

src, out = Path(sys.argv[1]), Path(sys.argv[2])
for index in sorted(src.glob("seed*/*/index.json")):
    target = out/index.relative_to(src)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(index, target)
    for npz in index.parent.glob("*.npz"):
        np.savez_compressed(target.parent/npz.name, trace_delta_reference=np.load(npz)["trace_delta_reference"])
print("slim traces under", out)
