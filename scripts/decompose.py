#!/usr/bin/env python3
"""Sparse J-space decomposition of replayed residuals, one lens layer at a time across every run directory given.

  python scripts/decompose.py data/replay/pilot/seed1000/leak data/replay/pilot/seed1000/none ...

Reads each directory's index.json and its turn .npz files. Writes <dir>/jspace.npz with support [rows, layers, 25] and coefficients, and <dir>/rows.json
naming each row by file, pass and position. A `compacted` (replay.py --compacted) or `placebo` (lane_c_blind.py) pass is
decomposed as a further pass.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]
from detector import jspace  # noqa: E402
from replay import LENS_FILE, MODEL_DIR  # noqa: E402

BATCH = 384


def load_rows(directory: Path, keep=None):
    index = json.loads((directory/"index.json").read_text())
    rows, residuals = [], []
    for turn in index["turns"]:
        data = np.load(directory/turn["file"])
        names = list(turn["positions"])
        # With nothing masked the compacted pass is the reference, so it is not decomposed twice. A placebo pass
        # (lane_c_blind.py) is decomposed like any other.
        labels = [x for x in ("actual", "reference", "compacted", "placebo") if x in data.files]
        for label in [x for x in labels if x != "compacted" or turn["masked_tokens"]]:
            for n, name in enumerate(names):
                if keep and name not in keep:
                    continue
                rows.append({"file": turn["file"], "pass": label, "position": name})
                residuals.append(data[label][n])
    return rows, np.stack(residuals)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--positions", nargs="*", help="decompose only these named positions (default every one)")
    args = parser.parse_args()
    lens = jspace.Lens(MODEL_DIR, LENS_FILE)
    work = []
    for directory in args.dirs:
        if (directory/"jspace.npz").exists():
            continue
        rows, residuals = load_rows(directory, set(args.positions or ()))
        work.append((directory, rows, residuals, np.full((len(rows), len(lens.layers), jspace.K), -1, dtype=np.int32),
                     np.zeros((len(rows), len(lens.layers), jspace.K), dtype=np.float32)))
        print(directory, len(rows), "rows", flush=True)
    for li, layer in enumerate(lens.layers):
        started = time.time()
        atoms, norms = lens.atoms(layer)
        for _, _, residuals, support, coefficients in work:
            for start in range(0, len(residuals), BATCH):
                X = torch.from_numpy(residuals[start:start + BATCH, li]).cuda()
                s, c = jspace.pursuit(X, atoms, norms)
                support[start:start + len(X), li] = s.cpu().numpy()
                coefficients[start:start + len(X), li] = c.cpu().numpy()
        del atoms, norms
        print(f"layer {layer} {time.time() - started:.1f}s", flush=True)
    for directory, rows, _, support, coefficients in work:
        np.savez_compressed(directory/"jspace.npz", support=support, coefficients=coefficients)
        (directory/"rows.json").write_text(json.dumps({"layers": lens.layers, "rows": rows}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
