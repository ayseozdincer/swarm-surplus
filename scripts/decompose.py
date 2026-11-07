#!/usr/bin/env python3
"""Sparse J-space decomposition of replayed residuals, one lens layer at a time across every run directory given.

  python scripts/decompose.py data/replay/pilot/seed1000/leak data/replay/pilot/seed1000/none ...

Reads each directory's index.json and its turn .npz files. A turn with no peer span has reference == actual, so only its
actual rows are decomposed. Writes <dir>/jspace.npz with support [rows, layers, 25] and coefficients, and <dir>/rows.json
naming each row by file, pass and position.
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


def load_rows(directory: Path):
    index = json.loads((directory/"index.json").read_text())
    rows, residuals = [], []
    for turn in index["turns"]:
        data = np.load(directory/turn["file"])
        names = list(turn["positions"])
        for label in ("actual", "reference") if turn["peer_token_spans"] else ("actual",):
            for n, name in enumerate(names):
                rows.append({"file": turn["file"], "pass": label, "position": name})
                residuals.append(data[label][n])
    return rows, np.stack(residuals)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", type=Path)
    args = parser.parse_args()
    lens = jspace.Lens(MODEL_DIR, LENS_FILE)
    work = []
    for directory in args.dirs:
        if (directory/"jspace.npz").exists():
            continue
        rows, residuals = load_rows(directory)
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
