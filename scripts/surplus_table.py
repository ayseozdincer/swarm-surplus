#!/usr/bin/env python3
"""J-space surplus mass per position across every turn of a replay, against the kernel-noise floor. Writes a JSON table.

Mass is the sum over concepts of the mean over layers of [z - z~]+, half the L1 distance between the two share vectors
averaged over layers, so it runs from 0 to 1. A branch with no peer message has reference_kind plain_kernel: its mass is what
kernel noise alone produces, measured by the same decomposition.

  python scripts/surplus_table.py experiments/phase0_controls/results/surplus_by_position.json data/replay/pilot/seed1000 [more seed directories]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]
from replay import band as depth_band  # noqa: E402


def shares(support, coefficients):
    coef = coefficients.astype(np.float64)
    total = coef.sum(axis=1, keepdims=True)
    frac = np.divide(coef, total, out=np.zeros_like(coef), where=total > 0)
    return [{int(t): float(v) for t, v in zip(s, f) if t >= 0} for s, f in zip(support, frac)]


def mass(z_a, z_r, keep=None):
    cells = [(a, r) for i, (a, r) in enumerate(zip(z_a, z_r)) if keep is None or i in keep]
    return float(np.mean([sum(max(v - r.get(c, 0.0), 0.0) for c, v in a.items()) for a, r in cells]))


def main() -> int:
    out, roots = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    table = defaultdict(lambda: defaultdict(list))
    for directory in sorted(p for root in roots for p in root.iterdir() if (p/"jspace.npz").exists()):
        meta = json.loads((directory/"rows.json").read_text())
        data = np.load(directory/"jspace.npz")
        support, coefficients = data["support"], data["coefficients"]
        index = {(r["file"], r["pass"], r["position"]): i for i, r in enumerate(meta["rows"])}
        band = {meta["layers"].index(l) for l in depth_band(26, 30)}
        for (file, label, position), i in index.items():
            if label != "actual":
                continue
            j = index[(file, "reference", position)]
            z_a, z_r = shares(support[i], coefficients[i]), shares(support[j], coefficients[j])
            key = position if not position.startswith(("answer_", "verdict_")) or position.endswith("_pre") else position.split("_")[0] + "_tokens"
            table[directory.name][key].append((mass(z_a, z_r), mass(z_a, z_r, band)))
    result = {b: {p: {"turns": len(v), "median_all_layers": float(np.median([x[0] for x in v])), "max_all_layers": float(np.max([x[0] for x in v])),
                      "median_L26_30": float(np.median([x[1] for x in v]))} for p, v in rows.items()} for b, rows in table.items()}
    out.write_text(json.dumps(result, indent=1) + "\n")
    positions = ["prompt_last", "first_generated", "first_thought", "after_think", "answer_pre", "verdict_pre", "end_of_response"]
    print("median mass, all layers / L26-30")
    print("branch".ljust(14) + "".join(p.ljust(18) for p in positions))
    for b, rows in result.items():
        print(b.ljust(14) + "".join((f"{rows[p]['median_all_layers']:.3f} / {rows[p]['median_L26_30']:.3f} ({rows[p]['turns']})" if p in rows else "-").ljust(18) for p in positions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
