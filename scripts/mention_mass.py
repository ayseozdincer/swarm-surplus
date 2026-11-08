#!/usr/bin/env python3
"""W6 sweep1. J-space surplus mass at mention_pre, per solve turn, in seeds 2000 to 2002. Writes
results/mention_mass_sweep1.json. CPU only, from the jspace.npz files of the replay.

In the leak branch only carol received Frank's answer. Her mass at the token before she first writes the answer is set
against the other readers of the same branch, against her own turn under the harmless notes, and against the plain-kernel
floor of the none branch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]
from surplus_table import mass, shares  # noqa: E402

SEEDS = (2000, 2001, 2002)
VARIANTS = ("leak", "benign_k1", "benign_k3", "proposal_k1", "proposal_k3", "none")
BANDS = {"L26-30": range(26, 31), "L50-62": range(50, 63), "all": range(63)}


def main() -> int:
    out = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            directory = ROOT/f"data/replay/sweep1/seed{seed}/{variant}"
            index = json.loads((directory/"index.json").read_text())
            rows = json.loads((directory/"rows.json").read_text())["rows"]
            z = np.load(directory/"jspace.npz")
            look = {(r["file"], r["pass"], r["position"]): i for i, r in enumerate(rows)}
            for turn in index["turns"]:
                if turn["phase"] != "solve" or "mention_pre" not in turn["positions"]:
                    continue
                i, j = look[(turn["file"], "actual", "mention_pre")], look[(turn["file"], "reference", "mention_pre")]
                z_a, z_r = shares(z["support"][i], z["coefficients"][i]), shares(z["support"][j], z["coefficients"][j])
                out.setdefault(str(seed), {}).setdefault(variant, {})[turn["agent"]] = {
                    "reference_kind": turn["reference_kind"], **{b: mass(z_a, z_r, set(l)) for b, l in BANDS.items()}}
    (ROOT/"results/mention_mass_sweep1.json").write_text(json.dumps(out, indent=1) + "\n")
    for seed, variants in out.items():
        for variant, agents in variants.items():
            print(seed, variant.ljust(12), {a: round(v["all"], 3) for a, v in agents.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
