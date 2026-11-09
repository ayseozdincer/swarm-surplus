#!/usr/bin/env python3
"""W3 follow-up, research question 3 on our own data, one turn and one seed. Writes experiments/phase0_controls/results/mention_readout.json.

For carol's solve turn in the leak, none and benign_k1 branches, at `first_generated` and at `mention_pre` (the token before
the reasoning first writes the submitted answer), per layer:

  relative residual delta   ||actual - reference|| / ||reference||
  floor                     the same quantity at the same named position on the none turn, whose second pass is the plain
                            kernel, so it holds kernel noise only
  ranks                     rank over the full vocabulary of the planted digits in each pass, and the best rank of the other digits
  J-space                   share of the planted digits in each pass and their surplus

`mention_pre` is not the same text across branches. She never writes the planted answer in none or benign_k1, so there it is
the token before her own answer. `first_generated` is the same token after the same prompt, apart from the note.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]
from detector import jspace  # noqa: E402
from replay import LENS_FILE, MODEL_DIR  # noqa: E402
from surplus_table import mass, shares  # noqa: E402

REPLAY = ROOT/"data/replay_mention/pilot/seed1000"
BRANCHES = ("leak", "benign_k1", "none")
POSITIONS = ("first_generated", "mention_pre")
PLANTED = ("7", "3")
OTHERS = tuple(d for d in "0123456789" if d not in PLANTED)


def load(branch):
    index = json.loads((REPLAY/branch/"index.json").read_text())
    turn = index["turns"][0]
    meta = json.loads((REPLAY/branch/"rows.json").read_text())
    z = np.load(REPLAY/branch/"jspace.npz")
    look = {(r["file"], r["pass"], r["position"]): i for i, r in enumerate(meta["rows"])}
    return index, turn, np.load(REPLAY/branch/turn["file"]), z, look


def relative(a, b):
    return np.linalg.norm(a - b, axis=-1)/np.linalg.norm(b, axis=-1)


def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    lens = jspace.Lens(MODEL_DIR, LENS_FILE)
    ids = {d: tok.encode(d, add_special_tokens=False)[0] for d in "0123456789"}
    loaded = {b: load(b) for b in BRANCHES}
    out = {"note": "mention_pre is the token before her own answer in none and benign_k1, not the same text", "positions": {}}
    for position in POSITIONS:
        floor = None
        index, turn, data, _, _ = loaded["none"]
        if position in turn["positions"]:
            n = list(turn["positions"]).index(position)
            floor = relative(data["actual"][n], data["reference"][n])
        out["positions"][position] = {}
        for branch in BRANCHES:
            index, turn, data, z, look = loaded[branch]
            if position not in turn["positions"]:
                out["positions"][position][branch] = None
                continue
            n = list(turn["positions"]).index(position)
            delta = relative(data["actual"][n], data["reference"][n])
            i, j = look[(turn["file"], "actual", position)], look[(turn["file"], "reference", position)]
            z_a, z_r = shares(z["support"][i], z["coefficients"][i]), shares(z["support"][j], z["coefficients"][j])
            rows = []
            for li, layer in enumerate(index["layers"]):
                a = lens.logits(torch.from_numpy(data["actual"][n, li])[None], layer)[0]
                r = lens.logits(torch.from_numpy(data["reference"][n, li])[None], layer)[0]
                rank = lambda x, d: int((x > x[ids[d]]).sum())
                rows.append({"layer": layer, "delta": float(delta[li]), "floor": None if floor is None else float(floor[li]),
                             "ratio": None if floor is None else float(delta[li]/floor[li]),
                             **{f"rank_{d}": [rank(a, d), rank(r, d)] for d in PLANTED},
                             "best_other": [min(rank(a, d) for d in OTHERS), min(rank(r, d) for d in OTHERS)],
                             "share": {d: [z_a[li].get(ids[d], 0.0), z_r[li].get(ids[d], 0.0)] for d in PLANTED}})
            out["positions"][position][branch] = {
                "token": turn["position_tokens"][position], "next_token": turn["position_tokens"].get("mention_0") if position == "mention_pre" else None,
                "kind": turn["reference_kind"], "jspace_mass": mass(z_a, z_r), "by_layer": rows,
                "layers_planted_in_jspace": sorted({l for row, l in zip(rows, index["layers"]) for d in PLANTED if max(row["share"][d]) > 0}),
                "layers_planted_surplus": sorted({l for row, l in zip(rows, index["layers"]) for d in PLANTED if row["share"][d][0] > row["share"][d][1]})}
    (ROOT/"experiments/phase0_controls/results/mention_readout.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("wrote experiments/phase0_controls/results/mention_readout.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
