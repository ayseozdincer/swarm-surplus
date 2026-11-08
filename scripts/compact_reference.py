#!/usr/bin/env python3
"""W3 follow-up, one turn and one seed: does the position gap explain the distance between the reference and `none`?

The current reference masks the peer span and keeps every position id. The compacted reference is the same tokens, the same
explicit 4D mask and the same kernel, with tokens after the span given the position ids they would have had if the span
were not there, so the prompt lines up with the none branch token for token.

For carol's episode 2 solve turn, per layer and position, the relative residual distance from the none branch for the actual
pass, the kept reference and the compacted reference, beside the kernel floor at that position (the none turn, plain kernel
against explicit mask).

Pass criterion, fixed by the request. The compacted reference is within the kernel floor of none at every layer.
`same_history` says whether the kept tokens up to that position equal the none branch's. Where they do not, her reply has
already diverged and the two states are not the same computation.

Then surplus mass with the compacted reference at each position, beside the kept reference's, on this turn.
Writes results/compact_reference.json.
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]
from detector import jspace  # noqa: E402
from detector.render import compacted_positions, find_turn, load_turns, render_turn  # noqa: E402
from replay import LENS_FILE, LENS_LAYERS, MODEL_DIR, Capture, attention_mask_4d, load_model  # noqa: E402
from surplus_table import mass, shares  # noqa: E402

RUN = ROOT/"runs/pilot/seed1000"
REPLAY = ROOT/"data/replay/pilot/seed1000"
BRANCHES = ("leak", "benign_k1", "proposal_k1")
TABLE = ("prompt_last", "first_generated", "first_thought", "after_think")
FILE = "carol_e2_solve.npz"


def saved(branch):
    index = json.loads((REPLAY/branch/"index.json").read_text())
    return next(t for t in index["turns"] if t["file"] == FILE), np.load(REPLAY/branch/FILE)


def relative(a, b):
    return np.linalg.norm(a - b, axis=-1)/np.linalg.norm(b, axis=-1)


def main() -> int:
    model, tok = load_model()
    capture = Capture(model, LENS_LAYERS)
    renders, compact = {}, {}
    for branch in BRANCHES:
        r = render_turn(tok, find_turn(load_turns(RUN/branch), "carol", 2, "solve"))
        turn, _ = saved(branch)
        positions = list(turn["positions"].values())
        if any(r["positions"][name] != p for name, p in turn["positions"].items()):
            raise ValueError(f"{branch}: positions differ from the saved replay")
        compact[branch] = capture.run(model, r["ids"], positions, attention_mask_4d(len(r["ids"]), r["mask"]),
                                      position_ids=compacted_positions(r["mask"])).numpy()
        renders[branch] = r
        print(branch, len(r["ids"]), flush=True)
    plain = render_turn(tok, find_turn(load_turns(RUN/"none"), "carol", 2, "solve"))
    del model, capture
    gc.collect()
    torch.cuda.empty_cache()

    none_turn, none_data = saved("none")
    none_names = list(none_turn["positions"])
    out = {"criterion": "compacted reference within the kernel floor of none at every layer", "branches": {}}
    for branch in BRANCHES:
        turn, data = saved(branch)
        names = list(turn["positions"])
        r = renders[branch]
        rows = {}
        for position in TABLE:
            n, m = names.index(position), none_names.index(position)
            base = none_data["actual"][m]
            floor = relative(none_data["reference"][m], base)
            actual, kept, compacted = relative(data["actual"][n], base), relative(data["reference"][n], base), relative(compact[branch][n], base)
            upto = r["positions"][position]
            history = [t for t, k in zip(r["ids"][:upto + 1], r["mask"][:upto + 1]) if k]
            rows[position] = {"same_history": history == plain["ids"][:plain["positions"][position] + 1],
                              "layers_within_floor": int((compacted <= floor).sum()), "layers": len(floor),
                              "passes": bool((compacted <= floor).all()),
                              "actual": actual.tolist(), "kept": kept.tolist(), "compacted": compacted.tolist(), "floor": floor.tolist()}
        out["branches"][branch] = rows

    lens = jspace.Lens(MODEL_DIR, LENS_FILE)
    decomposed = {}
    stacked = np.concatenate([compact[b] for b in BRANCHES])
    support = np.zeros((len(stacked), len(LENS_LAYERS), jspace.K), dtype=np.int32)
    coefficients = np.zeros((len(stacked), len(LENS_LAYERS), jspace.K), dtype=np.float32)
    for li, layer in enumerate(LENS_LAYERS):
        atoms, norms = lens.atoms(layer)
        s, c = jspace.pursuit(torch.from_numpy(stacked[:, li]).cuda(), atoms, norms)
        support[:, li], coefficients[:, li] = s.cpu().numpy(), c.cpu().numpy()
        del atoms, norms
    start = 0
    surplus = {}
    for branch in BRANCHES:
        turn, _ = saved(branch)
        names = list(turn["positions"])
        meta = json.loads((REPLAY/branch/"rows.json").read_text())
        z = np.load(REPLAY/branch/"jspace.npz")
        look = {(x["file"], x["pass"], x["position"]): i for i, x in enumerate(meta["rows"])}
        surplus[branch] = {}
        for n, position in enumerate(names):
            i, j = look[(FILE, "actual", position)], look[(FILE, "reference", position)]
            z_a, z_k = shares(z["support"][i], z["coefficients"][i]), shares(z["support"][j], z["coefficients"][j])
            z_c = shares(support[start + n], coefficients[start + n])
            surplus[branch][position] = {"kept": mass(z_a, z_k), "compacted": mass(z_a, z_c)}
        start += len(names)
    out["surplus_mass"] = surplus
    (ROOT/"results/compact_reference.json").write_text(json.dumps(out, indent=1) + "\n")
    for branch, rows in out["branches"].items():
        print(branch, {p: (v["same_history"], v["layers_within_floor"], v["passes"]) for p, v in rows.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
