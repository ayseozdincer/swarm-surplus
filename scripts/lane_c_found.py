#!/usr/bin/env python3
"""Lane C. The position-blind pipeline end to end: the trace says where, the lens or the direction says what. Writes
results/lane_c/found.json.

  SWARM_MODEL=qwen3_14b python scripts/lane_c_found.py --replay data/replay_14b/blind/sweep14b --runs runs/sweep14b

Where. For the blind agent's solve turn in leak, benign_k1, proposal_k1 and none of every seed, the maximum over the reply of a
sliding mean of delta over floor (band L16 to L38) for windows of 16 and 32 tokens, as scripts/lane_c_window.py. The found
location is that window's centre token and its last token. No knowledge of the mention is used.
What. One more actual and kept-reference pass of each turn (explicit 4D mask, plain kernel for none, as stage 1) reads the
residual at the found tokens. At each:
  naming rank   the negated best late-layer lens rank of the planted answer's first token (the leak branch's first-mention
                token, the same for every branch of a seed) in the actual pass
  naming lift   the most that rank rises from reference to actual over the late layers, in log rank
  share         the J-space surplus share of that token, its largest [z - z~]+ over the late layers
  source        cos(d, leak signature) - the largest cos with the benign_k1 and proposal_k1 signatures, late band, where a note's
                signature is the mean unit surplus direction of the other nine seeds' blind turns in that branch at their own found
                location; `nearest` is the note whose signature is closest
Leak against benign_k1 and against none, AUROC with matched-seed paired wins.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]
from detector.render import find_turn, load_turns, render_turn  # noqa: E402
from replay import LENS_FILE, LENS_LAYERS, MODEL_DIR, Capture, attention_mask_4d, band, load_model, on_model  # noqa: E402

BRANCHES = ("leak", "benign_k1", "proposal_k1", "none")
NOTES = ("leak", "benign_k1", "proposal_k1")
WINDOWS = (16, 32)
TRACE_BAND = list(range(16, 39))
LATE = sorted({on_model(l) for l in (52, 54, 56, 58, 60, 61, 62)})
LATE_BAND = band(50, 62)


def unit(x):
    return x/np.linalg.norm(x, axis=-1, keepdims=True)


def found(series, w):
    """(centre, last) offsets of the sliding window of w tokens with the largest mean."""
    w = min(w, len(series))
    means = np.convolve(series, np.ones(w)/w, mode="valid")
    i = int(means.argmax())
    return i + w//2, i + w - 1


def auroc(pos, neg):
    pos, neg = [x for x in pos if x is not None], [x for x in neg if x is not None]
    return sum((p > n) + 0.5*(p == n) for p in pos for n in neg)/(len(pos)*len(neg)) if pos and neg else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", type=Path, default=ROOT/"data/replay_14b/blind/sweep14b")
    parser.add_argument("--runs", type=Path, default=ROOT/"runs/sweep14b")
    parser.add_argument("--out", type=Path, default=ROOT/"results/lane_c/found.json")
    args = parser.parse_args()
    import torch
    from detector import jspace
    from surplus_table import shares

    seeds = sorted(p.name for p in args.replay.iterdir() if p.name.startswith("seed"))
    traces, turns = {}, {}
    for seed in seeds:
        for b in BRANCHES:
            turn = json.loads((args.replay/seed/b/"index.json").read_text())["turns"][0]
            data = np.load(args.replay/seed/b/turn["trace_file"])
            turns[seed, b] = turn
            traces[seed, b] = {"positions": data["positions"].tolist(), "delta": data["delta_reference"]}
    floor = np.median(np.concatenate([traces[s, "none"]["delta"] for s in seeds]), axis=0)

    locations = {}
    for (seed, b), t in traces.items():
        start = t["positions"].index(turns[seed, b]["positions"]["first_generated"])
        series = (t["delta"][start:]/floor[None, :])[:, TRACE_BAND].mean(1)
        mention = turns[seed, b]["positions"].get("mention_pre")
        for w in WINDOWS:
            centre, last = found(series, w)
            locations[seed, b, w] = {"centre": t["positions"][start + centre], "last": t["positions"][start + last],
                                     "mention_pre": mention}

    model, tok = load_model()
    lens = jspace.Lens(MODEL_DIR, LENS_FILE)
    lens.J = {layer: J.to(lens.device) for layer, J in lens.J.items()}
    capture = Capture(model, lens.layers)
    states = {}
    for seed in seeds:
        for b in BRANCHES:
            turn = turns[seed, b]
            r = render_turn(tok, find_turn(load_turns(args.runs/seed/b), turn["agent"], turn["episode_index"], "solve"))
            wanted = sorted({locations[seed, b, w][k] for w in WINDOWS for k in ("centre", "last")})
            n = len(r["ids"])
            masked = bool(r["peer_token_spans"])
            actual = capture.run(model, r["ids"], wanted, attention_mask_4d(n)).numpy()
            reference = capture.run(model, r["ids"], wanted, attention_mask_4d(n, r["mask"]) if masked else None).numpy()
            states[seed, b] = {"positions": wanted, "actual": actual, "reference": reference,
                               "tokens": {p: tok.decode([r["ids"][p]]) for p in wanted}}
            print(seed, b, wanted, flush=True)
    del model
    torch.cuda.empty_cache()

    # J-space of every captured row, one late layer at a time
    rows = [(key, label, i) for key, s in states.items() for label in ("actual", "reference") for i in range(len(s["positions"]))]
    z = {}
    for layer in LATE:
        li = list(lens.layers).index(layer)
        X = torch.from_numpy(np.stack([states[key][label][i, li] for key, label, i in rows])).cuda()
        atoms, norms = lens.atoms(layer)
        support, coefficients = jspace.pursuit(X, atoms, norms)
        for (key, label, i), sh in zip(rows, shares(support.cpu().numpy(), coefficients.cpu().numpy())):
            z[key, label, i, layer] = sh
        del atoms, norms

    def target(seed):
        t = turns[seed, "leak"]
        return t["position_token_ids"]["mention_0"]

    def at(seed, b, w, kind):
        s = states[seed, b]
        i = s["positions"].index(locations[seed, b, w][kind])
        a, r = s["actual"][i], s["reference"][i]
        t = target(seed)
        late = [list(lens.layers).index(l) for l in LATE]
        ra = np.array([int((lg > lg[t]).sum()) for lg in (lens.logits(torch.from_numpy(a[l])[None], LATE[k])[0] for k, l in enumerate(late))])
        rr = np.array([int((lg > lg[t]).sum()) for lg in (lens.logits(torch.from_numpy(r[l])[None], LATE[k])[0] for k, l in enumerate(late))])
        share = max(max(z[(seed, b), "actual", i, l].get(t, 0.0) - z[(seed, b), "reference", i, l].get(t, 0.0), 0.0) for l in LATE)
        return {"token": s["tokens"][locations[seed, b, w][kind]], "position": locations[seed, b, w][kind],
                "naming rank": -float(ra.min()), "naming lift": float(np.max(np.log1p(rr) - np.log1p(ra))), "share": float(share),
                "d": unit((a - r)[[list(lens.layers).index(l) for l in LATE_BAND]])}

    cells = {(seed, b, w, kind): at(seed, b, w, kind) for seed in seeds for b in BRANCHES for w in WINDOWS for kind in ("centre", "last")}
    for (seed, b, w, kind), c in cells.items():
        signatures = {n: np.mean([cells[o, n, w, kind]["d"] for o in seeds if o != seed], axis=0) for n in NOTES}
        sims = {n: float(np.mean(np.sum(c["d"]*unit(x), axis=-1))) for n, x in signatures.items()}
        c["source"] = sims["leak"] - max(sims[n] for n in NOTES if n != "leak")
        c["nearest"] = max(sims, key=sims.get)
        loc = locations[seed, b, w]
        c["offset from mention_pre"] = None if loc["mention_pre"] is None else loc[kind] - loc["mention_pre"]

    result = {"floor_band": [TRACE_BAND[0], TRACE_BAND[-1]], "late": LATE, "late_band": LATE_BAND, "claims": {}, "cells": {}}
    for (seed, b, w, kind), c in cells.items():
        result["cells"][f"{seed} {b} w{w} {kind}"] = {k: v for k, v in c.items() if k != "d"}
    for negative in ("benign_k1", "none", "proposal_k1"):
        for w in WINDOWS:
            for kind in ("centre", "last"):
                out = {}
                for score in ("naming rank", "naming lift", "share", "source"):
                    pos = [cells[s, "leak", w, kind][score] for s in seeds]
                    neg = [cells[s, negative, w, kind][score] for s in seeds]
                    out[score] = {"auroc": auroc(pos, neg), "paired_wins": sum(p > n for p, n in zip(pos, neg)), "pairs": len(seeds),
                                  "leak_median": float(np.median(pos)), "negative_median": float(np.median(neg))}
                out["nearest is leak"] = {"leak": sum(cells[s, "leak", w, kind]["nearest"] == "leak" for s in seeds),
                                          negative: sum(cells[s, negative, w, kind]["nearest"] == "leak" for s in seeds), "of": len(seeds)}
                result["claims"][f"leak vs {negative} | w{w} | {kind}"] = out
    for w in WINDOWS:
        for kind in ("centre", "last"):
            hits = [cells[s, b, w, kind]["nearest"] == b for s in seeds for b in NOTES]
            result["claims"][f"nearest-signature accuracy | w{w} | {kind}"] = {"accuracy": float(np.mean(hits)), "n": len(hits), "chance": 1/3}
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    for k, v in result["claims"].items():
        if k.startswith("leak vs benign_k1"):
            print(k, {s: f"{x['auroc']:.2f} ({x['paired_wins']}/{x['pairs']})" for s, x in v.items() if "auroc" in x}, v["nearest is leak"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
