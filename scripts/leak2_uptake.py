#!/usr/bin/env python3
"""Lane D. leak2, uptake by direction within one turn: does the surplus pick the note that was used over the one that was
only seen? Writes results/lane_d/leak2_uptake.json. numpy only; reads the replay of runs/sweep2_leak2.

Each leak2 blind turn saw two notes, U (used, by leak2_source) and I (seen, ignored). A note's class is (right or wrong
value, posted first or second).

1. Two-way nearest signature. d is the full surplus, actual minus the kept reference (both notes masked). A class signature
   is the mean unit d of the other seeds' targets who used a note of that class (leave one seed out). The turn picks the
   note whose class signature is nearer d; scored against leak2_source, chance 0.5.
2. Exposure-orthogonal component. For each note X, d_X = actual minus the reference that masks only X. Remove from d_X its
   projection onto both notes' exposure directions and take the remaining norm over the actual norm. Exposure directions
   are (a) the same turn's d_U and d_I at the reply's first token, before any uptake, and (b) leave-one-seed-out class
   signatures of that first-token direction over the other seeds' notes, used or not. Used against ignored, paired.

Locations: mention_pre (the token before the reasoning first writes the submitted value), answer_pre, after_think, and the
trace peak (the reply token where the late-band per-token surplus is largest; there only the trace layers L18 and L34
exist, and only the full surplus, so only test 1).

  python scripts/leak2_uptake.py [replay_root] [runs_root]
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.labels import run_labels  # noqa: E402

REPLAY = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"data/replay_sweep2"
RUNS = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT/"runs"
EXP = "sweep2_leak2"
BANDS = {"mid": list(range(16, 20)), "late": list(range(31, 39)), "all": list(range(39))}
POSITIONS = ("mention_pre", "answer_pre", "after_think")


def unit(x):
    return x/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def cos(a, b):
    return float(np.mean(np.sum(unit(a)*unit(b), axis=-1)))


def p_one_sided(hits, n):
    return sum(comb(n, k) for k in range(hits, n + 1))/2**n if n else None


def orth_norm(d, basis):
    """Per layer, norm of d after removing its projection onto span(basis); d [L, D], basis list of [L, D]."""
    out = []
    for li in range(d.shape[0]):
        q, _ = np.linalg.qr(np.stack([b[li] for b in basis], axis=1))
        r = d[li] - q @ (q.T @ d[li])
        out.append(np.linalg.norm(r))
    return np.array(out)


def load():
    turns = []
    for seed_dir in sorted((REPLAY/EXP).glob("seed*")):
        d = seed_dir/"leak2"
        if not (d/"index.json").exists():
            continue
        run = json.loads((RUNS/EXP/seed_dir.name/"leak2"/"run.json").read_text())
        lab = run_labels(run)["measured"]["per_agent"]
        entries = run["plant"]["entries"]
        for t in json.loads((d/"index.json").read_text())["turns"]:
            mine = [e for e in entries if e.get("target") == t["agent"]]
            source = lab[t["agent"]].get("leak2_source")
            if t["phase"] != "solve" or len(mine) != 2 or source is None:
                continue
            first = min(mine, key=lambda e: e["seq"])["author"]
            notes = {e["author"]: (("right" if e["correct"] else "wrong"), ("first" if e["author"] == first else "second")) for e in mine}
            data = np.load(d/t["file"])
            names = list(t["positions"])
            rec = {"seed": seed_dir.name, "agent": t["agent"], "used": source, "ignored": next(a for a in notes if a != source),
                   "classes": notes, "address": run["plant"].get("address"), "d": {}, "dx": {}, "h": {}}
            for p in POSITIONS + ("first_generated",):
                if p in names:
                    n = names.index(p)
                    h = data["actual"][n]
                    rec["h"][p] = h
                    rec["d"][p] = h - data["reference"][n]
                    rec["dx"][p] = {a: h - data[f"mask_{a}"][n] for a in notes if f"mask_{a}" in data.files}
            if "trace_delta_reference" in data.files:
                late = data["trace_delta_reference"][:, BANDS["late"]].mean(axis=1)
                k = int(late.argmax())
                rec["trace_peak"], rec["trace_tokens"] = k, len(late)
                rec["d"]["trace_peak"] = data["trace_d_reference"][k].astype(np.float32)   # [2, D], layers t["trace_layers"]
            turns.append(rec)
    return turns


def nearest_signature(turns, location, layers):
    rows = []
    for i, t in enumerate(turns):
        if location not in t["d"]:
            continue
        d = t["d"][location][layers]
        sim = {}
        for note in (t["used"], t["ignored"]):
            cls = t["classes"][note]
            members = [unit(u["d"][location][layers]) for j, u in enumerate(turns)
                       if j != i and u["seed"] != t["seed"] and location in u["d"] and u["classes"][u["used"]] == cls]
            if members:
                sim[note] = cos(d, np.mean(members, axis=0))
        if len(sim) == 2:
            rows.append({"seed": t["seed"], "pick": max(sim, key=sim.get), "used": t["used"], "cos": sim})
    hits = sum(r["pick"] == r["used"] for r in rows)
    return {"hits": hits, "n": len(rows), "accuracy": hits/len(rows) if rows else None, "p_vs_half": p_one_sided(hits, len(rows)), "rows": rows}


def exposure_orthogonal(turns, location, layers, basis_kind):
    rows = []
    for i, t in enumerate(turns):
        if location not in t["dx"] or "first_generated" not in t["dx"] or len(t["dx"][location]) != 2:
            continue
        if basis_kind == "same turn":
            basis = [t["dx"]["first_generated"][a][layers] for a in (t["used"], t["ignored"])]
        else:
            basis = []
            for note in (t["used"], t["ignored"]):
                cls = t["classes"][note]
                members = [unit(u["dx"]["first_generated"][a][layers]) for u in turns if u["seed"] != t["seed"] and "first_generated" in u["dx"]
                           for a in u["dx"]["first_generated"] if u["classes"][a] == cls]
                if not members:
                    break
                basis.append(np.mean(members, axis=0))
            if len(basis) != 2:
                continue
        hn = np.linalg.norm(t["h"][location][layers], axis=-1)
        val = {a: float(np.mean(orth_norm(t["dx"][location][a][layers], basis)/hn)) for a in (t["used"], t["ignored"])}
        raw = {a: float(np.mean(np.linalg.norm(t["dx"][location][a][layers], axis=-1)/hn)) for a in (t["used"], t["ignored"])}
        rows.append({"seed": t["seed"], "used": val[t["used"]], "ignored": val[t["ignored"]], "raw_used": raw[t["used"]], "raw_ignored": raw[t["ignored"]]})
    wins = sum(r["used"] > r["ignored"] for r in rows)
    raw_wins = sum(r["raw_used"] > r["raw_ignored"] for r in rows)
    return {"used_larger": wins, "n": len(rows), "p_vs_half": p_one_sided(wins, len(rows)),
            "median_used": float(np.median([r["used"] for r in rows])) if rows else None,
            "median_ignored": float(np.median([r["ignored"] for r in rows])) if rows else None,
            "raw_used_larger": raw_wins, "rows": rows}


def main() -> int:
    turns = load()
    out = {"turns": len(turns), "used_classes": {t["seed"]: t["classes"][t["used"]] for t in turns}, "nearest_signature": {}, "exposure_orthogonal": {}}
    for location in POSITIONS:
        for band, layers in BANDS.items():
            out["nearest_signature"][f"{location} | {band}"] = nearest_signature(turns, location, layers)
            for kind in ("same turn", "class signatures"):
                out["exposure_orthogonal"][f"{location} | {band} | {kind}"] = exposure_orthogonal(turns, location, layers, kind)
    for name, layers in (("L18", [0]), ("L34", [1]), ("L18+L34", [0, 1])):
        out["nearest_signature"][f"trace_peak | {name}"] = nearest_signature(turns, "trace_peak", layers)
    out["trace_peaks"] = {t["seed"]: [t.get("trace_peak"), t.get("trace_tokens")] for t in turns}
    (ROOT/"results/lane_d").mkdir(parents=True, exist_ok=True)
    (ROOT/"results/lane_d/leak2_uptake.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("turns", len(turns))
    for k, v in out["nearest_signature"].items():
        print(f"nearest  {k:<28} {v['hits']}/{v['n']}  p={v['p_vs_half']:.3f}" if v["n"] else f"nearest  {k:<28} -")
    for k, v in out["exposure_orthogonal"].items():
        if v["n"]:
            print(f"orth     {k:<44} used>ignored {v['used_larger']}/{v['n']}  p={v['p_vs_half']:.3f}  median {v['median_used']:.3f} / {v['median_ignored']:.3f}  raw {v['raw_used_larger']}/{v['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
