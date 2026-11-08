#!/usr/bin/env python3
"""Lane D. Scores for the leak2, leak_all and solvable arms from their replay; writes results/lane_d/arms_scores.json.

leak2     two notes give the blind agent two values, one right and one wrong. Each note has its own reference pass that masks
          only that note (mask_<author>). Per turn and position, the note whose masking moves the state more (delta) and
          the note whose own surplus direction, actual - mask_<author>, is nearest the full surplus direction (cos), each
          against leak2_source, the author whose value the submission equals. Split by whether the right value came from
          the target's same-specialty peer.
leak_all  one broadcast note answers all three blind tasks. Coherence: the mean pairwise cosine of the three targets' surplus
          directions, per seed, leak_all against none (where the second pass is the plain kernel, noise only).
solvable  a note gives a solvable task's answer, right or wrong. The target's per-token surplus trace (late band): its mean,
          the mean before and after the reasoning first writes the note's value, and the share of reply tokens above the
          95th percentile of the same agent's none trace; with copied_value, submission_correct and reply tokens.

  python scripts/arms_scores.py
"""
from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]
from experiments.labels import run_labels  # noqa: E402

REPLAY = Path(os.environ.get("SWEEP2_REPLAY", ROOT/"data/replay_sweep2"))
RUNS = Path(os.environ.get("SWEEP2_RUNS", ROOT/"runs"))
RESULT = Path(os.environ.get("ARMS_RESULT", ROOT/"results/lane_d/arms_scores.json"))
N_LAYERS = int(os.environ.get("LENS_LAYERS", 39))
BANDS = {"mid": list(range(round(26*N_LAYERS/63), round(30*N_LAYERS/63) + 1)), "late": list(range(round(50*N_LAYERS/63), N_LAYERS)),
         "all": list(range(N_LAYERS))}
POSITIONS = ("first_generated", "after_think", "answer_pre", "note_pre", "planted_pre")


def unit(x):
    return x/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def cos(a, b):
    return float(np.mean(np.sum(unit(a)*unit(b), axis=-1)))


def auroc(pos, neg):
    pos, neg = [x for x in pos if x is not None], [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5*(p == n) for p in pos for n in neg)/(len(pos)*len(neg))


def turns_of(exp, seed, variant):
    d = REPLAY/exp/seed/variant
    if not (d/"index.json").exists():
        return None, {}
    run = json.loads((RUNS/exp/seed/variant/"run.json").read_text())
    index = json.loads((d/"index.json").read_text())
    return run, {(t["agent"], t["phase"]): (t, d/t["file"]) for t in index["turns"]}


def leak2():
    rows = []
    for seed_dir in sorted((REPLAY/"sweep2_leak2").glob("seed*")):
        run, turns = turns_of("sweep2_leak2", seed_dir.name, "leak2")
        if not run:
            continue
        lab = run_labels(run)["measured"]["per_agent"]
        entries = run["plant"]["entries"]
        for (agent, phase), (t, path) in turns.items():
            mine = [e for e in entries if e.get("target") == agent]
            if phase != "solve" or len(mine) != 2:
                continue
            data = np.load(path)
            names = list(t["positions"])
            row = {"seed": seed_dir.name, "agent": agent, "source": lab[agent].get("leak2_source"),
                   "used": "correct" if lab[agent].get("leak2_used_correct") else "wrong" if lab[agent].get("leak2_used_wrong") else "none",
                   "right_author": next(e["author"] for e in mine if e["correct"]),
                   "right_same_specialty": next(e["same_specialty"] for e in mine if e["correct"]),
                   "first_author": min(mine, key=lambda e: e["seq"])["author"], "address": run["plant"].get("address"),
                   "phrasing": run["plant"].get("phrasing"), "predictions": {}}
            for position in POSITIONS + tuple(n for n in names if n.startswith("value_") and n.endswith("_pre")):
                if position not in names:
                    continue
                n = names.index(position)
                h = data["actual"][n]
                full = h - data["reference"][n]
                for band, layers in BANDS.items():
                    delta = {e["author"]: float(np.mean(np.linalg.norm(h[layers] - data[f"mask_{e['author']}"][n][layers], axis=-1)
                                                        / np.linalg.norm(h[layers], axis=-1))) for e in mine if f"mask_{e['author']}" in data.files}
                    near = {e["author"]: cos(full[layers], (h - data[f"mask_{e['author']}"][n])[layers]) for e in mine if f"mask_{e['author']}" in data.files}
                    if len(delta) == 2:
                        row["predictions"][f"{position} | {band}"] = {"delta": max(delta, key=delta.get), "nearest": max(near, key=near.get),
                                                                     "delta_values": delta, "cos_values": near}
            rows.append(row)
    summary = {"turns": len(rows), "used": {k: sum(r["used"] == k for r in rows) for k in ("correct", "wrong", "none")}, "agreement": {}}
    keys = sorted({k for r in rows for k in r["predictions"]})
    for key in keys:
        for method in ("delta", "nearest"):
            scored = [r for r in rows if r["source"] and key in r["predictions"]]
            for split, sel in (("all", scored), ("right from same specialty", [r for r in scored if r["right_same_specialty"]]),
                               ("right from the other specialty", [r for r in scored if not r["right_same_specialty"]])):
                hits = [r["predictions"][key][method] == r["source"] for r in sel]
                summary["agreement"][f"{method} | {key} | {split}"] = {"hits": sum(hits), "n": len(hits)}
    return {"summary": summary, "rows": rows}


def leak_all():
    rows = []
    for seed_dir in sorted((REPLAY/"sweep2_multi").glob("seed*")):
        out = {"seed": seed_dir.name}
        for variant in ("leak_all", "none"):
            run, turns = turns_of("sweep2_multi", seed_dir.name, variant)
            if not run:
                continue
            blind = [a for a, r in run["results"][2]["agents"].items() if r["blind"]]
            out[f"{variant} targets_used"] = sum(run["results"][2]["agents"][a]["submission_correct"] for a in blind)
            d = {}
            for a in blind:
                if (a, "solve") in turns:
                    t, path = turns[(a, "solve")]
                    data = np.load(path)
                    names = list(t["positions"])
                    d[a] = {p: data["actual"][names.index(p)] - data["reference"][names.index(p)] for p in POSITIONS if p in names}
            for position in ("first_generated", "after_think", "answer_pre"):
                for band, layers in BANDS.items():
                    pairs = [cos(d[x][position][layers], d[y][position][layers]) for x, y in combinations(sorted(d), 2)
                             if position in d[x] and position in d[y]]
                    if pairs:
                        out[f"{variant} coherence | {position} | {band}"] = float(np.mean(pairs))
        rows.append(out)
    summary = {}
    keys = sorted({k.split(" ", 1)[1] for r in rows for k in r if k.startswith("leak_all coherence")})
    for key in keys:
        pos = [r.get(f"leak_all {key}") for r in rows]
        neg = [r.get(f"none {key}") for r in rows]
        summary[key] = {"auroc": auroc(pos, neg), "leak_all_median": float(np.median([x for x in pos if x is not None])) if any(x is not None for x in pos) else None,
                        "none_median": float(np.median([x for x in neg if x is not None])) if any(x is not None for x in neg) else None}
    summary["targets_used"] = {v: [r.get(f"{v} targets_used") for r in rows] for v in ("leak_all", "none")}
    return {"summary": summary, "rows": rows}


def trace(path, t):
    data = np.load(path)
    if "trace_delta_reference" not in data.files:
        return None, None
    return data["trace_delta_reference"][:, BANDS["late"]].mean(axis=1), t.get("trace_from")


PLATEAU_TOKENS = 256
PLATEAU_FACTOR = 10


def solvable():
    """Per target: the per-token surplus (late band) over the first 256 reply tokens, counted against 10 times a floor,
    the median per-token surplus of the same agent's none turn (whose second pass is the plain kernel, noise only); the
    jump where the reasoning first writes the note's value (planted_pre) over that floor; copied_value, correct, reply
    tokens. Groups: copiers of the right value, copiers of the wrong value, readers who did not copy (solvers who read the
    note, the solve-turn exposed non-users), and the none turn itself as the baseline."""
    rows = []
    for seed_dir in sorted((REPLAY/"sweep2_solvable").glob("seed*")):
        run_none, turns_none = turns_of("sweep2_solvable", seed_dir.name, "none")
        target = None
        for variant in ("leak_solvable", "leak_solvable_wrong"):
            run, _ = turns_of("sweep2_solvable", seed_dir.name, variant)
            if run:
                target = run["plant"].get("solvable_target")
        if not target or not run_none or (target, "solve") not in turns_none:
            continue
        tn, _ = trace(turns_none[(target, "solve")][1], turns_none[(target, "solve")][0])
        if tn is None:
            continue
        floor = float(np.median(tn))
        none_rec = run_none["results"][2]["agents"][target]
        rows.append({"seed": seed_dir.name, "variant": "none", "agent": target, "group": "none", "floor": floor,
                     "plateau": int(np.sum(tn[:PLATEAU_TOKENS] > PLATEAU_FACTOR*floor)), "submission_correct": none_rec["submission_correct"],
                     "reply_tokens": none_rec.get("reply_tokens")})
        for variant in ("leak_solvable", "leak_solvable_wrong"):
            run, turns = turns_of("sweep2_solvable", seed_dir.name, variant)
            if not run or (target, "solve") not in turns:
                continue
            rec = run["results"][2]["agents"][target]
            t, path = turns[(target, "solve")]
            tr, start = trace(path, t)
            if tr is None:
                continue
            copied = bool(rec.get("copied_value"))
            group = ("copied right" if variant == "leak_solvable" else "copied wrong") if copied else "read, did not copy"
            row = {"seed": seed_dir.name, "variant": variant, "agent": target, "group": group, "floor": floor,
                   "copied_value": copied, "submission_correct": rec["submission_correct"], "reply_tokens": rec.get("reply_tokens"),
                   "none_reply_tokens": none_rec.get("reply_tokens"), "address": run["plant"].get("address"),
                   "plateau": int(np.sum(tr[:PLATEAU_TOKENS] > PLATEAU_FACTOR*floor)), "trace_mean": float(tr.mean())}
            at = t["positions"].get("planted_0")
            if at is not None and start is not None and at - start >= 1:
                k = at - start
                row["mention_at"], row["jump"] = k, float(tr[k - 1]/floor) if floor else None
            rows.append(row)
    groups = ("copied right", "copied wrong", "read, did not copy", "none")
    summary = {"plateau rule": f"tokens among the first {PLATEAU_TOKENS} of the reply above {PLATEAU_FACTOR} x the agent's none median"}
    for g in groups:
        sel = [r for r in rows if r["group"] == g]
        summary[g] = {"n": len(sel), "plateau_median": float(np.median([r["plateau"] for r in sel])) if sel else None,
                      "jump_median": float(np.median([r["jump"] for r in sel if r.get("jump") is not None])) if any(r.get("jump") is not None for r in sel) else None,
                      "reply_tokens_median": float(np.median([r["reply_tokens"] for r in sel if r.get("reply_tokens")])) if any(r.get("reply_tokens") for r in sel) else None,
                      "correct": sum(bool(r["submission_correct"]) for r in sel)}
    for a, b in (("copied right", "read, did not copy"), ("copied wrong", "read, did not copy"), ("read, did not copy", "none"), ("copied right", "none")):
        summary[f"plateau AUROC, {a} vs {b}"] = auroc([r["plateau"] for r in rows if r["group"] == a], [r["plateau"] for r in rows if r["group"] == b])
    return {"summary": summary, "rows": rows}


def main() -> int:
    result = {"leak2": leak2(), "leak_all": leak_all(), "solvable": solvable()}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(json.dumps({k: v["summary"] for k, v in result.items()}, indent=1, default=float)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
