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


def solvable():
    rows = []
    for seed_dir in sorted((REPLAY/"sweep2_solvable").glob("seed*")):
        run_none, turns_none = turns_of("sweep2_solvable", seed_dir.name, "none")
        for variant in ("leak_solvable", "leak_solvable_wrong"):
            run, turns = turns_of("sweep2_solvable", seed_dir.name, variant)
            if not run:
                continue
            target = run["plant"].get("solvable_target")
            if (target, "solve") not in turns:
                continue
            rec = run["results"][2]["agents"][target]
            t, path = turns[(target, "solve")]
            tr, start = trace(path, t)
            row = {"seed": seed_dir.name, "variant": variant, "agent": target, "copied_value": rec.get("copied_value"),
                   "submission_correct": rec["submission_correct"], "reply_tokens": rec.get("reply_tokens"),
                   "value_written": "planted_0" in t["positions"], "address": run["plant"].get("address")}
            if tr is not None:
                row["trace_mean"] = float(tr.mean())
                at = t["positions"].get("planted_0")
                if at is not None and start is not None and at - start >= 1:
                    k = at - start
                    row["mention_at"], row["before_mention"], row["after_mention"] = k, float(tr[max(0, k - 16):k].mean()), float(tr[k:k + 16].mean())
                if run_none and (target, "solve") in turns_none:
                    tn, _ = trace(turns_none[(target, "solve")][1], turns_none[(target, "solve")][0])
                    if tn is not None:
                        row["none_trace_mean"] = float(tn.mean())
                        row["share_above_none_p95"] = float(np.mean(tr > np.percentile(tn, 95)))
                        row["none_reply_tokens"] = run_none["results"][2]["agents"][target].get("reply_tokens")
            rows.append(row)
    summary = {}
    for variant in ("leak_solvable", "leak_solvable_wrong"):
        sel = [r for r in rows if r["variant"] == variant]
        summary[variant] = {"targets": len(sel), "copied_value": sum(bool(r["copied_value"]) for r in sel),
                            "submission_correct": sum(bool(r["submission_correct"]) for r in sel),
                            "reply_tokens_median": float(np.median([r["reply_tokens"] for r in sel if r["reply_tokens"]])) if sel else None,
                            "none_reply_tokens_median": float(np.median([r["none_reply_tokens"] for r in sel if r.get("none_reply_tokens")]))
                            if any(r.get("none_reply_tokens") for r in sel) else None}
    copiers = [r for r in rows if r["copied_value"] and "share_above_none_p95" in r]
    others = [r for r in rows if not r["copied_value"] and "share_above_none_p95" in r]
    summary["copiers vs not, share of reply above the none p95"] = {"auroc": auroc([r["share_above_none_p95"] for r in copiers], [r["share_above_none_p95"] for r in others]),
                                                                    "n": [len(copiers), len(others)]}
    return {"summary": summary, "rows": rows}


def main() -> int:
    result = {"leak2": leak2(), "leak_all": leak_all(), "solvable": solvable()}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(json.dumps({k: v["summary"] for k, v in result.items()}, indent=1, default=float)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
