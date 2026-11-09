#!/usr/bin/env python3
"""Lane G. The pre-registered leak2 attribution test on the new seeds, pooled with lane D's ten turns
(results/lane_g/REPORT.md, "Pre-registration"). Reads the replay of runs/sweep2_leak2 and its run records; writes a JSON of
per-turn scores and prints the report tables.

Per turn (the blind agent's leak2 solve turn), position and band: d = actual - reference (both notes masked) and, per
note X, d_X = actual - mask_X. nearest: the note whose d_X direction is nearest d (mean cosine over the band's layers).
delta: the note whose masking moves the state more (mean over the band of |d_X| / |actual|). signature: the cross-seed class
signatures of scripts/leak2_uptake.py (class = right or wrong value x posted first or second, mean unit d of the other seeds'
users of that class, leave one seed out), within the new turns. Each is scored against leak2_source, one-sided binomial
against 0.5. Primary: nearest at answer_pre, late band, pooled with lane D's rows (results/lane_d/arms_scores.json, key
leak2; lane D's mention_pre is its value_<source>_pre, the token before the reasoning first writes the submitted value;
its class-signature counts come from results/lane_d/leak2_uptake.json).

  python scripts/lane_g_leak2.py --replay data/replay_sweep2 --runs runs --out results/lane_g/leak2_scores.json
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.labels import collect, run_labels  # noqa: E402

EXP = "sweep2_leak2"
N_LAYERS = 39
BANDS = {"mid": list(range(round(26*N_LAYERS/63), round(30*N_LAYERS/63) + 1)),
         "late": list(range(round(50*N_LAYERS/63), N_LAYERS)), "all": list(range(N_LAYERS))}
POSITIONS = ("answer_pre", "mention_pre", "first_generated", "note_pre", "after_think")
PRIMARY = ("nearest", "answer_pre", "late")
SECONDARY = [("nearest", "mention_pre", "late"), ("nearest", "first_generated", "late"),
             ("delta", "answer_pre", "late"), ("delta", "mention_pre", "late"), ("delta", "first_generated", "late"),
             ("signature", "answer_pre", "late"), ("signature", "mention_pre", "late"), ("signature", "first_generated", "late")]
EXPLORATORY = [("nearest", "note_pre", "late"), ("nearest", "after_think", "late"), ("nearest", "answer_pre", "mid"),
               ("nearest", "answer_pre", "all"), ("nearest", "mention_pre", "all"), ("nearest", "first_generated", "all")]
# lane D's behaviour aggregates that arms_scores.json does not carry per row (its report, "Behaviour, leak2")
LANE_D_BEHAVIOUR = {"blind_correct_none": (0, 10), "disclosed_correct": (5, 5), "strict_correct_leak2": (3, 23),
                    "strict_correct_none": (5, 35)}


def unit(x):
    return x/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def cos(a, b):
    return float(np.mean(np.sum(unit(a)*unit(b), axis=-1)))


def p_one_sided(hits, n):
    return sum(comb(n, k) for k in range(hits, n + 1))/2**n if n else None


def load_turns(replay: Path, runs: Path) -> list[dict]:
    turns = []
    for seed_dir in sorted((replay/EXP).glob("seed*")):
        d = seed_dir/"leak2"
        if not (d/"index.json").exists():
            continue
        run = json.loads((runs/EXP/seed_dir.name/"leak2"/"run.json").read_text())
        lab = run_labels(run)["measured"]["per_agent"]
        entries = run["plant"]["entries"]
        for t in json.loads((d/"index.json").read_text())["turns"]:
            agent = t["agent"]
            mine = [e for e in entries if e.get("target") == agent]
            if t["phase"] != "solve" or len(mine) != 2:
                continue
            first = min(mine, key=lambda e: e["seq"])["author"]
            right = next(e for e in mine if e["correct"])
            rec = {"seed": seed_dir.name, "agent": agent, "source": lab[agent].get("leak2_source"),
                   "used": ("correct" if lab[agent].get("leak2_used_correct") else "wrong" if lab[agent].get("leak2_used_wrong") else "none"),
                   "right_author": right["author"], "right_same_specialty": bool(right.get("same_specialty")), "first_author": first,
                   "address": run["plant"].get("address"), "phrasing": run["plant"].get("phrasing"),
                   "disclosed": lab[agent].get("disclosed_source"), "submission_correct": lab[agent].get("submission_correct"),
                   "classes": {e["author"]: ("right" if e["correct"] else "wrong", "first" if e["author"] == first else "second") for e in mine},
                   "reply_tokens": lab[agent].get("reply_tokens"), "positions_present": [], "arrays": {}, "predictions": {}}
            data = np.load(d/t["file"])
            names = list(t["positions"])
            for position in POSITIONS:
                if position not in names:
                    continue
                n = names.index(position)
                h = data["actual"][n]
                full = h - data["reference"][n]
                dx = {e["author"]: h - data[f"mask_{e['author']}"][n] for e in mine if f"mask_{e['author']}" in data.files}
                if len(dx) != 2:
                    continue
                rec["positions_present"].append(position)
                rec["arrays"][position] = {"h": h, "d": full, "dx": dx}
                for band, layers in BANDS.items():
                    near = {a: cos(full[layers], x[layers]) for a, x in dx.items()}
                    delta = {a: float(np.mean(np.linalg.norm(x[layers], axis=-1)/np.linalg.norm(h[layers], axis=-1))) for a, x in dx.items()}
                    rec["predictions"][f"{position} | {band}"] = {"nearest": max(near, key=near.get), "delta": max(delta, key=delta.get),
                                                                 "cos_values": near, "delta_values": delta}
            turns.append(rec)
    return turns


def within_turn(turns, method, position, band):
    key = f"{position} | {band}"
    rows = [t for t in turns if t["source"] and key in t["predictions"]]
    hits = sum(t["predictions"][key][method] == t["source"] for t in rows)
    return hits, len(rows)


def signature(turns, position, layers):
    scored = [t for t in turns if t["source"] and position in t["arrays"]]
    rows = []
    for t in scored:
        d = t["arrays"][position]["d"][layers]
        sim = {}
        for note, cls in t["classes"].items():
            members = [unit(u["arrays"][position]["d"][layers]) for u in scored if u["seed"] != t["seed"] and u["classes"][u["source"]] == cls]
            if members:
                sim[note] = cos(d, np.mean(members, axis=0))
        if len(sim) == 2:
            rows.append({"seed": t["seed"], "pick": max(sim, key=sim.get), "source": t["source"], "cos": sim})
    return sum(r["pick"] == r["source"] for r in rows), len(rows), rows


def lane_d_counts(arms: Path, uptake: Path):
    """(method, position, band) -> (hits, n) on lane D's ten turns, and its rows."""
    rows = json.loads(arms.read_text())["leak2"]["rows"] if arms.exists() else []
    out = {}
    for method in ("nearest", "delta"):
        for position in POSITIONS:
            for band in BANDS:
                hits = n = 0
                for r in rows:
                    if not r["source"]:
                        continue
                    key = f"{'value_' + r['source'] + '_pre' if position == 'mention_pre' else position} | {band}"
                    if key in r["predictions"]:
                        n += 1
                        hits += r["predictions"][key][method] == r["source"]
                out[(method, position, band)] = (hits, n)
    if uptake.exists():
        sig = json.loads(uptake.read_text())["nearest_signature"]
        for position in POSITIONS:
            for band in BANDS:
                v = sig.get(f"{position} | {band}")
                out[("signature", position, band)] = (v["hits"], v["n"]) if v else (None, None)
    return out, rows


def fmt(hits, n):
    return "-" if n in (None, 0) else f"{hits}/{n}"


def fmt_p(hits, n):
    p = p_one_sided(hits, n) if n else None
    return "-" if p is None else (f"{p:.3g}" if p < 0.001 else f"{p:.3f}")


def behaviour(rows: list[dict]) -> dict:
    src = [r for r in rows if r["source"]]
    same = [r for r in src if r["source"] == (r["right_author"] if r["right_same_specialty"] else None) or
            (not r["right_same_specialty"] and r["source"] != r["right_author"])]
    # the same-specialty peer's note: the right note when right_same_specialty, else the other note
    same_taken = [r for r in src if (r["source"] == r["right_author"]) == r["right_same_specialty"]]
    right_same = [r for r in src if r["right_same_specialty"]]
    right_other = [r for r in src if not r["right_same_specialty"]]
    return {"n": len(rows), "used_correct": sum(r["used"] == "correct" for r in rows), "used_wrong": sum(r["used"] == "wrong" for r in rows),
            "used_none": sum(r["used"] == "none" for r in rows),
            "same_specialty_taken": (len(same_taken), len(src)),
            "same_specialty_taken_when_right": (sum(r["source"] == r["right_author"] for r in right_same), len(right_same)),
            "same_specialty_taken_when_wrong": (sum(r["source"] != r["right_author"] for r in right_other), len(right_other)),
            "first_posted_taken": (sum(r["source"] == r["first_author"] for r in src), len(src)),
            "right_by_address": {addr: (sum(r["used"] == "correct" for r in rows if r["address"] == addr), sum(r["address"] == addr for r in rows))
                                 for addr in ("name", "broadcast")}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", type=Path, default=ROOT/"data/replay_sweep2")
    parser.add_argument("--runs", type=Path, default=ROOT/"runs")
    parser.add_argument("--lane-d-arms", type=Path, default=ROOT/"results/lane_d/arms_scores.json")
    parser.add_argument("--lane-d-uptake", type=Path, default=ROOT/"results/lane_d/leak2_uptake.json")
    parser.add_argument("--out", type=Path, default=ROOT/"results/lane_g/leak2_scores.json")
    args = parser.parse_args()
    turns = load_turns(args.replay, args.runs)
    lane_d, lane_d_rows = lane_d_counts(args.lane_d_arms, args.lane_d_uptake)
    tests = {}
    for label, keys in (("primary", [PRIMARY]), ("secondary", SECONDARY), ("exploratory", EXPLORATORY)):
        for method, position, band in keys:
            if method == "signature":
                hits, n, rows = signature(turns, position, BANDS[band])
            else:
                hits, n = within_turn(turns, method, position, band)
                rows = None
            dh, dn = lane_d.get((method, position, band), (None, None))
            tests[f"{method} | {position} | {band}"] = {
                "kind": label, "new": [hits, n], "p_new": p_one_sided(hits, n), "lane_d": [dh, dn],
                "pooled": [hits + dh, n + dn] if dn else None, "p_pooled": p_one_sided(hits + dh, n + dn) if dn else None,
                "signature_rows": rows}
    # splits of the primary as in lane D
    key = f"{PRIMARY[1]} | {PRIMARY[2]}"
    splits = {}
    for name, sel in (("right from the same specialty", lambda r: r["right_same_specialty"]), ("right from the other specialty", lambda r: not r["right_same_specialty"]),
                      ("used the right value", lambda r: r["used"] == "correct"), ("used the wrong value", lambda r: r["used"] == "wrong"),
                      ("named", lambda r: r["address"] == "name"), ("broadcast", lambda r: r["address"] == "broadcast")):
        new = [t for t in turns if t["source"] and key in t["predictions"] and sel(t)]
        old = [r for r in lane_d_rows if r["source"] and key in r["predictions"] and sel(r)]
        splits[name] = {"new": [sum(t["predictions"][key]["nearest"] == t["source"] for t in new), len(new)],
                        "lane_d": [sum(r["predictions"][key]["nearest"] == r["source"] for r in old), len(old)]}
    # behaviour
    lab = collect(args.runs/EXP)["variants"] if (args.runs/EXP).exists() else {}
    agg = {}
    for variant in ("leak2", "none"):
        v = lab.get(variant)
        if v:
            agg[variant] = {k: [v[k]["num"], v[k]["den"]] for k in ("leak2_used_correct", "leak2_used_wrong", "leak2_used_none", "blind_correct_free",
                                                                   "disclosed_source", "strict_wrong_free") if v.get(k) and v[k]["den"] is not None}
    beh_new, beh_old = behaviour(turns), behaviour(lane_d_rows)
    out = {"turns": len(turns), "with_source": sum(1 for t in turns if t["source"]), "primary": f"{PRIMARY[0]} | {PRIMARY[1]} | {PRIMARY[2]}",
           "tests": tests, "primary_splits": splits, "behaviour_new": beh_new, "behaviour_lane_d": beh_old, "behaviour_aggregates_new": agg,
           "behaviour_aggregates_lane_d": LANE_D_BEHAVIOUR, "lane_d_counts": {f"{m} | {p} | {b}": v for (m, p, b), v in lane_d.items()},
           "rows": [{k: v for k, v in t.items() if k != "arrays"} for t in turns]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=float) + "\n")

    print(f"new turns {len(turns)}, with leak2_source {out['with_source']}\n")
    print("| Test | Where | New | p | Lane D | Pooled | p |\n| --- | --- | --- | --- | --- | --- | --- |")
    for name, v in tests.items():
        method, position, band = name.split(" | ")
        tag = {"primary": "**primary** ", "secondary": "", "exploratory": "exploratory, "}[v["kind"]]
        pooled = v["pooled"] or (None, None)
        print(f"| {tag}{method} | {position}, {band} | {fmt(*v['new'])} | {fmt_p(*v['new'])} | {fmt(*v['lane_d'])} | {fmt(*pooled)} | {fmt_p(*pooled) if v['pooled'] else '-'} |")
    print("\nPrimary by split (new / lane D):")
    for name, v in splits.items():
        print(f"  {name}: {fmt(*v['new'])} / {fmt(*v['lane_d'])}")
    print("\nBehaviour (new):", json.dumps({k: v for k, v in beh_new.items()}, default=str))
    print("Behaviour (lane D rows):", json.dumps({k: v for k, v in beh_old.items()}, default=str))
    print("Aggregates (new):", json.dumps(agg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
