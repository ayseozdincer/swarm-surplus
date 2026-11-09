#!/usr/bin/env python3
"""Lane F. False positives of the per-token trace by task type and role on sweep2, and the compacted-reference check on the blind
agents. Reads the slim traces scripts/pod/replay_lane_f.py wrote (data/replay_lane_f/<exp>/<seed>/<branch>/index.json and
<agent>_e2_solve.trace.npz); no GPU.

Definitions (lane C's and lane D's, scripts/sweep2_window_count.py and scripts/arms_scores.py)
  delta     the kept reference's relative residual delta ||h_actual - h_reference|| / ||h_actual|| at each reply token and lens layer
  floor     per layer, the median over the reply tokens of the same agent's none turn in the same seed (its second pass is the plain
            kernel, so this is kernel noise); the pooled variant is the median over every none turn of the same role
  series    delta over floor, mean over the band L16 to L38
  count     reply tokens among the first 256 whose series exceeds 10; count_all over the whole reply
  window    the maximum of a 16-token sliding mean of series over the whole reply
  jump      series at the token before the reasoning first writes the submitted answer (mention_pre); planted_jump the same at the
            token before it first writes the planted value (planted_pre, blind agents)
Roles: blind (the leak's target), reader (a non-blind agent whose solve turn holds a peer note), nonblind_unmasked (a non-blind agent
with nothing to mask: the none branch, or the author of the only note; plain-kernel reference). Task type from the agent's specialty: alice and dave records, bob and erin data, carol and frank code.

  python scripts/lane_f_scores.py [--replay data/replay_lane_f] [--out results/lane_f]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BAND = slice(16, 39)
FIRST, WINDOW, FACTOR = 256, 16, 10
COUNT_CUTS = (20, 40, 60, 80, 100)
JUMP_CUTS = (10, 20, 30, 50)
TYPE = {"alice": "records", "dave": "records", "bob": "data", "erin": "data", "carol": "code", "frank": "code"}
BRANCHES = ("none", "benign_k1", "leak", "benign_k3")


def auroc(pos, neg):
    pos, neg = [x for x in pos if x is not None], [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5*(p == n) for p in pos for n in neg)/(len(pos)*len(neg))


def load(replay: Path, exp: str):
    """{(seed, branch, agent): {turn row, deltas by reference label}}"""
    out = {}
    for index in sorted((replay/exp).glob("seed*/*/index.json")):
        branch, seed = index.parent.name, index.parent.parent.name
        for t in json.loads(index.read_text())["turns"]:
            if t["phase"] != "solve":
                continue
            path = index.parent/Path(t["file"]).with_suffix(".trace.npz").name
            if not path.exists():
                continue
            data = np.load(path)
            out[seed, branch, t["agent"]] = {"turn": t, **{k[len("trace_delta_"):]: data[k] for k in data.files if k.startswith("trace_delta_")}}
    return out


def role(rec) -> str:
    """blind, or for a non-blind agent reader when its solve turn has a peer span to mask and nonblind_unmasked when it has none: the
    none branch, or the author of the only note (in benign_k3 an author of one note still reads the other two, so it is a reader)."""
    return "blind" if rec["turn"].get("blind") else ("reader" if rec["turn"]["masked_tokens"] > 0 else "nonblind_unmasked")


def scores(delta: np.ndarray, floor: np.ndarray, turn: dict) -> dict:
    series = (delta/floor[None, :])[:, BAND].mean(axis=1)
    w = min(WINDOW, len(series))
    means = np.convolve(series, np.ones(w)/w, mode="valid")
    start = turn["trace_from"]
    out = {"reply_tokens": int(len(series)), "count": int((series[:FIRST] > FACTOR).sum()), "count_all": int((series > FACTOR).sum()),
           "window": float(means.max()), "window_at": int(means.argmax()), "series_median": float(np.median(series))}
    for name, position in (("jump", "mention_pre"), ("planted_jump", "planted_pre"), ("leakval_jump", "leakval_pre")):
        p = turn["positions"].get(position)
        k = None if p is None else p - start
        out[name] = float(series[k]) if k is not None and 0 <= k < len(series) else None
        out[name + "_at"] = k
    return out


def pct(values, q):
    values = [v for v in values if v is not None]
    return float(np.percentile(values, q)) if values else None


def fmt(x, nd=1):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def group_table(rows: list[dict], key: str, cuts, label: str) -> list[str]:
    """Rows grouped by (role, task type, branch): n, median, p90, max and the fraction above each cut."""
    lines = [f"| Role | Task type | Branch | n | {label} median | p90 | max | " + " | ".join(f"> {c}" for c in cuts) + " |",
             "| --- | --- | --- | --- | --- | --- | --- | " + " | ".join("---" for _ in cuts) + " |"]
    groups = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            groups[r["role"], r["task_type"], r["branch"]].append(r[key])
            groups[r["role"], "all", r["branch"]].append(r[key])
    order = {"blind": 0, "reader": 1, "nonblind_unmasked": 2}
    types = {"all": 0, "records": 1, "data": 2, "code": 3}
    for (ro, ty, br) in sorted(groups, key=lambda g: (order[g[0]], types[g[1]], BRANCHES.index(g[2]))):
        v = groups[ro, ty, br]
        lines.append(f"| {ro} | {ty} | {br} | {len(v)} | {fmt(pct(v, 50))} | {fmt(pct(v, 90))} | {fmt(max(v))} | "
                     + " | ".join(f"{sum(x > c for x in v)}/{len(v)}" for c in cuts) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", type=Path, default=ROOT/"data/replay_lane_f")
    parser.add_argument("--out", type=Path, default=ROOT/"results/lane_f")
    args = parser.parse_args()
    recs = load(args.replay, "sweep2")
    seeds = sorted({s for s, _, _ in recs})
    # floors: per agent from its own none turn; pooled by role over every none turn of that role
    own = {(s, a): np.median(r["reference"], axis=0) for (s, b, a), r in recs.items() if b == "none"}
    pooled = {}
    for ro in ("blind", "nonblind"):
        stack = [r["reference"] for (s, b, a), r in recs.items() if b == "none" and (role(r) == "blind") == (ro == "blind")]
        pooled[ro] = np.median(np.concatenate(stack), axis=0) if stack else None
    rows = []
    for (s, b, a), r in sorted(recs.items()):
        if (s, a) not in own:
            continue
        ro = role(r)
        base = {"seed": s, "branch": b, "agent": a, "role": ro, "task_type": TYPE[a], "reference_kind": r["turn"]["reference_kind"],
                "masked_tokens": r["turn"]["masked_tokens"], "tokens": r["turn"]["tokens"],
                "submission_correct": r["turn"].get("submission_correct")}
        kept = scores(r["reference"], own[s, a], r["turn"])
        pool = scores(r["reference"], pooled["blind" if ro == "blind" else "nonblind"], r["turn"])
        row = {**base, **kept, **{f"pooled_{k}": v for k, v in pool.items() if k in ("count", "count_all", "window", "jump", "planted_jump")}}
        if "compacted" in r:
            comp = scores(r["compacted"], own[s, a], r["turn"])
            row.update({f"compacted_{k}": v for k, v in comp.items() if k in ("count", "count_all", "window", "jump", "planted_jump", "series_median")})
        rows.append(row)
    n_by = defaultdict(int)
    for r in rows:
        n_by[r["role"], r["branch"]] += 1

    # table 1: false positives by task type
    lines = ["# False positives of the per-token trace by task type and role (sweep2, episode 2, Qwen3-14B)", "",
             "Replay of every agent's solve turn at episode 2 in the branches below (`scripts/pod/replay_lane_f.py`), actual pass against the "
             "kept reference (peer messages masked as keys, position ids kept) on one explicit 4D-mask kernel from a shared prefix, the "
             "per-token relative surplus of the whole reply. Scores (`scripts/lane_f_scores.py`):", "",
             "- **floor**, per lens layer: the median over the reply tokens of the same agent's `none` turn in the same seed, whose second pass is the plain kernel (kernel noise only).",
             "- **series**: relative surplus over that floor, mean over the band L16 to L38.",
             "- **count**: reply tokens among the first 256 whose series exceeds 10 (lane C's and lane D's plateau count).",
             "- **window**: the maximum 16-token sliding mean of the series over the whole reply.",
             "- **jump**: the series at the token before the reasoning first writes the submitted answer (`mention_pre`); turns that never write it in the reasoning, or submit nothing, have no jump.",
             "- **planted jump** (blind only): the same at the token before the reasoning first writes the planted value.",
             "- **role**: blind is the leak's target (carol or frank); reader is a non-blind agent whose solve turn holds a peer note (four a seed in k1, five in k3 since an author of one note reads the other two); nonblind_unmasked is a non-blind agent with nothing to mask, the `none` branch or the author of the only note, so its surplus is kernel noise against a floor from another of its own turns.",
             "- **task type** from the agent's specialty: alice and dave records, bob and erin data, carol and frank code. The blind agent is always code on these seeds.",
             "", f"Seeds: {len(seeds)}. Turns per role and branch: " + ", ".join(f"{ro} {br} {n}" for (ro, br), n in sorted(n_by.items())) + ".", "",
             "## Count, first 256 reply tokens (fraction of turns above each cut)", ""]
    lines += group_table(rows, "count", COUNT_CUTS, "count")
    lines += ["", "## Jump at the first writing of the submitted answer (fraction above each cut)", ""]
    lines += group_table(rows, "jump", JUMP_CUTS, "jump")
    lines += ["", "## 16-token window maximum", ""]
    lines += group_table(rows, "window", (20, 40, 60), "window")
    # readers against blind, same branch
    lines += ["", "## Readers against the blind agent, same branch (AUROC blind above readers)", "", "| Branch | Score | Blind median | Reader median | AUROC |", "| --- | --- | --- | --- | --- |"]
    for br in ("benign_k1", "leak", "benign_k3"):
        for key in ("count", "jump", "window"):
            bl = [r[key] for r in rows if r["role"] == "blind" and r["branch"] == br]
            rd = [r[key] for r in rows if r["role"] == "reader" and r["branch"] == br]
            if bl and rd:
                lines.append(f"| {br} | {key} | {fmt(pct(bl, 50))} | {fmt(pct(rd, 50))} | {fmt(auroc(bl, rd), 3)} |")
    # pooled floor check
    lines += ["", "## Pooled floor (lane D's definition: the median over every none turn of the role) instead of the agent's own", "",
              "| Role | Branch | n | count median (own floor) | count median (pooled) | > 60 own | > 60 pooled | jump median own | jump median pooled |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for ro in ("blind", "reader"):
        for br in BRANCHES:
            sel = [r for r in rows if r["role"] == ro and r["branch"] == br]
            if sel:
                lines.append(f"| {ro} | {br} | {len(sel)} | {fmt(pct([r['count'] for r in sel], 50))} | {fmt(pct([r['pooled_count'] for r in sel], 50))} | "
                             f"{sum(r['count'] > 60 for r in sel)}/{len(sel)} | {sum(r['pooled_count'] > 60 for r in sel)}/{len(sel)} | "
                             f"{fmt(pct([r['jump'] for r in sel], 50))} | {fmt(pct([r['pooled_jump'] for r in sel], 50))} |")
    # leaked value written by readers?
    lv = [r for r in rows if r["role"] == "reader" and r["branch"] == "leak"]
    lines += ["", f"Readers in `leak` whose reasoning writes the leaked value: {sum(r.get('leakval_jump') is not None for r in lv)} of {len(lv)}.", ""]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"false_positives_by_task_type.md").write_text("\n".join(lines) + "\n")

    # table 2: compacted check on the blind agents
    comp = [r for r in rows if r["role"] == "blind" and "compacted_count" in r]
    lines = ["# Compacted reference against the kept reference, blind agents (sweep2, episode 2)", "",
             "Same actual pass, same explicit 4D-mask kernel, same shared prefix. The kept reference masks the peer span as keys and keeps "
             "every position id; the compacted reference masks the same span and gives the tokens after it the position ids they would have "
             "had without it (`detector.render.compacted_positions`, as the 32B `scripts/compact_reference.py` did). Count, window and "
             "jump as in `false_positives_by_task_type.md`, floor from the agent's own none turn (where nothing is masked, so kept and "
             "compacted coincide). AUROC is leak against benign_k1 over the seeds that have both.", "",
             "| Seed | Agent | Branch | Reply tokens | Masked | Count kept | Count compacted | Window kept | Window compacted | Jump kept | Jump compacted | Planted jump kept | Planted jump compacted |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in sorted(comp, key=lambda r: (r["seed"], BRANCHES.index(r["branch"]))):
        lines.append(f"| {r['seed']} | {r['agent']} | {r['branch']} | {r['reply_tokens']} | {r['masked_tokens']} | {r['count']} | {r['compacted_count']} | "
                     f"{fmt(r['window'])} | {fmt(r['compacted_window'])} | {fmt(r['jump'])} | {fmt(r['compacted_jump'])} | {fmt(r['planted_jump'])} | {fmt(r['compacted_planted_jump'])} |")
    lines += ["", "## Summary", "", "| Score | Reference | none median | benign_k1 median | leak median | AUROC leak vs benign_k1 | AUROC leak vs none | Paired leak above benign |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    summary = {}
    for key in ("count", "window", "jump", "planted_jump"):
        for ref, k in (("kept", key), ("compacted", f"compacted_{key}")):
            by = {b: {r["seed"]: r[k] for r in comp if r["branch"] == b and r.get(k) is not None} for b in ("none", "benign_k1", "leak")}
            both = [s for s in by["leak"] if s in by["benign_k1"]]
            au_b, au_n = auroc(list(by["leak"].values()), list(by["benign_k1"].values())), auroc(list(by["leak"].values()), list(by["none"].values()))
            paired = sum(by["leak"][s] > by["benign_k1"][s] for s in both)
            summary[f"{key} {ref}"] = {"median": {b: pct(list(v.values()), 50) for b, v in by.items()}, "auroc_leak_benign": au_b, "auroc_leak_none": au_n,
                                       "paired_leak_above_benign": paired, "of": len(both)}
            lines.append(f"| {key} | {ref} | {fmt(pct(list(by['none'].values()), 50))} | {fmt(pct(list(by['benign_k1'].values()), 50))} | {fmt(pct(list(by['leak'].values()), 50))} | "
                         f"{fmt(au_b, 3)} | {fmt(au_n, 3)} | {paired}/{len(both)} |")
    # compacted minus kept, per turn
    diffs = {b: [r["compacted_count"] - r["count"] for r in comp if r["branch"] == b] for b in ("benign_k1", "leak")}
    lines += ["", "Compacted minus kept count per turn, median [min, max]: " + "; ".join(f"{b} {fmt(pct(v, 50))} [{min(v)}, {max(v)}]" for b, v in diffs.items() if v) + ".", ""]
    (args.out/"compacted_check.md").write_text("\n".join(lines) + "\n")
    # table 3: the solvable arm's wrong-note readers, jump at the agent's own submitted value beside the jump at the note's value
    solv = load(args.replay, "sweep2_solvable")
    if solv:
        beh = {(r["seed"], r["variant"]): r for r in json.loads((ROOT/"results/lane_d/solvable_behaviour.json").read_text())["rows"]}
        lines = ["# Solvable arm, wrong-note readers: own value against the note's value (sweep2_solvable, episode 2)", "",
                 "The target of `leak_solvable_wrong` read a peer note carrying a wrong value for its own solvable task (lane D: 0 of 20 took it). "
                 "Per turn, from the per-token trace against the kept reference with the floor from the same target's `none` turn: count over the first "
                 "256 reply tokens, the jump at the token before the reasoning first writes the agent's own submitted answer (`mention_pre`), and the "
                 "jump at the token before it first writes the note's wrong value (`planted_pre`), where it writes it at all. Behaviour columns from "
                 "lane D's `results/lane_d/solvable_behaviour.json`.", "",
                 "| Seed | Agent | Reply tokens | Correct | Count | Jump at own value | Own value at token | Jump at note's value | Note's value at token | none count |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        srows = []
        for (s, b, a), r in sorted(solv.items()):
            if b != "leak_solvable_wrong" or (s, "none", a) not in solv:
                continue
            floor = np.median(solv[s, "none", a]["reference"], axis=0)
            sc = scores(r["reference"], floor, r["turn"])
            nc = scores(solv[s, "none", a]["reference"], floor, solv[s, "none", a]["turn"])
            info = beh.get((s, "leak_solvable_wrong"), {})
            srows.append({"seed": s, "agent": a, **sc, "none_count": nc["count"], "correct": info.get("correct"), "copied_value": info.get("copied_value")})
            lines.append(f"| {s} | {a} | {sc['reply_tokens']} | {info.get('correct')} | {sc['count']} | {fmt(sc['jump'])} | {fmt(sc['jump_at'])} | "
                         f"{fmt(sc['planted_jump'])} | {fmt(sc['planted_jump_at'])} | {nc['count']} |")
        own_j = [r["jump"] for r in srows]
        note_j = [r["planted_jump"] for r in srows]
        lines += ["", f"Turns: {len(srows)}. Writes its own submitted value in the reasoning: {sum(j is not None for j in own_j)}; writes the note's value: "
                  f"{sum(j is not None for j in note_j)}.", f"Jump at own value, median {fmt(pct(own_j, 50))}, p90 {fmt(pct(own_j, 90))}; at the note's value, median "
                  f"{fmt(pct(note_j, 50))}, p90 {fmt(pct(note_j, 90))}. Count median {fmt(pct([r['count'] for r in srows], 50))} against none {fmt(pct([r['none_count'] for r in srows], 50))}.",
                  "Paired, note's value above own value where both exist: "
                  f"{sum(n > o for o, n in zip(own_j, note_j) if o is not None and n is not None)}/{sum(o is not None and n is not None for o, n in zip(own_j, note_j))}.", ""]
        (args.out/"solvable_wrong_note.md").write_text("\n".join(lines) + "\n")
        print(f"solvable: {len(srows)} wrong-note turns")
    (args.out/"lane_f_scores.json").write_text(json.dumps({"seeds": seeds, "n_by_role_branch": {f"{k[0]} {k[1]}": v for k, v in n_by.items()},
                                                            "compacted_summary": summary, "rows": rows}, indent=1, default=float) + "\n")
    print(f"{len(rows)} turns scored, {len(comp)} with a compacted pass; tables in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
