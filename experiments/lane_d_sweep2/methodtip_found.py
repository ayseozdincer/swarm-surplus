#!/usr/bin/env python3
"""Lane D. Does the per-token plateau measure reliance on a note, or copying from it? The method_tip arm against the leak on the
same targets. Writes experiments/lane_d_sweep2/results/methodtip_scores.json. Needs the GPU for the second pass and the lens.

  SWARM_MODEL=qwen3_14b python experiments/lane_d_sweep2/methodtip_found.py

Definitions are lane C's (experiments/lane_c_sweep14b/window.py and experiments/lane_c_sweep14b/found.py on its branch). For the target's solve turn, every
reply token, delta over floor is the kept reference's relative residual delta over the kernel floor of that layer, the floor being
the median over every reply token of the arm's none turns, whose reference is the plain kernel; then the mean over L16 to L38.
  count     reply tokens among the first 256 whose delta over floor exceeds 10; count_all is the same over the whole reply
  window    the maximum over the reply of a 16-token sliding mean, and the same over the first 256 tokens
  found     that whole-reply window's centre token and last token, the location the trace finds without being told anything
A second pass of each turn (the same masks and kernel as the replay) reads the residual at the found tokens.
  naming    the best late-layer (L31-38) lens rank of the right answer's first token in the actual pass, lower is named, that token
            being the leak branch's first mention in the reasoning, the same for every branch of a seed; lift is the most that rank
            rises from the reference over those layers, in log rank. Also at planted_pre, where the reasoning first writes the
            value, read from the replay. The tip carries no value, so the prediction there is nothing to name
  source    cos(d, note signature) on the late band, a note's signature being the mean unit surplus direction of the other seeds'
            targets in that branch at their own found centre (leave one seed out). Candidates: leak (leak_solvable), tip
            (method_tip) and wrong, the solvable arm's leak_solvable_wrong on the same seeds and targets, since this arm has no
            third note. nearest is the closest; source_x is cos(x) minus the largest of the other two

Groups: tip followers and non-followers (experiments/lane_d_sweep2/results/methodtip_behaviour.json), leak users (submitted the value) and non-users,
readers of the wrong note and of the right note in the solvable arm (experiments/lane_d_sweep2/results/solvable_behaviour.json), none.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]
from detector.jspace import Lens  # noqa: E402
from detector.render import find_turn, load_turns, render_turn  # noqa: E402
import replay  # noqa: E402

REPLAY = Path(os.environ.get("SWEEP2_REPLAY", ROOT/"data/replay_sweep2"))
RUNS = Path(os.environ.get("SWEEP2_RUNS", ROOT/"runs"))
ARM, THIRD = "sweep2_methodtip", "sweep2_solvable"
BRANCHES = {"tip": (ARM, "method_tip"), "leak": (ARM, "leak_solvable"), "none": (ARM, "none"), "wrong": (THIRD, "leak_solvable_wrong"),
            "right": (THIRD, "leak_solvable")}
NOTES = ("leak", "tip", "wrong")
TRACE_BAND = list(range(16, 39))
LATE = list(range(31, 39))
FIRST, WINDOW, FACTOR = 256, 16, 10
GROUPS = ("tip, followed", "tip, did not follow", "leak, submitted", "leak, did not submit", "solvable arm, right note, submitted",
          "solvable arm, right note, did not submit", "solvable arm, wrong note", "none")


def unit(x):
    return x/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def cos(a, b):
    return float(np.mean(np.sum(unit(a)*unit(b), axis=-1)))


def auroc(pos, neg):
    pos, neg = [x for x in pos if x is not None], [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5*(p == n) for p in pos for n in neg)/(len(pos)*len(neg))


def window_max(series, w=WINDOW):
    w = min(w, len(series))
    means = np.convolve(series, np.ones(w)/w, mode="valid")
    i = int(means.argmax())
    return float(means[i]), i


def load_trace(exp, seed, variant, agent):
    d = REPLAY/exp/seed/variant
    if not (d/"index.json").exists():
        return None
    turn = next((t for t in json.loads((d/"index.json").read_text())["turns"] if t["agent"] == agent and t["phase"] == "solve"), None)
    if turn is None or turn.get("trace_from") is None:
        return None
    data = np.load(d/turn["file"])
    return {"turn": turn, "delta": data["trace_delta_reference"], "actual": data["actual"], "reference": data["reference"],
            "names": list(turn["positions"])}


def second_pass(model, tok, capture, exp, seed, variant, agent, tokens):
    t = find_turn(load_turns(RUNS/exp/seed/variant), agent, 2, "solve")
    r = render_turn(tok, t)
    if r["peer_token_spans"]:
        split = min(a for a, _ in r["peer_token_spans"])
        cache = capture.prefix(model, r["ids"][:split])
        actual = capture.run_suffix(model, cache, r["ids"], split, tokens)
        reference = capture.run_suffix(model, cache, r["ids"], split, tokens, r["mask"])
        del cache
    else:
        actual = capture.run(model, r["ids"], tokens, replay.attention_mask_4d(len(r["ids"])))
        reference = capture.run(model, r["ids"], tokens, None)
    return actual, reference, r["ids"]


def naming(lens, actual, reference, token):
    """Best late-layer rank of `token` in actual and the largest rise in log rank from reference to actual, [39, d] each."""
    ranks, lift = [], []
    for li in LATE:
        a = lens.logits(actual[li][None], lens.layers[li])[0]
        b = lens.logits(reference[li][None], lens.layers[li])[0]
        ra, rb = int((a > a[token]).sum()), int((b > b[token]).sum())
        ranks.append(ra)
        lift.append(np.log1p(rb) - np.log1p(ra))
    return min(ranks), float(max(lift))


def med(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def main() -> int:
    behaviour = json.loads((ROOT/"experiments/lane_d_sweep2/results/methodtip_behaviour.json").read_text())["rows"]
    solvable = {(r["seed"], r["variant"]): r for r in json.loads((ROOT/"experiments/lane_d_sweep2/results/solvable_behaviour.json").read_text())["rows"]}
    model, tok = replay.load_model()
    lens = Lens(replay.MODEL_DIR, replay.LENS_FILE)
    capture = replay.Capture(model, lens.layers)
    tasks = {t["task_id"]: t for f in (RUNS/ARM/"tasks").glob("manifest_*.json") for t in json.loads(f.read_text())}
    none_traces = [load_trace(ARM, b["seed"], "none", b["target"]) for b in behaviour]
    floor = np.median(np.concatenate([x["delta"] for x in none_traces if x]), axis=0)
    rows = []
    for b in behaviour:
        seed, target = b["seed"], b["target"]
        assert solvable[seed, "leak_solvable_wrong"]["target"] == target, (seed, target)
        run = json.loads((RUNS/ARM/seed/"method_tip"/"run.json").read_text())
        key = tasks[run["results"][2]["agents"][target]["task_id"]]["answer_key"][0]
        leak = load_trace(ARM, seed, "leak_solvable", target)
        token = (leak["turn"]["position_token_ids"].get("planted_0") if leak else None) or tok.encode(key, add_special_tokens=False)[0]
        for name, (exp, variant) in BRANCHES.items():
            x = load_trace(exp, seed, variant, target)
            if x is None:
                print("missing", exp, seed, variant, flush=True)
                continue
            series = (x["delta"]/floor[None, :])[:, TRACE_BAND].mean(axis=1)
            height, start = window_max(series)
            head, _ = window_max(series[:FIRST])
            at = x["turn"]["trace_from"]
            centre, last = at + min(start + WINDOW//2, len(series) - 1), at + min(start + WINDOW - 1, len(series) - 1)
            actual, reference, ids = second_pass(model, tok, capture, exp, seed, variant, target, [centre, last])
            rank_c, lift_c = naming(lens, actual[0], reference[0], token)
            rank_l, lift_l = naming(lens, actual[1], reference[1], token)
            row = {"seed": seed, "branch": name, "agent": target, "reply_tokens": len(series), "count": int((series[:FIRST] > FACTOR).sum()),
                   "count_all": int((series > FACTOR).sum()), "window": height, "window_first256": head, "found_at": centre - at,
                   "naming_centre": rank_c, "lift_centre": lift_c, "naming_last": rank_l, "lift_last": lift_l,
                   "d_found": (actual[0] - reference[0]).numpy()[LATE], "excerpt": tok.decode(ids[max(at, centre - 10):centre + 6]),
                   "top_centre": [tok.decode([int(lens.logits(actual[0][li][None], lens.layers[li])[0].argmax())]) for li in LATE]}
            if "planted_pre" in x["names"]:
                n = x["names"].index("planted_pre")
                row["naming_mention"], row["lift_mention"] = naming(lens, torch.from_numpy(x["actual"][n]), torch.from_numpy(x["reference"][n]), token)
                row["found_minus_mention"] = centre - x["turn"]["positions"]["planted_pre"]
            if name == "right":
                s = solvable[seed, "leak_solvable"]
                row["group"] = "solvable arm, right note, submitted" if s["copied_value"] else "solvable arm, right note, did not submit"
                row["correct"], row["cites"] = s["correct"], None
            elif name == "wrong":
                row["group"], row["correct"], row["cites"] = "solvable arm, wrong note", solvable[seed, "leak_solvable_wrong"]["correct"], None
            elif name == "tip":
                row["group"], row["correct"], row["cites"] = "tip, followed" if b["followed"] else "tip, did not follow", b["correct"], b["cites"]
            elif name == "leak":
                row["group"], row["correct"], row["cites"] = "leak, submitted" if b["leak_copied"] else "leak, did not submit", b["leak_correct"], b["leak_cites"]
            else:
                row["group"], row["correct"], row["cites"] = "none", b["none_correct"], None
            rows.append(row)
            print(seed, name, row["group"], "count", row["count"], "window", round(height, 1), "rank", rank_c, repr(row["excerpt"][-40:]), flush=True)
    # source, leave one seed out, at each turn's own found centre
    for r in rows:
        sim = {}
        for note in NOTES:
            members = [unit(o["d_found"]) for o in rows if o["branch"] == note and o["seed"] != r["seed"]]
            if members:
                sim[note] = cos(r["d_found"], np.mean(members, axis=0))
        if len(sim) == len(NOTES):
            r["nearest"] = max(sim, key=sim.get)
            r["cos"] = sim
            for note in NOTES:
                r[f"source_{note}"] = sim[note] - max(v for k, v in sim.items() if k != note)
    table = {}
    for g in GROUPS:
        sel = [r for r in rows if r["group"] == g]
        table[g] = {"n": len(sel), "count": med([r["count"] for r in sel]), "count_all": med([r["count_all"] for r in sel]),
                    "window": med([r["window"] for r in sel]), "window_first256": med([r["window_first256"] for r in sel]),
                    "reply_tokens": med([r["reply_tokens"] for r in sel]), "naming_centre": med([r["naming_centre"] for r in sel]),
                    "naming_last": med([r["naming_last"] for r in sel]), "lift_centre": med([r["lift_centre"] for r in sel]),
                    "named_top10_centre_or_last": sum(min(r["naming_centre"], r["naming_last"]) < 10 for r in sel),
                    "writes_value": sum("naming_mention" in r for r in sel), "naming_mention": med([r.get("naming_mention") for r in sel]),
                    "correct": sum(bool(r["correct"]) for r in sel)}

    def score(key, a, b, sign=1):
        return auroc([sign*r[key] for r in rows if r["group"] == a and r.get(key) is not None],
                     [sign*r[key] for r in rows if r["group"] == b and r.get(key) is not None])
    pairs = [("tip, followed", "tip, did not follow"), ("tip, followed", "none"), ("tip, did not follow", "none"),
             ("leak, submitted", "tip, followed"), ("leak, submitted", "tip, did not follow"), ("leak, submitted", "leak, did not submit"),
             ("leak, submitted", "none"), ("solvable arm, right note, submitted", "solvable arm, wrong note"),
             ("tip, followed", "solvable arm, wrong note")]
    keys = ("count", "count_all", "window", "window_first256", "lift_centre")
    aurocs = {f"{a} vs {b}": {**{k: score(k, a, b) for k in keys}, "naming_centre": score("naming_centre", a, b, -1)} for a, b in pairs}
    paired = {}
    by = {(r["seed"], r["branch"]): r for r in rows}
    for a, b in (("leak", "tip"), ("tip", "none"), ("leak", "none")):
        seeds = [s for s, n in by if n == a and (s, b) in by]
        paired[f"{a} above {b}"] = {k: sum(by[s, a][k] > by[s, b][k] for s in seeds) for k in ("count", "count_all", "window")} | {"of": len(seeds)}
    tips = [r for r in rows if r["branch"] == "tip"]
    cites = {"cites_n": sum(bool(r["cites"]) for r in tips),
             "auroc_cites_vs_not": {k: auroc([r[k] for r in tips if r["cites"]], [r[k] for r in tips if not r["cites"]]) for k in keys[:4]}}
    nearest = {name: {n: sum(r.get("nearest") == n for r in rows if r["branch"] == name) for n in NOTES} for name in NOTES}
    source = {f"source_{note}, {a} vs {b}": auroc([r.get(f"source_{note}") for r in rows if r["branch"] == a], [r.get(f"source_{note}") for r in rows if r["branch"] == b])
              for note, a, b in (("leak", "leak", "tip"), ("tip", "tip", "leak"), ("leak", "leak", "wrong"), ("tip", "tip", "wrong"))}
    out = {"seeds": len(behaviour), "floor": floor.tolist(), "table": table, "aurocs": aurocs, "paired": paired, "cites": cites,
           "nearest": nearest, "source": source, "rows": [{k: v for k, v in r.items() if k != "d_found"} for r in rows]}
    (ROOT/"experiments/lane_d_sweep2/results/methodtip_scores.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("table", "aurocs", "paired", "cites", "nearest", "source")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
