#!/usr/bin/env python3
"""Lane C. A trace score that does not know where the first mention is, and the per-token trace of one seed. CPU only, on the
stage-1 traces (scripts/lane_c_blind.py). Writes results/lane_c/window.json and results/lane_c/trace_seed3000.csv.

  SWARM_MODEL=qwen3_14b python scripts/lane_c_window.py --replay data/replay_14b/blind/sweep14b --runs runs/sweep14b

For the blind agent's solve turn, every reply token from first_generated on, delta over floor is the relative residual delta
of the reference (or the placebo) divided by the kernel floor of that layer. The floor is the median over every reply token of
the ten none turns, whose reference is the plain kernel. Per turn and band (L16 to L38, all layers):
  window w      the maximum over the reply of the mean of delta over floor in a sliding window of w tokens, w = 8, 16, 32
  above 10      the fraction of reply tokens whose delta over floor exceeds 10, and their count. A leak reply is short, so the
                fraction carries reply length; the count does not divide by it. Reply length alone is scored as a control
Then leak against benign_k1 and leak against none, AUROC with matched-seed paired wins, and for the leak branch where each
window maximum lands, its centre in tokens after the first mention's mention_pre.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]
from replay import LENS_LAYERS, MODEL_DIR  # noqa: E402

WINDOWS = (8, 16, 32)
BANDS = {"L16-38": list(range(16, 39)), "all": list(LENS_LAYERS)}
BRANCHES = ("leak", "benign_k1", "none")
MARKERS = ("first_generated", "after_think", "mention_pre", "answer_pre")


def load(replay: Path):
    out = {}
    for seed in sorted(p for p in replay.iterdir() if p.name.startswith("seed")):
        out[seed.name] = {}
        for branch in BRANCHES + ("benign_k3", "proposal_k1", "proposal_k3"):
            if not (seed/branch/"index.json").exists():
                continue
            turn = json.loads((seed/branch/"index.json").read_text())["turns"][0]
            data = np.load(seed/branch/turn["trace_file"])
            out[seed.name][branch] = {"turn": turn, "positions": data["positions"],
                                      "reference": data["delta_reference"], "placebo": data["delta_placebo"]}
    return out


def window_max(series, w):
    if len(series) < w:
        return float(series.mean()), 0
    means = np.convolve(series, np.ones(w)/w, mode="valid")
    i = int(means.argmax())
    return float(means[i]), i + w//2


def auroc(pos, neg):
    return sum((p > n) + 0.5*(p == n) for p in pos for n in neg)/(len(pos)*len(neg))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", type=Path, default=ROOT/"data/replay_14b/blind/sweep14b")
    parser.add_argument("--runs", type=Path, default=ROOT/"runs/sweep14b")
    parser.add_argument("--out", type=Path, default=ROOT/"results/lane_c/window.json")
    parser.add_argument("--csv-seed", default="seed3000")
    args = parser.parse_args()
    traces = load(args.replay)
    floor = np.median(np.concatenate([t["none"]["reference"] for t in traces.values() if "none" in t]), axis=0)
    scores = {}
    for seed, branches in traces.items():
        scores[seed] = {}
        for branch, t in branches.items():
            start = t["positions"].tolist().index(t["turn"]["positions"]["first_generated"])
            mention = t["turn"]["positions"].get("mention_pre")
            mention = None if mention is None else t["positions"].tolist().index(mention) - start
            cell = {"reply_tokens": len(t["positions"]) - start, "mention_offset": mention}
            for kind in ("reference", "placebo"):
                ratio = t[kind][start:]/floor[None, :]
                for band, layers in BANDS.items():
                    series = ratio[:, layers].mean(1)
                    cell[f"{kind} {band} above 10"] = float((series > 10).mean())
                    cell[f"{kind} {band} count above 10"] = int((series > 10).sum())
                    for w in WINDOWS:
                        value, centre = window_max(series, w)
                        cell[f"{kind} {band} window {w}"] = value
                        cell[f"{kind} {band} window {w} centre"] = centre
                        if mention is not None:
                            cell[f"{kind} {band} window {w} centre - mention"] = centre - mention
            scores[seed][branch] = cell
    keys = [k for k in next(iter(scores.values()))["leak"] if (" window " in k and not k.endswith(("centre", "mention"))) or k.endswith("above 10")]
    keys.append("reply tokens, negated")
    for branches in scores.values():
        for cell in branches.values():
            cell["reply tokens, negated"] = -cell["reply_tokens"]
    claims = {}
    for negative in ("benign_k1", "none", "benign_k3", "proposal_k1", "proposal_k3"):
        pairs = [(s[  "leak"], s[negative]) for s in scores.values() if negative in s]
        claims[f"leak vs {negative}"] = {k: {"auroc": auroc([p[k] for p, _ in pairs], [n[k] for _, n in pairs]),
                                             "paired_wins": sum(p[k] > n[k] for p, n in pairs), "pairs": len(pairs),
                                             "leak_median": float(np.median([p[k] for p, _ in pairs])),
                                             "negative_median": float(np.median([n[k] for _, n in pairs]))} for k in keys}
    landing = {k: [scores[s]["leak"].get(f"{k} centre - mention") for s in sorted(scores)]
               for k in keys if "window" in k}
    result = {"floor": floor.tolist(), "windows": WINDOWS, "bands": {k: [v[0], v[-1]] for k, v in BANDS.items()},
              "scores": scores, "claims": claims, "leak_window_centre_minus_mention": landing}
    args.out.write_text(json.dumps(result, indent=1) + "\n")

    from transformers import AutoTokenizer
    from detector.render import find_turn, load_turns, render_turn
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    rows = []
    for branch in ("leak", "benign_k1"):
        t = traces[args.csv_seed][branch]
        turn = t["turn"]
        r = render_turn(tok, find_turn(load_turns(args.runs/args.csv_seed/branch), turn["agent"], turn["episode_index"], "solve"))
        marks = {turn["positions"][m]: m for m in MARKERS if m in turn["positions"]}
        ratio = t["reference"]/floor[None, :]
        placebo = t["placebo"]/floor[None, :]
        for k, p in enumerate(t["positions"].tolist()):
            rows.append({"branch": branch, "token_index": p, "token_text": tok.decode([r["ids"][p]]),
                         "delta_over_floor_L26": round(float(ratio[k, 26]), 3), "delta_over_floor_L16": round(float(ratio[k, 16]), 3),
                         "delta_over_floor_band_L16_38": round(float(ratio[k, 16:39].mean()), 3),
                         "placebo_over_floor_band_L16_38": round(float(placebo[k, 16:39].mean()), 3), "marker": marks.get(p, "")})
    path = args.out.parent/f"trace_{args.csv_seed}.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, cells in claims.items():
        print(name, {k: f"{v['auroc']:.2f} ({v['paired_wins']}/{v['pairs']})" for k, v in cells.items() if k.startswith("reference")})
    print("wrote", args.out, path, len(rows), "rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
