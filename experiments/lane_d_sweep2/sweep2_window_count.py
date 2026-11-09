#!/usr/bin/env python3
"""Lane D. Lane C's two plateau scores on sweep2's blind agents, leak against benign_k1 and none: count above 10 in the first 256 reply
tokens and the 16-token window, delta over floor over L16 to L38, floor from the 20 none turns. Writes experiments/lane_d_sweep2/results/sweep2_window_count.json.
Needs the replay files, not the GPU.

  python experiments/lane_d_sweep2/sweep2_window_count.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPLAY = Path(os.environ.get("SWEEP2_REPLAY", ROOT/"data/replay_sweep2"))/"sweep2"
BRANCHES = ("leak", "benign_k1", "none")


def load(seed, branch):
    d = REPLAY/seed/branch
    turn = next(t for t in json.loads((d/"index.json").read_text())["turns"] if t["phase"] == "solve" and t.get("trace_from") is not None)
    return turn, np.load(d/turn["file"])["trace_delta_reference"]


def auroc(pos, neg):
    return sum((p > n) + 0.5*(p == n) for p in pos for n in neg)/(len(pos)*len(neg))


def main() -> int:
    seeds = sorted(p.name for p in REPLAY.iterdir() if p.name.startswith("seed"))
    floor = np.median(np.concatenate([load(s, "none")[1] for s in seeds]), axis=0)
    rows = {}
    for seed in seeds:
        rows[seed] = {}
        for branch in BRANCHES:
            turn, delta = load(seed, branch)
            series = (delta/floor[None, :])[:, 16:39].mean(axis=1)
            w = min(16, len(series))
            means = np.convolve(series, np.ones(w)/w, mode="valid")
            start = int(means.argmax())
            mention = turn["positions"].get("mention_pre")
            rows[seed][branch] = {"window": float(means[start]), "window_at": start, "count": int((series[:256] > 10).sum()),
                                  "mention_at": None if mention is None else mention - turn["positions"]["first_generated"]}
    summary = {}
    for key in ("count", "window"):
        values = {b: [rows[s][b][key] for s in seeds] for b in BRANCHES}
        summary[key] = {"auroc_leak_benign": auroc(values["leak"], values["benign_k1"]), "auroc_leak_none": auroc(values["leak"], values["none"]),
                        "paired_leak_above_benign": sum(a > b for a, b in zip(values["leak"], values["benign_k1"])), "of": len(seeds),
                        "median": {b: float(np.median(values[b])) for b in BRANCHES}}
    summary["window_lands_on_mention_leak"] = sum(r["leak"]["mention_at"] is not None and r["leak"]["window_at"] - 8 <= r["leak"]["mention_at"] <= r["leak"]["window_at"] + 24
                                                  for r in rows.values())
    (ROOT/"experiments/lane_d_sweep2/results/sweep2_window_count.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
