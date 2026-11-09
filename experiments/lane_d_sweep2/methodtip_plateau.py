#!/usr/bin/env python3
"""Lane D. What sets the size of the plateau in the method_tip arm: the note's length, or where in the reply it falls. Reads the replay
traces and experiments/lane_d_sweep2/results/methodtip_scores.json, needs the tokenizer and not the GPU. Writes experiments/lane_d_sweep2/results/methodtip_plateau.json.

  count and placebo   reply tokens among the first 256 above 10 times the floor, kept reference and the shuffled-span reference (a random
                      sanctioned span as long as the turn's peer spans, so it controls for how much is masked), mean over L16 to L38
  note_tokens         the planted note's length in tokens; masked_tokens is every peer span of the turn together
  bins                mean delta over floor in the reply's tokens 0-64, 64-128, 128-192, 192-256 and 256 on
  quotes              share of the words of four letters or more in the found window's text that are in the planted note

  SWARM_MODEL=qwen3_14b SWARM_ARTIFACTS=/root/artifacts python experiments/lane_d_sweep2/methodtip_plateau.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
REPLAY = Path(os.environ.get("SWEEP2_REPLAY", ROOT/"data/replay_sweep2"))
RUNS = Path(os.environ.get("SWEEP2_RUNS", ROOT/"runs"))
ARTIFACTS = Path(os.environ.get("SWARM_ARTIFACTS", "/root/artifacts"))
ARM, THIRD = "sweep2_methodtip", "sweep2_solvable"
BRANCHES = {"tip": (ARM, "method_tip"), "leak": (ARM, "leak_solvable"), "none": (ARM, "none"), "wrong": (THIRD, "leak_solvable_wrong"),
            "right": (THIRD, "leak_solvable")}
BAND = list(range(16, 39))
FIRST, FACTOR = 256, 10
EDGES = (0, 64, 128, 192, 256)
WORD = re.compile(r"[a-z_]{4,}")


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    scores = json.loads((ROOT/"experiments/lane_d_sweep2/results/methodtip_scores.json").read_text())
    tok = AutoTokenizer.from_pretrained(ARTIFACTS/"qwen3_14b", local_files_only=True)
    floor = np.array(scores["floor"])
    rows = []
    for r in scores["rows"]:
        exp, variant = BRANCHES[r["branch"]]
        d = REPLAY/exp/r["seed"]/variant
        turn = next(t for t in json.loads((d/"index.json").read_text())["turns"] if t["agent"] == r["agent"] and t["phase"] == "solve")
        data = np.load(d/turn["file"])
        series = (data["trace_delta_reference"]/floor[None, :])[:, BAND].mean(axis=1)
        placebo = (data["trace_delta_shuffled"]/floor[None, :])[:, BAND].mean(axis=1) if "trace_delta_shuffled" in data.files else None
        entries = json.loads((RUNS/exp/r["seed"]/variant/"run.json").read_text())["plant"].get("entries") or []
        note = entries[0]["text"] if entries else ""
        words = set(WORD.findall(r["excerpt"].lower()))
        rows.append({"seed": r["seed"], "branch": r["branch"], "group": r["group"], "count": int((series[:FIRST] > FACTOR).sum()),
                     "placebo": int((placebo[:FIRST] > FACTOR).sum()) if placebo is not None else None,
                     "note_tokens": len(tok.encode(note, add_special_tokens=False)), "masked_tokens": turn["masked_tokens"],
                     "bins": [float(series[a:b].mean()) if len(series) > a else None for a, b in zip(EDGES, EDGES[1:])] +
                             [float(series[EDGES[-1]:].mean()) if len(series) > EDGES[-1] else None],
                     "found_at": r["found_at"], "found_minus_mention": r.get("found_minus_mention"),
                     "quotes": len(words & set(WORD.findall(note.lower())))/len(words) if words and note else None,
                     "excerpt": r["excerpt"]})

    def med(values):
        values = [v for v in values if v is not None]
        return float(np.median(values)) if values else None
    table = {}
    for g in dict.fromkeys(r["group"] for r in rows):
        sel = [r for r in rows if r["group"] == g]
        table[g] = {"n": len(sel), "count": med([r["count"] for r in sel]), "placebo": med([r["placebo"] for r in sel]),
                    "note_tokens": med([r["note_tokens"] for r in sel]), "masked_tokens": med([r["masked_tokens"] for r in sel]),
                    "bins": [med([r["bins"][i] for r in sel]) for i in range(len(EDGES))], "found_at": med([r["found_at"] for r in sel]),
                    "found_at_or_near_mention": sum(r["found_minus_mention"] is not None and abs(r["found_minus_mention"]) <= 16 for r in sel),
                    "writes_value": sum(r["found_minus_mention"] is not None for r in sel), "quotes": med([r["quotes"] for r in sel]),
                    "found_quotes_half": sum((r["quotes"] or 0) >= 0.5 for r in sel)}
    corr = {b: {"count_note_tokens": spearman([r["count"] for r in rows if r["branch"] == b], [r["note_tokens"] for r in rows if r["branch"] == b]),
                "count_masked_tokens": spearman([r["count"] for r in rows if r["branch"] == b], [r["masked_tokens"] for r in rows if r["branch"] == b]),
                "placebo_masked_tokens": spearman([r["placebo"] for r in rows if r["branch"] == b], [r["masked_tokens"] for r in rows if r["branch"] == b])}
            for b in ("tip", "leak", "wrong", "right")}
    out = {"table": table, "spearman": corr, "rows": rows}
    (ROOT/"experiments/lane_d_sweep2/results/methodtip_plateau.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"table": table, "spearman": corr}, indent=0, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
