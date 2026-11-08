#!/usr/bin/env python3
"""Lane D. One sweep2 seed's per-token trace as a CSV, the format of lane C's results/lane_c/trace_seed3000.csv: the blind target's
solve turn in the leak and benign_k1 branches, the first 256 reply tokens from first_generated. Delta over floor is the relative
residual delta of the kept reference (placebo: the shuffled-span reference) over the kernel floor of that layer, the floor being the
median over every reply token of the 20 none turns, whose reference is the plain kernel. Writes results/lane_d/trace_seed<N>.csv and
the two notes on one line in results/lane_d/trace_seed<N>_notes.json. Needs the tokenizer and the replay files, not the GPU.

  SWARM_MODEL=qwen3_14b SWARM_ARTIFACTS=/root/artifacts python scripts/lane_d_trace_csv.py seed5046
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
from detector.render import find_turn, load_turns, render_turn  # noqa: E402

REPLAY = Path(os.environ.get("SWEEP2_REPLAY", ROOT/"data/replay_sweep2"))/"sweep2"
RUNS = Path(os.environ.get("SWEEP2_RUNS", ROOT/"runs"))/"sweep2"
ARTIFACTS = Path(os.environ.get("SWARM_ARTIFACTS", "/root/artifacts"))
MARKERS = ("first_generated", "after_think", "mention_pre", "answer_pre")
BRANCHES = ("leak", "benign_k1")
TOKENS = 256


def solve_turn(seed: str, branch: str) -> tuple[dict, np.lib.npyio.NpzFile]:
    d = REPLAY/seed/branch
    turn = next(t for t in json.loads((d/"index.json").read_text())["turns"] if t["phase"] == "solve" and t.get("trace_from") is not None)
    return turn, np.load(d/turn["file"])


def main() -> int:
    seed = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(ARTIFACTS/"qwen3_14b", local_files_only=True)
    seeds = sorted(p.name for p in REPLAY.iterdir() if p.name.startswith("seed"))
    floor = np.median(np.concatenate([solve_turn(s, "none")[1]["trace_delta_reference"] for s in seeds]), axis=0)
    rows, notes = [], {"seed": seed, "floor_turns": len(seeds)}
    for branch in BRANCHES:
        turn, data = solve_turn(seed, branch)
        start = turn["positions"]["first_generated"]
        assert turn["trace_from"] == start, (turn["trace_from"], start)
        r = render_turn(tok, find_turn(load_turns(RUNS/seed/branch), turn["agent"], turn["episode_index"], "solve"))
        marks = {turn["positions"][m]: m for m in MARKERS if m in turn["positions"]}
        ratio = data["trace_delta_reference"]/floor[None, :]
        placebo = data["trace_delta_shuffled"]/floor[None, :]
        for k in range(min(TOKENS, len(ratio))):
            p = start + k
            rows.append({"branch": branch, "token_index": p, "token_text": tok.decode([r["ids"][p]]),
                         "delta_over_floor_L26": round(float(ratio[k, 26]), 3), "delta_over_floor_L16": round(float(ratio[k, 16]), 3),
                         "delta_over_floor_band_L16_38": round(float(ratio[k, 16:39].mean()), 3),
                         "placebo_over_floor_band_L16_38": round(float(placebo[k, 16:39].mean()), 3), "marker": marks.get(p, "")})
        run = json.loads((RUNS/seed/branch/"run.json").read_text())
        entry = run["plant"]["entries"][0]       # k = 1 in both branches; a benign note has no target
        notes.update({"target": turn["agent"], f"{branch}_author": entry["author"], f"{branch}_note": entry["text"],
                      f"{branch}_mention_pre_offset": turn["positions"]["mention_pre"] - start if "mention_pre" in turn["positions"] else None})
    out = ROOT/"results/lane_d"
    with (out/f"trace_{seed}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out/f"trace_{seed}_notes.json").write_text(json.dumps(notes, ensure_ascii=False) + "\n")
    print(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
