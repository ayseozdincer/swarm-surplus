#!/usr/bin/env python3
"""Lane D. One solvable-arm target's per-token trace as a CSV, the columns of results/lane_d/trace_seed5034.csv without the placebo: the note
branch and the none branch of a seed, the whole reply from first_generated. Delta over floor is the relative residual delta of the kept
reference over the kernel floor of that layer, the floor being the median over every reply token of the arm's 20 none turns of the target,
whose reference is the plain kernel. Markers: first_generated, after_think, answer_pre, mention_pre at the token before the first writing of
the note's value (in none, of the right value, since none never holds a wrong one), and own_answer_pre where the agent first writes the
answer it submits, when that is a different place. Writes results/lane_d/trace_solvable_<seed>_<case>.csv and one line of JSON in
trace_solvable_<seed>_<case>_notes.json. Needs the tokenizer and the replay files, not the GPU.

  python scripts/solvable_trace_csv.py 5006 computed leak_solvable
  python scripts/solvable_trace_csv.py 5021 checked leak_solvable_wrong
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

ARM = "sweep2_solvable"
REPLAY = Path(os.environ.get("SWEEP2_REPLAY", ROOT/"data/replay_sweep2"))/ARM
RUNS = Path(os.environ.get("SWEEP2_RUNS", ROOT/"runs"))/ARM
TOKENIZER = os.environ.get("SWARM_TOKENIZER", "Qwen/Qwen3-14B")


def traced_turn(seed: str, branch: str) -> tuple[dict, np.lib.npyio.NpzFile]:
    d = REPLAY/seed/branch
    turn = next(t for t in json.loads((d/"index.json").read_text())["turns"] if t["phase"] == "solve" and t.get("trace_from") is not None)
    return turn, np.load(d/turn["file"])


def main() -> int:
    seed, case, note_branch = f"seed{sys.argv[1]}", sys.argv[2], sys.argv[3]
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    seeds = sorted(p.name for p in REPLAY.iterdir() if p.name.startswith("seed"))
    floor = np.median(np.concatenate([traced_turn(s, "none")[1]["trace_delta_reference"] for s in seeds]), axis=0)
    rows, first_writing = [], {}
    for branch in (note_branch, "none"):
        turn, data = traced_turn(seed, branch)
        agent, start = turn["agent"], turn["positions"]["first_generated"]
        r = render_turn(tok, find_turn(load_turns(RUNS/seed/branch), agent, turn["episode_index"], "solve"))
        delta = data["trace_delta_reference"]
        assert len(r["ids"]) == turn["tokens"] and turn["trace_from"] == start and len(delta) == turn["tokens"] - start, (seed, branch)
        positions = turn["positions"]
        marks = {positions[m]: m for m in ("first_generated", "after_think", "answer_pre") if m in positions}
        marks[positions["planted_pre"]] = "mention_pre"
        if positions["mention_pre"] != positions["planted_pre"]:
            marks[positions["mention_pre"]] = "own_answer_pre"
        ratio = delta/floor[None, :]
        for k in range(len(ratio)):
            p = start + k
            rows.append({"branch": branch, "token_index": p, "token_text": tok.decode([r["ids"][p]]),
                         "delta_over_floor_L26": round(float(ratio[k, 26]), 3), "delta_over_floor_L16": round(float(ratio[k, 16]), 3),
                         "delta_over_floor_band_L16_38": round(float(ratio[k, 16:39].mean()), 3), "marker": marks.get(p, "")})
        first_writing[branch] = {"reply_token": positions["planted_pre"] - start, "reply_tokens": len(ratio),
                                 "share_of_reply": round((positions["planted_pre"] - start)/len(ratio), 3)}
        if positions["mention_pre"] != positions["planted_pre"]:
            first_writing[branch]["own_answer_reply_token"] = positions["mention_pre"] - start
    run = json.loads((RUNS/seed/note_branch/"run.json").read_text())
    entry = run["plant"]["entries"][0]
    rec = run["results"][2]["agents"][turn["agent"]]
    notes = {"seed": seed, "case": case, "branch": note_branch, "agent": turn["agent"], "author": entry["author"], "note": entry["text"],
             "value": entry["value"], "value_correct": entry["correct"], "submitted": rec["submission"]["answer"], "submission_correct": rec["submission_correct"],
             "first_writing": first_writing, "floor_turns": len(seeds)}
    out = ROOT/"results/lane_d"
    name = f"trace_solvable_{sys.argv[1]}_{case}"
    with (out/f"{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out/f"{name}_notes.json").write_text(json.dumps(notes, ensure_ascii=False) + "\n")
    print(json.dumps(notes, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
