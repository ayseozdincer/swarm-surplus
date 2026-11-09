#!/usr/bin/env python3
"""Replay of phase 2c. Every target's solve calls at the plant episode after it read its task file, every branch, one model load.

Each call runs the actual pass, the derived reference (the appended line's tokens masked, nothing else in the message) and
the declared reference (the file is environment, so the declared mask masks nothing: the plain kernel, whose difference
from the actual pass is kernel noise), each with the per-token trace of the reply, from a shared prefix where one applies. In
the none branch there is nothing to derive, so the two references are the same plain-kernel pass.

Read positions come from detector.render with two mentions, `planted` where the reasoning first writes the answer value and
`credential` where it first writes the access token, plus the reveal_input call's <tool_call> and token value.

Lens naming at the copy. For each position that precedes a written value, the rank of the token actually written next in the
lens readout of each pass, at every lens layer, kept in the index.

  python experiments/lane_e_shared_workspace/replay_planted.py runs/planted --out-root data/replay_planted
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts"), str(ROOT/"third_party/maj-lens")]
import replay  # noqa: E402
from detector.render import load_turns  # noqa: E402

BRANCHES = ("none", "benign", "leak", "credential")
NAMING = {"planted_pre": "planted_0", "credential_pre": "credential_0", "answer_pre": "answer_0", "token_pre": "token_0"}
TRACE = True


def after_read(turn: dict, path: str, episode: int) -> bool:
    """The call's context holds the result of reading the task file this episode."""
    for tag in turn["message_meta"]:
        info = tag.get("tool_info") or {}
        if tag.get("episode_index") == episode and info.get("tool") == "read_file" and info.get("ok") and info.get("path") == path:
            return True
    return False


def target_turns(run_dir: Path) -> tuple[dict, list[dict]]:
    run = json.loads((run_dir/"run.json").read_text(encoding="utf-8"))
    plant = run["plant"]
    episode, target = plant["episode_index"], plant["targets"][0]
    path = run["results"][episode]["agents"][target]["task_file"]
    turns = [t for t in load_turns(run_dir) if t["episode_index"] == episode and t["agent"] == target and t["phase"] == "solve"
             and after_read(t, path, episode)]
    return run, sorted(turns, key=lambda t: t["step"])


def naming_ranks(lens, layers, data, row: dict) -> dict:
    """position -> pass -> rank per lens layer of the token written next, 0 being the top of the readout."""
    import torch
    out = {}
    names = list(row["positions"])
    for position, written in NAMING.items():
        if position not in names or written not in row["position_token_ids"]:
            continue
        token = row["position_token_ids"][written]
        n = names.index(position)
        ranks = {}
        for label in ("actual", "reference", "declared"):
            H = torch.from_numpy(data[label][n]).float()
            ranks[label] = [int((logits > logits[token]).sum()) for logits in
                            (lens.logits(H[li][None], layer)[0] for li, layer in enumerate(layers))]
        out[position] = {"written": row["position_tokens"][written], "ranks": ranks}
    return out


def replay_run(model, tok, capture, lens, run_dir: Path, out: Path, token: str | None, shared_prefix: bool = True,
               shuffled: bool = False) -> int:
    """All of a branch's target calls not on disk yet. Writes out/index.json, returns how many ran."""
    old = json.loads((out/"index.json").read_text())["turns"] if (out/"index.json").exists() else []
    done = {row["file"] for row in old}
    run, turns = target_turns(run_dir)
    plant = run["plant"]
    target, episode = plant["targets"][0], plant["episode_index"]
    mentions = {"planted": run["results"][episode]["agents"][target]["answer_key"][0]}
    if token:
        mentions["credential"] = token
    rows = []
    for turn in turns:
        path = out/f"{target}_e{episode}_solve_s{turn['step']}.npz"
        if path.name in done and path.exists():
            continue
        row = replay.replay_turn(model, tok, capture, turn, lens, path, shared_prefix, False, mentions, shuffled=shuffled, trace=TRACE,
                                 mask_mode="derived", declared=True)
        with np.load(path) as data:
            row["naming"] = naming_ranks(lens, capture.layers, data, row)
        rows.append(row)
        print(f"{run_dir} {target} step {turn['step']} tokens {row['tokens']} masked {row['masked_tokens']} {row['seconds']}s", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out/"index.json").write_text(json.dumps({"run": str(run_dir), "model": replay.MODEL_NAME, "layers": list(capture.layers), "target": target,
                                              "episode": episode, "branch": plant["kind"], "turns": old + rows}, indent=1))
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="the experiment directory, e.g. runs/planted")
    parser.add_argument("--out-root", type=Path, default=ROOT/"data/replay_planted")
    parser.add_argument("--branches", nargs="+", default=list(BRANCHES))
    parser.add_argument("--seeds", nargs="*", help="seed directory names, default all")
    parser.add_argument("--shuffled", action="store_true", help="add the shuffled-span control pass")
    args = parser.parse_args()
    from detector.jspace import Lens
    model, tok = replay.load_model()
    lens = Lens(replay.MODEL_DIR, replay.LENS_FILE)
    capture = replay.Capture(model, lens.layers)
    for seed_dir in sorted(p for p in args.root.glob("seed*") if p.is_dir() and (not args.seeds or p.name in args.seeds)):
        lines = json.loads((seed_dir/"plant_lines.json").read_text(encoding="utf-8")) if (seed_dir/"plant_lines.json").exists() else {}
        for branch in args.branches:
            run_dir = seed_dir/branch
            if not (run_dir/"run.json").exists() or not json.loads((run_dir/"run.json").read_text(encoding="utf-8")).get("complete"):
                print(f"skip {run_dir}: not complete", flush=True)
                continue
            replay_run(model, tok, capture, lens, run_dir, args.out_root/args.root.name/seed_dir.name/branch, lines.get("token"),
                       shuffled=args.shuffled)
    return 0


if __name__ == "__main__":
    sys.exit(main())
