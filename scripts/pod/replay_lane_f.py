#!/usr/bin/env python3
"""Lane F replay on sweep2, episode 2, one model load, highest priority first: the turns lane D's replay did not trace.

Units, in order
  blind       the blind agent's solve turn in none, benign_k1 and leak, actual, kept reference and compacted reference (position
              ids closed up after each masked span, detector.render.compacted_positions), all traced over the whole reply
  nonblind_a  every non-blind agent's solve turn in none and benign_k1, actual and kept reference, traced
  nonblind_b  the same in leak and benign_k3
  solvable    the solvable target's solve turn in leak_solvable_wrong and none (runs/sweep2_solvable), traced, the note's value as
              the planted mention
  forbidden   the blind agent's solve turn in sweep2's forbidden_leak (none twin in the blind unit), actual and kept, traced
  forbidden2  the same in runs/sweep2_forbidden2, forbidden_leak and none
  norule      the same in runs/sweep2_norule, leak and none

Every pass runs on the explicit 4D-mask kernel from a shared prefix (scripts/replay.py --shared-prefix); no shuffled-span control
and no direction arrays, only the per-token relative surplus of each reference (trace_delta_*). Beside each .npz a slim
<name>.trace.npz holds just the trace arrays, for syncing down. Turns already on disk are skipped. Units run seed by seed, every
branch of a seed together, so a partial run still gives paired turns.

  SWARM_MODEL=qwen3_14b SWARM_ARTIFACTS=/root/artifacts python scripts/pod/replay_lane_f.py runs --out-root data/replay_lane_f \
      --units blind nonblind_a nonblind_b solvable --stop-at <unix time>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts"), str(ROOT/"third_party/maj-lens")]
import replay  # noqa: E402
from detector.render import load_turns  # noqa: E402

EPISODE = 2
# unit -> [(experiment, branch, who, compacted)]
UNITS = {
    "blind": [("sweep2", b, "blind", True) for b in ("none", "benign_k1", "leak")],
    "nonblind_a": [("sweep2", b, "nonblind", False) for b in ("none", "benign_k1")],
    "nonblind_b": [("sweep2", b, "nonblind", False) for b in ("leak", "benign_k3")],
    "solvable": [("sweep2_solvable", b, "target", False) for b in ("leak_solvable_wrong", "none")],
    # extra positive turns (second request): the blind agent in sweep2's forbidden_leak (its none twin is in the blind unit), in
    # sweep2_forbidden2's forbidden_leak and none, and in sweep2_norule's leak and none
    "forbidden": [("sweep2", "forbidden_leak", "blind", False)],
    "forbidden2": [("sweep2_forbidden2", b, "blind", False) for b in ("none", "forbidden_leak")],
    "norule": [("sweep2_norule", b, "blind", False) for b in ("none", "leak")],
}
BUSY = Path("/workspace/busy")


def solvable_target(seed_dir: Path) -> str | None:
    path = seed_dir/"leak_solvable"/"run.json"
    return json.loads(path.read_text())["plant"].get("solvable_target") if path.exists() else None


def slim(path: Path) -> None:
    data = np.load(path)
    keep = {k: data[k] for k in data.files if k.startswith("trace_")}
    np.savez(path.with_suffix(".trace.npz"), **keep)


def run_branch(model, tok, capture, lens, run_dir: Path, out: Path, who: str, compacted: bool, stop_at: float | None) -> tuple[int, bool]:
    """Replays the chosen agents' solve turns of run_dir; returns (turns done, stopped early)."""
    run = json.loads((run_dir/"run.json").read_text())
    if not run.get("complete"):
        print(f"skip {run_dir}: not complete", flush=True)
        return 0, False
    agents = run["results"][EPISODE]["agents"]
    blind = {a for a, r in agents.items() if r["blind"]}
    target = solvable_target(run_dir.parent) if who == "target" else None
    keep = blind if who == "blind" else ({a for a in agents if a not in blind} if who == "nonblind" else {target})
    old = json.loads((out/"index.json").read_text())["turns"] if (out/"index.json").exists() else []
    done = {row["file"] for row in old}
    planted = replay.planted_answers(run_dir, EPISODE)        # blind agent -> its task's withheld answer
    entries = run["plant"].get("entries") or []
    if who == "target":       # the note's value (wrong in leak_solvable_wrong); none reads where it writes the right one
        sibling = json.loads((run_dir.parent/"leak_solvable"/"run.json").read_text())["plant"]["entries"]
        planted = {target: (entries[0].get("value") if entries else None) or sibling[0]["value"]}
    rows, stopped = [], False
    for turn in load_turns(run_dir):
        if turn["episode_index"] != EPISODE or turn["phase"] != "solve" or turn["agent"] not in keep:
            continue
        path = out/f"{turn['agent']}_e{EPISODE}_solve.npz"
        if path.name in done and path.exists():
            if not path.with_suffix(".trace.npz").exists():
                slim(path)
            continue
        if stop_at and time.time() > stop_at:
            stopped = True
            break
        mentions = {"planted": planted[turn["agent"]]} if turn["agent"] in planted else None
        if who == "nonblind" and run["plant"].get("kind") == "leak" and blind:      # does a non-blind reader write the leaked value?
            mentions = {"leakval": planted[next(iter(blind))]}
        row = replay.replay_turn(model, tok, capture, turn, lens, path, shared_prefix=True, compacted=compacted, mentions=mentions,
                                 shuffled=False, trace=True, trace_direction=False)
        row.update({"blind": turn["agent"] in blind, "task_type": agents[turn["agent"]]["task_type"],
                    "note_author": turn["agent"] in {e.get("author") for e in entries},
                    "submission_correct": agents[turn["agent"]].get("submission_correct"), "reply_tokens": agents[turn["agent"]].get("reply_tokens")})
        slim(path)
        rows.append(row)
        if BUSY.parent.exists():
            BUSY.touch()
        print(f"{run_dir} {row['agent']} tokens {row['tokens']} prefix {row['shared_prefix_tokens']} masked {row['masked_tokens']} "
              f"{row['reference_kind']} {row['seconds']}s", flush=True)
    if rows or not old:
        out.mkdir(parents=True, exist_ok=True)
        (out/"index.json").write_text(json.dumps({"run": str(run_dir), "model": replay.MODEL_NAME, "layers": capture.layers,
                                                  "turns": old + rows}, indent=1))
    return len(rows), stopped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("--out-root", type=Path, default=ROOT/"data/replay_lane_f")
    parser.add_argument("--units", nargs="*", default=list(UNITS))
    parser.add_argument("--seeds", nargs="*", default=None, help="seed directory names, default every seed")
    parser.add_argument("--stop-at", type=float, default=None, help="unix time after which no new turn is started")
    args = parser.parse_args()
    from detector.jspace import Lens
    model, tok = replay.load_model()
    lens = Lens(replay.MODEL_DIR, replay.LENS_FILE)
    capture = replay.Capture(model, lens.layers)
    started = time.time()
    total = 0
    for unit in args.units:
        exps = {exp for exp, _, _, _ in UNITS[unit]}
        for exp in exps:
            seeds = sorted(p for p in (args.runs_root/exp).glob("seed*") if not args.seeds or p.name in args.seeds)
            for seed_dir in seeds:
                n = 0
                for _, branch, who, compacted in [u for u in UNITS[unit] if u[0] == exp]:
                    if not (seed_dir/branch/"run.json").exists():
                        continue
                    k, stopped = run_branch(model, tok, capture, lens, seed_dir/branch, args.out_root/exp/seed_dir.name/branch, who, compacted, args.stop_at)
                    n += k
                    if stopped:
                        print(f"== stop-at reached in {unit} {seed_dir.name} {branch}, {total + n} turns, {time.time() - started:.0f}s", flush=True)
                        return 0
                total += n
                print(f"== {unit} {exp} {seed_dir.name} {n} turns, {total} total, {time.time() - started:.0f}s elapsed", flush=True)
    print(f"== done {total} turns in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
