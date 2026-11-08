#!/usr/bin/env python3
"""Lane D replay plan for sweep2, seed by seed so a cut-off run still leaves whole seeds. One model load.

Episode 2 of every branch, shared prefix, kept and compacted references, the 14B lens (SWARM_MODEL=qwen3_14b).
  solve turns, every agent     none, benign_k1, leak, forbidden_leak,    claims 1 to 3, the three note signatures (command_k1
                               command_k1                                takes proposal_k1's place), forbidden contrast
  solve turn, blind agent      benign_k3, command_k3                     claim 2 with k3, the blind agent in every branch
  verdict turns, every agent   command_k1, command_k3                    the command contrast
  verdict turns, every agent   none                                      the unexposed verdict for the command contrast
The first pass over all seeds does the solve turns of none, leak, forbidden_leak, benign_k1 and command_k1 and the command
verdicts; the second the blind agent in benign_k3 and command_k3 and the none verdicts.

  python scripts/pod/replay_sweep2.py runs/sweep2 --out-root data/replay_sweep2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts"), str(ROOT/"third_party/maj-lens")]
import replay  # noqa: E402
from detector.render import load_turns  # noqa: E402

EPISODE = 2
# Two passes over the seeds, so a deadline drops extras before it drops seeds. variant, phases, blind only
CORE = [("none", ("solve",), False), ("leak", ("solve",), False), ("forbidden_leak", ("solve",), False),
        ("benign_k1", ("solve",), False), ("command_k1", ("solve",), False), ("command_k1", ("verdict",), False),
        ("command_k3", ("verdict",), False)]
EXTRA = [("benign_k3", ("solve",), True), ("command_k3", ("solve",), True), ("none", ("verdict",), False)]


def run_plan(model, tok, capture, lens, run_dir: Path, out: Path, phases, blind_only: bool) -> int:
    old = json.loads((out/"index.json").read_text())["turns"] if (out/"index.json").exists() else []
    done = {row["file"] for row in old}
    blind = replay.blind_agents(run_dir, EPISODE)
    planted = replay.planted_answers(run_dir, EPISODE)
    rows = []
    for turn in load_turns(run_dir):
        if turn["episode_index"] != EPISODE or turn["phase"] not in phases or (blind_only and turn["agent"] not in blind):
            continue
        path = out/f"{turn['agent']}_e{EPISODE}_{turn['phase']}.npz"
        if path.name in done and path.exists():
            continue
        mentions = {"planted": planted[turn["agent"]]} if turn["agent"] in planted else None
        rows.append(replay.replay_turn(model, tok, capture, turn, lens, path, True, True, mentions))
        r = rows[-1]
        print(f"{run_dir} {r['agent']} {r['phase']} tokens {r['tokens']} prefix {r['shared_prefix_tokens']} {r['seconds']}s", flush=True)
    if rows or not old:
        out.mkdir(parents=True, exist_ok=True)
        (out/"index.json").write_text(json.dumps({"run": str(run_dir), "model": replay.MODEL_NAME, "layers": capture.layers,
                                                  "turns": old + rows}, indent=1))
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out-root", type=Path, default=ROOT/"data/replay_sweep2")
    parser.add_argument("--none-verdicts", action="store_true")
    parser.add_argument("--seeds", nargs="*")
    parser.add_argument("--stop-at", type=float, default=None, help="unix time after which no new seed is started")
    parser.add_argument("--follow", type=int, default=None, metavar="N",
                        help="generation is still running: replay each seed once all its branches have a run.json, until N "
                             "seeds are done, then the extra pass")
    args = parser.parse_args()
    from detector.jspace import Lens
    model, tok = replay.load_model()
    lens = Lens(replay.MODEL_DIR, replay.LENS_FILE)
    capture = replay.Capture(model, lens.layers)
    if args.follow:
        variants = {v for v, _, _ in CORE + EXTRA}
        done, started = [], time.time()
        while len(done) < args.follow and not (args.stop_at and time.time() > args.stop_at):
            ready = [p for p in sorted(args.root.glob("seed*")) if p.name not in done and all((p/v/"run.json").exists() for v in variants)]
            if not ready:
                time.sleep(30)
                continue
            for seed_dir in ready:
                n = sum(run_plan(model, tok, capture, lens, seed_dir/v, args.out_root/Path(*(seed_dir/v).parts[-3:]), phases, blind)
                        for v, phases, blind in CORE if json.loads((seed_dir/v/"run.json").read_text()).get("complete"))
                done.append(seed_dir.name)
                print(f"== core {seed_dir.name} {n} turns, {len(done)} seeds, {time.time() - started:.0f}s elapsed", flush=True)
        args.seeds = done
    passes = [("none verdicts", [("none", ("verdict",), False)])] if args.none_verdicts else (
        [("extra", EXTRA)] if args.follow else [("core", CORE), ("extra", EXTRA)])
    seeds = sorted(p for p in args.root.glob("seed*") if p.is_dir() and (not args.seeds or p.name in args.seeds))
    started = time.time()
    for name, plan in passes:
        for seed_dir in seeds:
            if args.stop_at and time.time() > args.stop_at:
                print(f"stop-at reached in the {name} pass before {seed_dir.name}", flush=True)
                return 0
            n = 0
            for variant, phases, blind_only in plan:
                run_dir = seed_dir/variant
                if not (run_dir/"run.json").exists() or not json.loads((run_dir/"run.json").read_text()).get("complete"):
                    print(f"skip {run_dir}: not complete", flush=True)
                    continue
                n += run_plan(model, tok, capture, lens, run_dir, args.out_root/Path(*run_dir.parts[-3:]), phases, blind_only)
            print(f"== {name} {seed_dir.name} {n} turns, {time.time() - started:.0f}s elapsed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
