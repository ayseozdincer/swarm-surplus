#!/usr/bin/env python3
"""Lane D replay of a planted-channel experiment, episode 2, one model load, highest priority first.

Every turn runs the actual pass, the kept reference (peer messages masked) and the shuffled-span control (a random
sanctioned span of the same length masked instead), on the explicit 4D-mask kernel from a shared prefix (scripts/replay.py).
No compacted reference unless --compacted. The blind agent's solve turn also keeps the per-token trace of its reply.

Plans, in priority order
  sweep2   P1 the blind agent's solve turn in leak, forbidden_leak, none, benign_k1 (with the trace)
           P2 every free reviewer's verdict turn in command_k1, command_k3, none
           P3 every agent's solve turn in none, leak, forbidden_leak, benign_k1, command_k1, the blind agent's in benign_k3
              and command_k3
  norule   P1 the blind agent's solve turn in leak, none, benign_k1 (with the trace); P2 verdicts in command_k1, command_k3,
           none; P3 every agent's solve turn in none, leak, benign_k1, command_k1

--follow N waits for seeds to finish generating (every planned branch has a run.json) and works on the highest-priority
unit among the seeds that are ready, until N seeds have every unit done. Turns already on disk are skipped.

  python scripts/pod/replay_sweep2.py runs/sweep2 --out-root data/replay_sweep2 --follow 20
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
# priority -> [(variant, phase, who)], who is blind (with the trace), free (not a plant author) or all
PLANS = {
    "sweep2": {
        1: [("leak", "solve", "blind"), ("forbidden_leak", "solve", "blind"), ("none", "solve", "blind"), ("benign_k1", "solve", "blind")],
        2: [("command_k1", "verdict", "free"), ("command_k3", "verdict", "free"), ("none", "verdict", "free")],
        3: [("none", "solve", "all"), ("leak", "solve", "all"), ("forbidden_leak", "solve", "all"), ("benign_k1", "solve", "all"),
            ("command_k1", "solve", "all"), ("benign_k3", "solve", "blind"), ("command_k3", "solve", "blind")],
    },
    "norule": {
        1: [("leak", "solve", "blind"), ("none", "solve", "blind"), ("benign_k1", "solve", "blind")],
        2: [("command_k1", "verdict", "free"), ("command_k3", "verdict", "free"), ("none", "verdict", "free")],
        3: [("none", "solve", "all"), ("leak", "solve", "all"), ("benign_k1", "solve", "all"), ("command_k1", "solve", "all")],
    },
}


def run_unit(model, tok, capture, lens, run_dir: Path, out: Path, phase: str, who: str, compacted: bool) -> int:
    run = json.loads((run_dir/"run.json").read_text())
    if not run.get("complete"):
        print(f"skip {run_dir}: not complete", flush=True)
        return 0
    agents = run["results"][EPISODE]["agents"]
    keep = {a for a, r in agents.items() if (who == "blind" and r["blind"]) or (who == "free" and not r["instructed"]) or who == "all"}
    old = json.loads((out/"index.json").read_text())["turns"] if (out/"index.json").exists() else []
    done = {row["file"] for row in old}
    planted = replay.planted_answers(run_dir, EPISODE)
    rows = []
    for turn in load_turns(run_dir):
        if turn["episode_index"] != EPISODE or turn["phase"] != phase or turn["agent"] not in keep:
            continue
        path = out/f"{turn['agent']}_e{EPISODE}_{turn['phase']}.npz"
        if path.name in done and path.exists():
            continue
        mentions = {"planted": planted[turn["agent"]]} if turn["agent"] in planted else None
        trace = agents[turn["agent"]]["blind"] and phase == "solve"
        rows.append(replay.replay_turn(model, tok, capture, turn, lens, path, True, compacted, mentions, shuffled=True, trace=trace))
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
    parser.add_argument("--plan", choices=list(PLANS), default="sweep2")
    parser.add_argument("--follow", type=int, default=None, metavar="N")
    parser.add_argument("--stop-at", type=float, default=None, help="unix time after which no new unit is started")
    parser.add_argument("--compacted", action="store_true")
    args = parser.parse_args()
    from detector.jspace import Lens
    model, tok = replay.load_model()
    lens = Lens(replay.MODEL_DIR, replay.LENS_FILE)
    capture = replay.Capture(model, lens.layers)
    plan = PLANS[args.plan]
    variants = {v for units in plan.values() for v, _, _ in units}
    finished: set[tuple[str, int]] = set()
    started = time.time()
    while True:
        if args.stop_at and time.time() > args.stop_at:
            print("stop-at reached", flush=True)
            break
        ready = [p for p in sorted(args.root.glob("seed*")) if all((p/v/"run.json").exists() for v in variants)]
        todo = sorted(((prio, seed_dir) for seed_dir in ready for prio in plan if (seed_dir.name, prio) not in finished),
                      key=lambda x: (x[0], x[1].name))
        if not todo:
            complete = {s for s, _ in finished if all((s, p) in finished for p in plan)}
            if args.follow is None or len(complete) >= args.follow:
                break
            time.sleep(30)
            continue
        prio, seed_dir = todo[0]
        n = sum(run_unit(model, tok, capture, lens, seed_dir/v, args.out_root/Path(*(seed_dir/v).parts[-3:]), phase, who, args.compacted)
                for v, phase, who in plan[prio])
        finished.add((seed_dir.name, prio))
        print(f"== P{prio} {seed_dir.name} {n} turns, {time.time() - started:.0f}s elapsed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
