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
  leak2    the blind agent's solve turn in leak2 and none, with one extra reference per note (mask_<author>)
  multi    the three blind agents' solve turns in leak_all and none
  solvable the solvable target's solve turn in leak_solvable, leak_solvable_wrong and none, with the trace
  methodtip the same target's solve turn in method_tip, leak_solvable and none, with the trace; in method_tip the planted positions
           read where it writes the right answer, which the tip does not carry

It waits for seeds to finish generating (every planned branch has a run.json) and always works on the highest unit of
--order among the seeds that are ready, until every arm has its --expect seeds done. Turns already on disk are skipped.

  python scripts/pod/replay_sweep2.py runs --out-root data/replay_sweep2
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
    # each leak2 note gets its own reference pass (mask_<author>), so the surplus can be attributed to one of the two
    "leak2": {1: [("leak2", "solve", "blind"), ("none", "solve", "blind")]},
    "multi": {1: [("leak_all", "solve", "blind"), ("none", "solve", "blind")]},
    "solvable": {1: [("leak_solvable", "solve", "target"), ("leak_solvable_wrong", "solve", "target"), ("none", "solve", "target")]},
    "forbidden2": {1: [("forbidden_leak", "solve", "blind"), ("none", "solve", "blind")]},
    "methodtip": {1: [("method_tip", "solve", "target"), ("leak_solvable", "solve", "target"), ("none", "solve", "target")]},
}


def solvable_target(seed_dir: Path) -> str | None:
    path = seed_dir/"leak_solvable"/"run.json"
    return json.loads(path.read_text())["plant"].get("solvable_target") if path.exists() else None


def run_unit(model, tok, capture, lens, run_dir: Path, out: Path, phase: str, who: str, compacted: bool) -> int:
    run = json.loads((run_dir/"run.json").read_text())
    if not run.get("complete"):
        print(f"skip {run_dir}: not complete", flush=True)
        return 0
    agents = run["results"][EPISODE]["agents"]
    target = solvable_target(run_dir.parent) if who == "target" else None
    keep = {a for a, r in agents.items() if (who == "blind" and r["blind"]) or (who == "free" and not r["instructed"]) or who == "all"
            or a == target}
    old = json.loads((out/"index.json").read_text())["turns"] if (out/"index.json").exists() else []
    done = {row["file"] for row in old}
    planted = replay.planted_answers(run_dir, EPISODE)
    entries = run["plant"].get("entries") or []
    if who == "target":       # the planted value is the note's, right or wrong; none reads where it writes the right one
        sibling = json.loads((run_dir.parent/"leak_solvable"/"run.json").read_text())["plant"]["entries"]
        planted = {target: (entries[0].get("value") if entries else None) or sibling[0]["value"]}
    rows = []
    for turn in load_turns(run_dir):
        if turn["episode_index"] != EPISODE or turn["phase"] != phase or turn["agent"] not in keep:
            continue
        path = out/f"{turn['agent']}_e{EPISODE}_{turn['phase']}.npz"
        if path.name in done and path.exists():
            continue
        mentions = {"planted": planted[turn["agent"]]} if turn["agent"] in planted else None
        extra = None
        if run["plant"].get("kind") == "leak2":
            mine = [e for e in entries if e.get("target") == turn["agent"]]
            mentions = {**(mentions or {}), **{f"value_{e['author']}": e["value"] for e in mine if e.get("value")}}
            extra = {e["author"]: e["text"] for e in mine}
        trace = (agents[turn["agent"]]["blind"] or turn["agent"] == target) and phase == "solve"
        rows.append(replay.replay_turn(model, tok, capture, turn, lens, path, True, compacted, mentions, shuffled=True, trace=trace,
                                       extra_refs=extra))
        r = rows[-1]
        print(f"{run_dir} {r['agent']} {r['phase']} tokens {r['tokens']} prefix {r['shared_prefix_tokens']} {r['seconds']}s", flush=True)
    if rows or not old:
        out.mkdir(parents=True, exist_ok=True)
        (out/"index.json").write_text(json.dumps({"run": str(run_dir), "model": replay.MODEL_NAME, "layers": capture.layers,
                                                  "turns": old + rows}, indent=1))
    return len(rows)


ARM_PLAN = {"sweep2": "sweep2", "sweep2_norule": "norule", "sweep2_leak2": "leak2", "sweep2_multi": "multi",
            "sweep2_solvable": "solvable", "sweep2_forbidden2": "forbidden2", "sweep2_methodtip": "methodtip"}
# the priority order across arms: blind solve turns first (leak2, the rule contrast, leak_all, solvable), then command verdicts
DEFAULT_ORDER = ("sweep2:1 sweep2_leak2:1 sweep2_norule:1 sweep2_multi:1 sweep2_solvable:1 sweep2:2 sweep2_norule:2 "
                 "sweep2:3 sweep2_norule:3")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_root", type=Path, help="the directory holding the experiment directories, e.g. runs")
    parser.add_argument("--out-root", type=Path, default=ROOT/"data/replay_sweep2")
    parser.add_argument("--order", default=DEFAULT_ORDER, help="space-separated exp:priority units, highest first")
    parser.add_argument("--expect", nargs="*", default=["sweep2=20", "sweep2_norule=20", "sweep2_leak2=10", "sweep2_multi=10",
                                                        "sweep2_solvable=10"], help="exp=N seeds to wait for")
    parser.add_argument("--stop-at", type=float, default=None, help="unix time after which no new unit is started")
    parser.add_argument("--compacted", action="store_true")
    args = parser.parse_args()
    from detector.jspace import Lens
    model, tok = replay.load_model()
    lens = Lens(replay.MODEL_DIR, replay.LENS_FILE)
    capture = replay.Capture(model, lens.layers)
    order = [(u.split(":")[0], int(u.split(":")[1])) for u in args.order.split()]
    expect = {e.split("=")[0]: int(e.split("=")[1]) for e in args.expect}
    finished: set[tuple[str, str, int]] = set()
    started = time.time()
    while not (args.stop_at and time.time() > args.stop_at):
        unit = None
        for exp, prio in order:
            plan = PLANS[ARM_PLAN[exp]]
            variants = {v for units in plan.values() for v, _, _ in units}
            ready = [p for p in sorted((args.runs_root/exp).glob("seed*")) if all((p/v/"run.json").exists() for v in variants)]
            todo = [p for p in ready if (exp, p.name, prio) not in finished]
            if todo:
                unit = (exp, prio, todo[0])
                break
        if unit is None:
            done = all(sum(1 for e, _, p in finished if e == exp and p == prio) >= expect.get(exp, 0) for exp, prio in order)
            if done:
                break
            time.sleep(30)
            continue
        exp, prio, seed_dir = unit
        n = sum(run_unit(model, tok, capture, lens, seed_dir/v, args.out_root/Path(*(seed_dir/v).parts[-3:]), phase, who, args.compacted)
                for v, phase, who in PLANS[ARM_PLAN[exp]][prio])
        finished.add((exp, seed_dir.name, prio))
        print(f"== {exp} P{prio} {seed_dir.name} {n} turns, {time.time() - started:.0f}s elapsed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
