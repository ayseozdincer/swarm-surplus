#!/usr/bin/env python3
"""Print the phase 2b (shared workspace) audit for an experiment root. Rewrites each
run's audit.json and the root's labels_shared_file.json from run.json.
Usage: python experiments/lane_e_shared_workspace/summarize_shared.py runs/<exp> [--edges]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from swarm.audit import collect_shared  # noqa: E402


def ep(x) -> str:
    return "-" if x is None else str(x + 1)   # episodes are printed 1-based


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--edges", action="store_true", help="also print every answer-carrying read")
    args = p.parse_args()
    root = Path(args.root)
    lab = collect_shared(root)
    print(f"{lab['experiment']}  runs {lab['n_complete']}/{lab['n_runs']} complete. Episodes are 1-based.")
    hdr = (f"{'run':<10} {'ws calls':>8} {'first call':>10} {'first write':>11} {'first comm':>10} "
           f"{'first carried':>13} {'blind ok':>9} {'via channel':>11} {'carried':>7} {'derived P/R':>12}")
    print(hdr)
    print("-" * len(hdr))
    for r in lab["runs"]:
        name = Path(r["manifest"]).parent.name if "/" in str(r["manifest"]) else r["manifest"]
        if r.get("error"):
            print(f"{name:<10} error {r['error']}")
            continue
        mc = r["mask_check"]
        pr = "-" if not mc.get("reads") else f"{mc['edge_precision'] if mc['edge_precision'] is not None else '-'}/" \
                                             f"{mc['edge_recall'] if mc['edge_recall'] is not None else '-'}"
        done = "" if r["complete"] else " (incomplete)"
        print(f"{name:<10} {r['n_tool_calls']:>8} {ep(r['first_call_episode']):>10} {ep(r['first_write_episode']):>11} "
              f"{ep(r['first_comm_episode']):>10} {ep(r['first_carried_episode']):>13} "
              f"{str(r['blind_correct']) + '/' + str(r['blind']):>9} {str(r['blind_correct_via_channel']) + '/' + str(r['blind']):>11} "
              f"{r['carried_before_submit']:>7} {pr:>12}{done}")
    print()
    print(f"runs with any workspace call {lab['runs_with_any_call']}, any write {lab['runs_with_any_write']}, "
          f"a read of another agent's bytes {lab['runs_with_comm']}, an answer or withheld piece carried before "
          f"submission {lab['runs_with_carried']}")
    print(f"blind correct {lab['blind_correct']}/{lab['blind']}, via write-then-read {lab['blind_correct_via_channel']}"
          f"/{lab['blind']}, carried before submission {lab['carried_before_submit']}, write-then-read "
          f"{lab['write_then_read']}")
    mc = lab["mask_check"]
    print(f"derived mask over {mc['reads']} reads, edges tp {mc['edge_tp']} fp {mc['edge_fp']} fn {mc['edge_fn']}, "
          f"precision {mc['edge_precision']} recall {mc['edge_recall']}")
    print("\naudited graph, writer -> reader: read episodes [answer episodes]")
    for r in lab["runs"]:
        if r.get("error") or not r.get("graph"):
            continue
        name = Path(r["manifest"]).parent.name
        edges = ", ".join(f"{g['writer']}->{g['reader']} {[e + 1 for e in g['read_episodes']]}"
                          + (f" [{', '.join(str(e + 1) for e in g['answer_episodes'])}]" if g["answer_episodes"] else "")
                          for g in r["graph"])
        print(f"  {name} {edges}")
    if args.edges:
        print("\nanswer-carrying reads")
        for d in sorted(root.glob("seed*/shared_file/audit.json")):
            aud = json.load(open(d))
            for x in aud["answer_edges"]:
                print(f"  {d.parent.parent.name} ep {x['read_episode'] + 1} {x['writer']}->{x['reader']} target "
                      f"{x['target']} ep {x['target_episode'] + 1} {x['kind']} before={x['before_target_submit']} "
                      f"{x['path']!r} {x['evidence'][:100]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
