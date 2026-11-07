#!/usr/bin/env python3
"""Print collusive rates per setting under a sweep root. Recomputes labels.json."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.labels import collect  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root")
    args = p.parse_args()
    root = Path(args.root)
    settings = sorted(d for d in root.iterdir() if d.is_dir() and d.name != "tasks")
    hdr = f"{'setting':<32} {'runs':>7} {'collusive':>9} {'mutual':>7} {'len|wrong':>9} {'strict':>7} {'sub_acc':>7} {'accept':>7} {'forced':>7} {'reward':>7}"
    print(hdr)
    print("-" * len(hdr))
    fmt = lambda v: "    -" if v is None else f"{v:5.2f}"  # noqa: E731
    for d in settings:
        lab = collect(d)
        print(f"{lab['setting']:<32} {lab['n_complete']:>3}/{lab['n_runs']:<3} {fmt(lab['collusive_rate']):>9} {fmt(lab['mutual_rate']):>7} "
              f"{fmt(lab['lenient_given_wrong_submission']):>9} {fmt(lab['strict_wrong_rate']):>7} {fmt(lab['submission_accuracy']):>7} "
              f"{fmt(lab['acceptance_rate']):>7} {fmt(lab['forced_rate']):>7} {fmt(lab['mean_reward_per_agent_episode']):>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
