#!/usr/bin/env python3
"""Print measured-episode compliance per variant for an experiment root."""
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
    lab = collect(Path(args.root))
    fmt = lambda a: "    -    " if a["rate"] is None else f"{a['rate']:4.2f} {a['num']:>2}/{a['den']:<2}"  # noqa: E731
    hdr = f"{'variant':<14} {'runs':>6} {'lenient|wrong':>13} {'leak used':>11} {'blind ok':>11} {'wrong subs':>12} {'exposed':>11} {'post len|wrong':>15}"
    print(f"{lab['experiment']}  seeds {lab['n_seeds']}")
    print(hdr)
    print("-" * len(hdr))
    for name, v in lab["variants"].items():
        post = v.get("post_lenient_wrong_free")
        print(f"{name:<14} {v['n_complete']:>2}/{v['n_runs']:<3} {fmt(v['lenient_wrong_free']):>13} {fmt(v['leak_used_free']):>11} "
              f"{fmt(v['blind_correct_free']):>11} {fmt(v['wrong_submission_rate']):>12} {fmt(v['exposed_free']):>11} "
              f"{(fmt(post) if post else '    -    '):>15}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
