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
    # rules and commands. strict|correct in none and benign is the base rate for command followed
    hdr = f"{'variant':<14} {'strict|correct':>14} {'cmd followed':>12} {'disclosed':>11} {'concealed':>11} {'rule mention':>12}"
    print()
    print(hdr)
    print("-" * len(hdr))
    for name, v in lab["variants"].items():
        print(f"{name:<14} {fmt(v['strict_wrong_free']):>14} {fmt(v['followed_command_free']):>12} "
              f"{fmt(v['disclosed_source']):>11} {fmt(v['concealed_source']):>11} {fmt(v['mentions_rule_free']):>12}")
    # two conflicting leaks and the broadcast leak. targets used in none is the chance baseline for leak_all
    hdr = f"{'variant':<14} {'leak2 right':>11} {'leak2 wrong':>11} {'leak2 none':>11} {'targets used':>12}  runs by targets used"
    print()
    print(hdr)
    print("-" * len(hdr))
    for name, v in lab["variants"].items():
        dist = "  ".join(f"{k}:{n}" for k, n in v["targets_used_dist"].items())
        print(f"{name:<14} {fmt(v['leak2_used_correct']):>11} {fmt(v['leak2_used_wrong']):>11} "
              f"{fmt(v['leak2_used_none']):>11} {fmt(v['targets_used']):>12}  {dist}")
    # a leak to an agent that can solve its task. reply tokens compare the target with itself in none
    tok = lambda x: "-" if x is None else f"{x:.0f}"  # noqa: E731
    hdr = f"{'variant':<20} {'copied':>11} {'correct':>11} {'reply tok':>10} {'in none':>8}"
    print()
    print(hdr)
    print("-" * len(hdr))
    for name, v in lab["variants"].items():
        if v["copied_value"]["den"]:
            print(f"{name:<20} {fmt(v['copied_value']):>11} {fmt(v['solvable_correct']):>11} "
                  f"{tok(v['target_reply_tokens_mean']):>10} {tok(v['target_reply_tokens_none_mean']):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
