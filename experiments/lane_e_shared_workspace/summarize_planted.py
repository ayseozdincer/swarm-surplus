#!/usr/bin/env python3
"""Print the phase 2c target labels per branch and per seed, and the derived spans against the owner record.
Rewrites the root's labels_planted_file.json from run.json.
Usage: python experiments/lane_e_shared_workspace/summarize_planted.py runs/<exp> [--notes]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from swarm.planted_file import BRANCHES, collect_planted  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--notes", action="store_true", help="also print each target's line, answer and submission note")
    args = p.parse_args()
    lab = collect_planted(Path(args.root))
    hdr = (f"{'branch':<11} {'n':>3} {'read file':>9} {'saw line':>8} {'used value':>10} {'correct':>7} {'reveal':>6} {'right token':>11} "
           f"{'disclosed':>9} {'concealed':>9} {'accepted':>8} {'refused':>7} {'reply tok':>9}")
    print(lab["experiment"])
    print(hdr)
    print("-"*len(hdr))
    for name in BRANCHES:
        v = lab["by_branch"][name]
        print(f"{name:<11} {v['n']:>3} {v['read_file']:>9} {v['exposed_line']:>8} {v['used_value']:>10} {v['correct']:>7} {v['reveal_called']:>6} "
              f"{v['reveal_token_right']:>11} {v['disclosed_among_correct']:>9} {v['concealed_among_correct']:>9} {v['accepted']:>8} "
              f"{v['submit_refused']:>7} {str(v['reply_tokens_median']):>9}")
    print(f"\ndisclosed and concealed are among correct targets. named seeds, of each branch's n: "
          f"{ {k: v['named'] for k, v in lab['by_branch'].items()} }")
    print(f"derived spans against the owner record {lab['mask_check']}")
    if lab["incomplete"]:
        print("incomplete", lab["incomplete"])
    if args.notes:
        for r in lab["targets"]:
            print(f"\nseed{r['seed']} {r['branch']} {r['target']} ({r['address']}, phrasing {r['phrasing']}), author {r['author']}, line {r['line_chars']} chars, {r['line_tokens']} tokens"
                  f"{' FALLBACK' if r['line_fallback'] else ''}")
            print(f"  line   {r['line']!r}")
            print(f"  answer {r['answer'][:80]!r}  correct {r['correct']}  source {r['source_match']!r}")
            print(f"  note   {r['note'][:240]!r}")
            print(f"  steps {r['solve_steps']}, reply tokens by step {r['reply_tokens_by_step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
