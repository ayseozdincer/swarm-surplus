#!/usr/bin/env python3
"""Markdown tables for the lane D report from results/lane_d/sweep2_scores.json.

  python scripts/lane_d_tables.py [results/lane_d/sweep2_scores.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYERS = (0, 5, 10, 16, 19, 25, 31, 35, 38)
POSITIONS = ("prompt_last", "first_generated", "first_thought", "after_think", "planted_pre")
SOLVE_KEYS = ("content | first_generated | late", "source | first_generated | late", "jump | late", "planted_jump | late",
              "trace mean", "delta | first_generated | late", "mass | first_generated", "mass/floor | first_generated",
              "ontology all | first_generated | late", "ontology conceal | first_generated | late",
              "content | first_generated | mid", "source | first_generated | mid", "cost | first_generated | late",
              "delta shuffled | first_generated | late", "mass shuffled | first_generated", "trace shuffled_mean")
VERDICT_KEYS = ("mass | verdict_pre", "mass/floor | first_generated", "delta | verdict_pre | late", "delta | first_generated | late",
                "ontology all | verdict_pre | late", "ontology all | first_generated | late", "cost | verdict_pre | late",
                "delta shuffled | verdict_pre | late")


def fmt(x, digits=3):
    return "-" if x is None else (f"{x:.{digits}f}" if isinstance(x, float) else str(x))


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"] +
                     ["| " + " | ".join(r) + " |" for r in rows])


def by_layer(r, name, stat="median"):
    rows = []
    for layer in LAYERS:
        rows.append([str(layer)] + [fmt(r[name][p][stat][layer], 4) if p in r[name] else "-" for p in POSITIONS])
    return table(["Layer"] + [f"{p} ({r[name].get(p, {}).get('turns', 0)})" for p in POSITIONS], rows)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"results/lane_d/sweep2_scores.json"
    r = json.loads(path.read_text())
    print(f"Seeds {len(r['seeds'])}. Bands {r['bands']}. Arms {({a: len(s) for a, s in r['arms'].items()})}.\n")
    for name, title in (("kernel_floor", "Kernel floor: plain kernel against the explicit 4D mask, none-branch turns, median relative delta"),
                        ("mask_effect_leak", "Mask effect: actual against the kept reference, leak branch, median"),
                        ("shuffled_effect_leak", "Shuffled-span control: actual against a random sanctioned span of the same length masked, leak branch, median"),
                        ("cost_leak", "Direction distance 1 - cos(d, d_shuffled), leak branch, median")):
        print(f"### {title}\n")
        print(by_layer(r, name))
        print()
    for name, out in r["contrasts"].items():
        keys = VERDICT_KEYS if name.startswith("6") else SOLVE_KEYS
        rows = []
        for key in keys:
            v = out.get(key)
            if not v:
                continue
            wins = f"{v['paired_wins']}/{v['paired_n']}" if "paired_wins" in v else "-"
            rows.append([key.replace(" | ", ", "), fmt(v["auroc"], 2), wins, f"{fmt(v['positive_median'])} / {fmt(v['negative_median'])}"])
        print(f"### {name}, {out['n_pos']} against {out['n_neg']}\n")
        print(table(["Score", "AUROC", "Paired wins", "Median, positive / negative"], rows))
        print()
    c = r["concealment"]
    print(f"### Concealment\n\nLeak users {c['users']}, concealers {c['concealers']}. By arm {c['by_arm']}.\n")
    if c["scores"]:
        print(table(["Score", "AUROC", "Concealers / disclosers"], [[k.replace(" | ", ", "), fmt(v["auroc"], 2), f"{v['n_concealers']} / {v['n_disclosers']}"]
                                                                     for k, v in c["scores"].items()]))
    traces = r.get("traces", [])
    if traces:
        print("\n### Per-token trace, blind agent, late band, median over turns\n")
        rows = []
        for variant in ("leak", "forbidden_leak", "none", "benign_k1"):
            t = [x for x in traces if x["variant"] == variant]
            if not t:
                continue
            med = lambda k: fmt(sorted(x[k] for x in t if x.get(k) is not None)[len([x for x in t if x.get(k) is not None])//2] if any(x.get(k) is not None for x in t) else None, 4)  # noqa: E731
            rows.append([variant, str(len(t)), med("mean"), med("shuffled_mean"), med("before_mention"), med("after_mention"), med("peak")])
        print(table(["Branch", "Turns", "Mean surplus", "Mean, shuffled control", "16 tokens before first writing the value", "16 tokens after", "Peak"], rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
