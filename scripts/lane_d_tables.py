#!/usr/bin/env python3
"""Markdown tables for the lane D report from results/lane_d/sweep2_scores.json.

  python scripts/lane_d_tables.py [results/lane_d/sweep2_scores.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOOR_LAYERS = (0, 5, 10, 16, 19, 25, 31, 35, 38)
FLOOR_POSITIONS = ("prompt_last", "first_generated", "first_thought", "after_think")
SCORE_KEYS = ("mass/floor | first_generated", "content | first_generated | late", "source | first_generated | late",
              "ontology | first_generated | late", "delta | first_generated | late", "jump | late", "planted_jump | late",
              "mass | planted_pre", "ontology | planted_pre | late", "naming rank (lower is named)",
              "content | first_generated | mid", "source | first_generated | mid", "ontology | first_generated | all")
VERDICT_KEYS = ("mass | first_generated", "mass | verdict_pre", "delta | first_generated | late", "delta | verdict_pre | late",
                "ontology | first_generated | late", "ontology | verdict_pre | late", "ontology | verdict_pre | all")


def fmt(x, digits=3):
    return "-" if x is None else (f"{x:.{digits}f}" if isinstance(x, float) else str(x))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"results/lane_d/sweep2_scores.json"
    r = json.loads(path.read_text())
    print(f"Seeds {len(r['seeds'])}. Bands {r['bands']}. Primary {r['primary']}.\n")
    print("### Kernel floor, relative residual delta, plain kernel against the explicit 4D mask, median / p95\n")
    floor, effect = r["kernel_floor"], r["mask_effect_leak"]
    rows = []
    for layer in FLOOR_LAYERS:
        row = [str(layer)]
        for p in FLOOR_POSITIONS:
            f = floor.get(p)
            row.append(f"{f['median'][layer]:.4f} / {f['p95'][layer]:.4f}" if f else "-")
        rows.append(row)
    print(table(["Layer"] + [f"{p} ({floor.get(p, {}).get('turns', 0)} turns)" for p in FLOOR_POSITIONS], rows))
    print("\n### Mask effect on the leak branch, median relative delta, kept / compacted\n")
    rows = []
    for layer in FLOOR_LAYERS:
        row = [str(layer)]
        for p in FLOOR_POSITIONS + ("planted_pre",):
            k, c = effect["reference"].get(p), effect["compacted"].get(p)
            row.append(f"{k['median'][layer]:.4f} / {c['median'][layer]:.4f}" if k and c else "-")
        rows.append(row)
    print(table(["Layer"] + list(FLOOR_POSITIONS + ("planted_pre",)), rows))
    for name in r["contrasts"]["reference"]:
        kept, comp = r["contrasts"]["reference"][name], r["contrasts"]["compacted"].get(name, {})
        keys = VERDICT_KEYS if name.startswith("6") else SCORE_KEYS
        print(f"\n### {name}, {kept['n_pos']} against {kept['n_neg']}\n")
        rows = []
        for key in keys:
            k, c = kept.get(key), comp.get(key)
            if not k:
                continue
            wins = f"{k['paired_wins']}/{k['paired_n']}" if "paired_wins" in k else "-"
            rows.append([key, fmt(k["auroc"], 2), fmt(c["auroc"], 2) if c else "-", wins,
                         f"{fmt(k['positive_median'])} / {fmt(k['negative_median'])}"])
        print(table(["Score", "AUROC kept", "AUROC compacted", "Paired wins, kept", "Median positive / negative, kept"], rows))
    print("\n### First mention of the withheld value, blind agent\n")
    rows = []
    for m in sorted(r["first_mention"], key=lambda m: (m["variant"], m["seed"])):
        if m["variant"] not in ("leak", "forbidden_leak", "none", "benign_k1"):
            continue
        n = m.get("naming") or {}
        rows.append([m["seed"], m["variant"], m["agent"], str(m["leak_used"]), str(m["writes_value"]),
                     ",".join(m.get("rule_phases") or []) or "-", fmt(m.get("planted_jump | late"), 1),
                     f"{n.get('actual', '-')} / {n.get('reference', '-')}" if n else "-"])
    print(table(["Seed", "Branch", "Agent", "Used", "Writes the value", "Rule in", "Jump / floor, late", "Best rank late, actual / ref"], rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
