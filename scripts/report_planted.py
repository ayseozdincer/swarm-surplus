#!/usr/bin/env python3
"""Markdown tables for phase 2c from labels_planted_file.json and scripts/planted_scores.py's output.

  python scripts/report_planted.py runs/planted results/lane_e/planted_scores.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BRANCHES = ("none", "benign", "leak", "credential")
MEDIANS = ("delta | first_generated", "delta | after_think", "delta | answer_pre", "mass | first_generated", "mass | after_think", "mass | answer_pre",
           "first16", "before16", "after16", "plateau", "jump", "jump_delta")
HEADLINE = ("source | first_generated", "source | after_think", "source | answer_pre", "content | after_think", "delta | first_generated", "delta | after_think",
            "mass | first_generated", "mass | after_think", "first16", "after16", "plateau", "jump", "jump_delta")


def fmt(x, digits=2) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}f}" if abs(x) >= 10**-digits or x == 0 else f"{x:.1e}"
    return str(x)


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + " --- |"*len(headers)]
    out += ["| " + " | ".join(fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def median(rows: list[dict], key: str, field: str) -> tuple[float | None, int]:
    values = [r[key].get(field) for r in rows if r[key].get(field) is not None]
    return (float(np.median(values)) if values else None, len(values))


def main() -> int:
    runs, scores_path = Path(sys.argv[1]), Path(sys.argv[2])
    labels = json.loads((runs/"labels_planted_file.json").read_text())
    scores = json.loads(scores_path.read_text())
    by = labels["by_branch"]
    print("### Behaviour, target at the plant episode\n")
    print(table(["branch", "n", "read file", "saw line", "used value", "correct", "reveal called", "right token", "disclosed of correct", "concealed of correct", "accepted", "refused before reading", "median reply tokens"],
                [[b, by[b]["n"], by[b]["read_file"], by[b]["exposed_line"], by[b]["used_value"], by[b]["correct"], by[b]["reveal_called"], by[b]["reveal_token_right"],
                  by[b]["disclosed_among_correct"], by[b]["concealed_among_correct"], by[b]["accepted"], by[b]["submit_refused"], by[b]["reply_tokens_median"]] for b in BRANCHES]))
    targets = labels["targets"]
    split = []
    for address in ("name", "broadcast"):
        for b in ("leak", "credential"):
            sel = [r for r in targets if r["branch"] == b and r["address"] == address]
            split.append([b, address, len(sel), sum(bool(r["correct"]) for r in sel), sum(bool(r["reveal_token_right"]) for r in sel)])
    print("\nBy address.\n")
    print(table(["branch", "address", "n", "correct", "right token"], split))
    tokens = {b: [r["line_tokens"] for r in targets if r["branch"] == b and r["line_tokens"]] for b in BRANCHES[1:]}
    print("\nLine length, median tokens " + ", ".join(f"{b} {int(np.median(v))}" for b, v in tokens.items() if v)
          + f". Fallback lines {sum(bool(r['line_fallback']) for r in targets)}. Derived spans against the owner record {labels['mask_check']}.\n")
    print("### Declared against derived, same turns, AUROC at L16 to L38\n")
    print("`first` is the target's first call after reading the file, `last` the call that submits.\n")
    for title, rows in scores["headline"].items():
        contrast = scores["contrasts"][title]
        if not rows:
            print(f"{title}: {contrast['n_pos']} against {contrast['n_neg']}, nothing to compare.\n")
            continue
        print(f"**{title}**, {contrast['n_pos']} against {contrast['n_neg']}.\n")
        print(table(["metric", "declared", "derived"], [[k.replace(" | ", ", "), v["declared"], v["derived"]] for k, v in rows.items()]))
        print()
    print("### Medians per branch at L16 to L38, first call after the read\n")
    rows = []
    for key in MEDIANS:
        row = [key.replace(" | ", ", ")]
        for b in BRANCHES:
            sel = [r for r in scores["rows"] if r["branch"] == b]
            d, _ = median(sel, "derived first", key)
            n, _ = median(sel, "declared first", key)
            row += [d, n]
        rows.append(row)
    print(table(["metric"] + [f"{b} {ref}" for b in BRANCHES for ref in ("derived", "declared")], rows))
    print("\n### What the declared mask leaves\n")
    kernel = scores["kernel"]["declared delta, by branch"]
    print(table(["branch", "n", "median declared delta at first_generated"], [[b, kernel[b]["n"], kernel[b]["median"]] for b in BRANCHES]))
    print(f"\nMedian masked tokens under the derived mask {scores['kernel']['masked tokens median']}.\n")
    print("### Credential, the surplus at the reveal_input call\n")
    reveal = scores["reveal_call"]
    medians = reveal["medians"]
    print(f"{reveal['n']} targets called it with a token at a replayed call, {reveal['right_token']} with the right token.\n")
    print(table(["position", "derived delta", "declared delta", "derived delta over none floor", "declared delta over none floor", "derived mass", "declared mass"],
                [[p, medians.get(f"derived delta {p}"), medians.get(f"declared delta {p}"), medians.get(f"derived delta/floor {p}"), medians.get(f"declared delta/floor {p}"),
                  medians.get(f"derived mass {p}"), medians.get(f"declared mass {p}")] for p in ("tool_call_pre", "token_pre")]))
    print("\n### Lens naming at the copy, rank of the token written next (0 is the top), best over L16 to L38, median over targets\n")
    print(table(["where", "n", "actual", "derived reference", "declared reference"],
                [[k, v["n"], v["actual_median"], v["derived_median"], v["declared_median"]] for k, v in scores["naming"].items()]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
