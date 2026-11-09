"""Shared helpers for the paper tables. Standard library only, deterministic."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "tables"


def rel(p: Path) -> str:
    """Path relative to the repo root, for table headers."""
    return str(Path(p).resolve().relative_to(ROOT))


def load(p: Path):
    with open(p) as f:
        return json.load(f)


def quantile(xs, q):
    """Linear interpolation between order statistics (numpy default)."""
    s = sorted(xs)
    if not s:
        return None
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def median(xs):
    return quantile(xs, 0.5)


def summary(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0, "min": None, "q1": None, "median": None, "q3": None, "max": None}
    return {"n": len(xs), "min": min(xs), "q1": quantile(xs, 0.25), "median": median(xs),
            "q3": quantile(xs, 0.75), "max": max(xs), "p90": quantile(xs, 0.9)}


def auroc(pos, neg):
    """Mann-Whitney AUROC: P(pos > neg) + 0.5 P(tie)."""
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def paired_wins(pos_by_seed, neg_by_seed):
    """Seeds where pos > neg; ties counted separately."""
    seeds = [s for s in pos_by_seed if s in neg_by_seed]
    wins = sum(1 for s in seeds if pos_by_seed[s] > neg_by_seed[s])
    ties = sum(1 for s in seeds if pos_by_seed[s] == neg_by_seed[s])
    return wins, ties, len(seeds)


def bootstrap_auroc(pos_by_seed, neg_by_seed, n_resamples=2000, rng_seed=0):
    """Resample seeds with replacement (each seed carries its pos and neg turn),
    recompute the AUROC each time, return the 2.5 and 97.5 percentiles."""
    seeds = sorted(s for s in pos_by_seed if s in neg_by_seed)
    rng = random.Random(rng_seed)
    vals = []
    for _ in range(n_resamples):
        pick = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        a = auroc([pos_by_seed[s] for s in pick], [neg_by_seed[s] for s in pick])
        vals.append(a)
    return quantile(vals, 0.025), quantile(vals, 0.975)


def fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if abs(x - round(x)) < 1e-9 and abs(x) < 1e6:
            return str(int(round(x)))
        return f"{x:.{nd}f}"
    return str(x)


def frac(num, den):
    if den == 0:
        return "-"
    return f"{num}/{den} = {num / den:.2f}"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write(name: str, script: str, lines):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    text = f"<!-- produced by paper/build/{script}; do not edit by hand -->\n" + "\n".join(lines).rstrip() + "\n"
    p.write_text(text)
    print("wrote", rel(p))
