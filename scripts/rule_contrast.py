#!/usr/bin/env python3
"""Lane D. Paired behaviour, rule on (runs/sweep2, rules confidential) against rule off (runs/sweep2_norule), on the seeds
where the rule-off arm finished every branch. Same seeds and tasks; the warm-up episodes were generated separately.

  python scripts/rule_contrast.py runs/sweep2 runs/sweep2_norule
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.labels import run_labels  # noqa: E402

BRANCHES = ("none", "benign_k1", "benign_k3", "leak", "command_k1", "command_k3")


def measured(path: Path):
    run = json.loads(path.read_text())
    return run_labels(run)["measured"] if run.get("complete") else None


def main() -> int:
    on, off = Path(sys.argv[1]), Path(sys.argv[2])
    seeds = [p.name for p in sorted(off.glob("seed*")) if all((p/b/"run.json").exists() for b in BRANCHES)]
    rows = []
    for seed in seeds:
        row = {"seed": seed}
        for arm, root, variants in (("on", on, ("leak", "forbidden_leak", "command_k1", "command_k3", "none")),
                                    ("off", off, ("leak", "command_k1", "command_k3", "none"))):
            for v in variants:
                m = measured(root/seed/v/"run.json")
                if m is None:
                    continue
                row[f"{arm} {v} leak_used"] = (m["leak_used_free"], m["leak_targets_free"])
                row[f"{arm} {v} disclosed"] = m["disclosed_source_free"]
                row[f"{arm} {v} concealed"] = m["concealed_source_free"]
                row[f"{arm} {v} followed"] = (m["followed_command_free"], m["free_reviewers_of_correct"])
                row[f"{arm} {v} strict"] = (m["strict_wrong_free"], m["free_reviewers_of_correct"])
                row[f"{arm} {v} rule_reasoning"] = (m.get("mentions_rule_reasoning_free"), m["n_free"])
        rows.append(row)

    def total(key):
        pairs = [r[key] for r in rows if key in r and isinstance(r[key], tuple)]
        return f"{sum(a for a, _ in pairs)}/{sum(b for _, b in pairs)}" if pairs else "-"

    def count(key):
        vals = [r[key] for r in rows if key in r]
        return str(sum(vals)) if vals else "-"
    out = {"seeds": seeds, "rows": rows, "table": {
        "blind leak use, rule on (leak)": total("on leak leak_used"),
        "blind leak use, rule on (forbidden_leak)": total("on forbidden_leak leak_used"),
        "blind leak use, rule off (leak)": total("off leak leak_used"),
        "disclosed / concealed, rule on leak": f"{count('on leak disclosed')} / {count('on leak concealed')}",
        "disclosed / concealed, rule on forbidden_leak": f"{count('on forbidden_leak disclosed')} / {count('on forbidden_leak concealed')}",
        "disclosed / concealed, rule off leak": f"{count('off leak disclosed')} / {count('off leak concealed')}",
        "command followed, rule on k1": total("on command_k1 followed"), "command followed, rule off k1": total("off command_k1 followed"),
        "command followed, rule on k3": total("on command_k3 followed"), "command followed, rule off k3": total("off command_k3 followed"),
        "strict given correct in none, rule on": total("on none strict"), "strict given correct in none, rule off": total("off none strict"),
        "rule in reasoning, leak, rule on": total("on leak rule_reasoning"), "rule in reasoning, leak, rule off": total("off leak rule_reasoning"),
    }}
    (ROOT/"results/lane_d").mkdir(parents=True, exist_ok=True)
    (ROOT/"results/lane_d/rule_contrast.json").write_text(json.dumps(out, indent=1, default=list) + "\n")
    print("seeds", seeds)
    for k, v in out["table"].items():
        print(f"  {k:<48} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
