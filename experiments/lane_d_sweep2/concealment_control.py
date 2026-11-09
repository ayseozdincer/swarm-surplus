#!/usr/bin/env python3
"""Lane D. Control for the concealers-against-disclosers contrast (experiments/lane_d_sweep2/sweep2_scores.py concealment): the slip-copiers are
all on long-string answers and the exact concealers on integers, and two seeds appear in two arms. Direction score again, leaving
out every user of the same seed number, overall and within answer type. Writes experiments/lane_d_sweep2/results/concealment_control.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]
from sweep2_scores import BANDS, EXPERIMENTS, REPLAY, auroc, cos, load_arm, near, unit  # noqa: E402


def main() -> int:
    users = []
    for exp in EXPERIMENTS:
        for seed_name, seed in load_arm(exp, None).items():
            for variant in ("leak", "forbidden_leak"):
                v = seed.get(variant)
                for agent, key in (v["planted"].items() if v else ()):
                    lab, rec = v["labels"][agent], v["turns"].get((agent, "solve"))
                    if rec is not None and near(lab["answer"], key):
                        users.append({"exp": exp, "seed": seed_name, "variant": variant, "concealed": not lab["disclosed_source"],
                                      "string": len(str(key)) > 6, "d": rec["d"]})
    out = {"users": len(users), "concealers": sum(u["concealed"] for u in users), "by_answer_type": {}}
    for kind, sel in (("all", users), ("string answers", [u for u in users if u["string"]]), ("integer answers", [u for u in users if not u["string"]])):
        out["by_answer_type"][kind] = {"concealers": sum(u["concealed"] for u in sel), "disclosers": sum(not u["concealed"] for u in sel)}
        for position in ("note_pre", "answer_pre"):
            for band_name in ("mid", "late", "all"):
                band = BANDS[band_name]
                rows = [u for u in sel if position in u["d"]]
                scores = []
                for u in rows:
                    c = [unit(w["d"][position][band]) for w in rows if w["seed"] != u["seed"] and w["concealed"]]
                    s = [unit(w["d"][position][band]) for w in rows if w["seed"] != u["seed"] and not w["concealed"]]
                    if c and s:
                        d = u["d"][position][band]
                        scores.append((u["concealed"], float(np.mean(cos(d, np.mean(c, axis=0))) - np.mean(cos(d, np.mean(s, axis=0))))))
                out["by_answer_type"][kind][f"{position} | {band_name}"] = auroc([x for c, x in scores if c], [x for c, x in scores if not c])
    (ROOT/"experiments/lane_d_sweep2/results/concealment_control.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
