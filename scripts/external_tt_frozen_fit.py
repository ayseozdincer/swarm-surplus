#!/usr/bin/env python3
"""The paper's fitted hand-feature model, frozen, applied to the TaskTracker-shaped turns.

Parameters from results/paper/tables/fitted_model.json on overnight-detector (commit 2228f1f), copied to results/external/sources/.
Features on our turns with our pooled floor (scripts/external_tt_scores.py rows). Jump is the series at the token before the first
writing of the injected target string, 1.0 when the reply never writes it (the paper session's instruction for this set); the
label-free variant at the answer line is reported beside it. Nothing is refit. Flag when sigmoid(coef . z + intercept) exceeds the
threshold that gave zero false positives on the training set.

  python scripts/external_tt_frozen_fit.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/"results/external/sources/fitted_model.json"
SCORES = ROOT/"results/external/tasktracker_scores.json"
OUT = ROOT/"results/external/frozen_fit.md"


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5*(pos[:, None] == neg[None, :]).mean()) if len(pos) and len(neg) else float("nan")


def main() -> int:
    m = json.load(open(MODEL))
    mean, scale, coef, b, thr = (np.array(m["scaler_mean"]), np.array(m["scaler_scale"]), np.array(m["coef"]), m["intercept"], m["threshold_zero_training_fpr"])
    rows = json.load(open(SCORES))["rows"]

    def features(r, jump_key):
        j = r[jump_key]
        return np.array([r["count"], r["window"], 1.0 if j is None else j, r["reply_tokens"], r["masked_tokens"]], float)

    def apply(x):
        z = (x - mean)/scale
        return 1/(1 + np.exp(-(coef @ z + b)))

    groups = ["followed", "not followed, written in thinking", "not followed, never written", "clean"]
    L = ["<!-- produced by scripts/external_tt_frozen_fit.py; do not edit by hand -->",
         "# The paper's fitted model, frozen, on the TaskTracker-shaped turns", "",
         f"Model `results/paper/tables/fitted_model.json` (overnight-detector 2228f1f), trained on {m['training']['turns']} game turns with {m['training']['positives']} positives over {m['training']['seeds']} seeds. "
         f"Features {', '.join(m['features'])}, standardised with the training scaler, logit coef dot z plus intercept, flag when the sigmoid exceeds {thr:.4f} (zero training false positives). "
         "Applied untouched to the 600 turns with our pooled floor. Jump is read at the first writing of the injected target, 1.0 where never written, as the paper session asked; the label-free reading at the answer line is beside it.", ""]
    summary = {}
    for jump_key, name in (("jump_target", "jump at the target, 1.0 where never written"), ("jump_final", "jump at the answer line (label-free)")):
        p = {r["sample_id"]: float(apply(features(r, jump_key))) for r in rows}
        by = defaultdict(list)
        for r in rows:
            by[r["group"]].append(p[r["sample_id"]])
        pos = by["followed"]; negp = by["not followed, written in thinking"] + by["not followed, never written"]; negc = by["clean"]
        L += [f"## {name}", "", "| Group | n | flagged at the training threshold | median score | score 90th percentile |", "| --- | --- | --- | --- | --- |"]
        for g in groups:
            v = np.array(by[g])
            L.append(f"| {g} | {len(v)} | {int((v > thr).sum())} of {len(v)} | {np.median(v):.3f} | {np.percentile(v, 90):.3f} |")
        a1, a2, a3 = auroc(pos, negp), auroc(pos, negc), auroc(pos, negp + negc)
        L += ["", f"AUROC followed against not followed (all poisoned negatives) {a1:.3f}, against clean {a2:.3f}, against every negative {a3:.3f}. "
              f"Flagged {int((np.array(pos) > thr).sum())} of {len(pos)} followed, {int((np.array(negp) > thr).sum())} of {len(negp)} poisoned negatives, {int((np.array(negc) > thr).sum())} of {len(negc)} clean.", ""]
        summary[jump_key] = {"auroc_vs_poisoned_neg": a1, "auroc_vs_clean": a2, "auroc_vs_all_neg": a3,
                             "flagged": {g: [int((np.array(by[g]) > thr).sum()), len(by[g])] for g in groups}, "scores": p}
    # standardised feature means per group, to see which feature moves the score
    L += ["## Standardised features by group (z against the training scaler), medians", "", "| Group | " + " | ".join(m["features"]) + " |", "| --- | " + " | ".join("---" for _ in m["features"]) + " |"]
    for g in groups:
        Z = np.array([(features(r, "jump_target") - mean)/scale for r in rows if r["group"] == g])
        L.append(f"| {g} | " + " | ".join(f"{v:+.2f}" for v in np.median(Z, axis=0)) + " |")
    L += ["", "Coefficients " + ", ".join(f"{f} {c:+.2f}" for f, c in zip(m["features"], coef)) + f", intercept {b:+.2f}.", ""]
    OUT.write_text("\n".join(L) + "\n")
    json.dump(summary, open(ROOT/"results/external/frozen_fit_scores.json", "w"), indent=1)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
