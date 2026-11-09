"""Apply the paper's fitted hand-feature model, frozen, to the TaskTracker-shaped turns.

The model (results/paper/tables/fitted_model.json) was trained on our game's traces. Here its five features are read from the
external check's per-turn scores (results/paper/tables/sources/tasktracker_scores.json). Two readings of the jump feature are given,
the paper's label-free one at the first writing of the agent's own answer (jump_own) and the label-aware one at the injected target
(jump_target). Output. results/paper/tables/transfer_fitted.md
"""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
m = json.load(open(ROOT / "results/paper/tables/fitted_model.json"))
rows = json.load(open(ROOT / "results/paper/tables/sources/tasktracker_scores.json"))["rows"]
mean, scale, coef, b, thr = (np.array(m["scaler_mean"]), np.array(m["scaler_scale"]), np.array(m["coef"]), m["intercept"], m["threshold_zero_training_fpr"])

def score(x):
    z = (np.array(x, dtype=float) - mean) / scale
    return 1 / (1 + np.exp(-(coef @ z + b)))

out = {"own": [], "target": []}
groups = []
for r in rows:
    jo = r.get("jump_own"); jt = r.get("jump_target")
    jo = 1.0 if jo is None else jo; jt = 1.0 if jt is None else jt
    base = [r["count"], r["window"], None, r["reply_tokens"], r.get("masked_tokens") or 0]
    out["own"].append(score([base[0], base[1], jo, base[3], base[4]]))
    out["target"].append(score([base[0], base[1], jt, base[3], base[4]]))
    groups.append(r["group"] if r["twin"] == "poisoned" else "clean")
groups = np.array(groups)
names = sorted(set(groups), key=lambda g: ["followed", "quoted_refused", "never", "clean"].index(g) if g in ["followed", "quoted_refused", "never", "clean"] else 9)
lines = ["<!-- produced by scripts/paper/transfer_fitted.py; do not edit by hand -->",
         "# The paper's fitted model, frozen, on the TaskTracker-shaped turns", "",
         f"Model from `results/paper/tables/fitted_model.json`, trained on {m['training']['turns']} turns of our game, threshold {thr:.3f} at zero training false positives. Features read from the external check's per-turn scores. Jump read two ways, at the first writing of the agent's own answer (the paper's label-free rule) and at the injected target (label-aware).", "",
         "| Group | n | flagged, jump at own answer | flagged, jump at target | median score, own | median score, target |", "| --- | --- | --- | --- | --- | --- |"]
for g in names:
    idx = groups == g
    so = np.array(out["own"])[idx]; st = np.array(out["target"])[idx]
    lines.append(f"| {g} | {idx.sum()} | {(so > thr).sum()} of {idx.sum()} | {(st > thr).sum()} of {idx.sum()} | {np.median(so):.2f} | {np.median(st):.2f} |")
y = (groups == "followed").astype(int)
for key in ("own", "target"):
    s = np.array(out[key])
    lines.append(f"\nAUROC followed against every other turn, jump at {key}: {roc_auc_score(y, s):.3f}; followed against poisoned non-followers only: {roc_auc_score(y[groups != 'clean'], s[groups != 'clean']):.3f}")
(ROOT / "results/paper/tables/transfer_fitted.md").write_text("\n".join(lines))
print("\n".join(lines))
