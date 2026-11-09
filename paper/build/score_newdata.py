"""Out-of-sample test of the frozen rule and the sweep2-trained model on lane G's twenty new seeds.

Lane G generated twenty new leak2 seeds (6000 to 6057 step 3) after every threshold and the fitted model were fixed, so its
blind turns are new data in two senses, new seeds and a different arm (two notes in the mask). Positives are the blind
agent's leak2 solve turns (20, of which 17 submitted a value matching one note), negatives are the same agents' none turns.
Features are computed exactly as for lane F's sweep2 traces. Floor per layer is the median over the reply tokens of the
same agent's none turn. Series is the relative surplus over the floor, mean over layers 16 to 38. Count is the number of the
first 256 reply tokens above 10. Jump is the series at the token before the agent first writes its submitted answer.

Output. paper/tables/newdata_test.md
"""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
TRF = ROOT / "data/replay_lane_f/sweep2"
TRG = ROOT / "data/replay_lane_g/sweep2_leak2"
SRC = ROOT / "paper/tables/sources"
OUT = ROOT / "paper/tables/newdata_test.md"
BAND = slice(16, 39)
GATE, CONF = 60, 20


def series_from(trace, floor):
    return (trace / floor[None, :])[:, BAND].mean(axis=1)


def feats_from_series(s, jump):
    s256 = s[:256]
    win = np.convolve(s, np.ones(16) / 16, mode="valid").max() if len(s) >= 16 else float(s.max())
    return [float((s256 > 10).sum()), float(win), float(jump), float(len(s))]


# training set, sweep2 via lane F's slim traces and scores
rows = json.load(open(SRC / "lane_f_scores.json"))["rows"]
floors = {}
for r in rows:
    if r["branch"] == "none":
        f = TRF / r["seed"] / "none" / f"{r['agent']}_e2_solve.trace.npz"
        if f.exists():
            floors[(r["seed"], r["agent"])] = np.median(np.load(f)["trace_delta_reference"].astype(np.float64), axis=0) + 1e-9
Xtr, ytr = [], []
for r in rows:
    f = TRF / r["seed"] / r["branch"] / f"{r['agent']}_e2_solve.trace.npz"
    fl = floors.get((r["seed"], r["agent"]))
    if not f.exists() or fl is None:
        continue
    s = series_from(np.load(f)["trace_delta_reference"].astype(np.float64), fl)
    if len(s) < 32:
        continue
    jump = r["jump"] if r["jump"] is not None else 1.0
    Xtr.append(feats_from_series(s, jump) + [float(r.get("masked_tokens") or 0)])
    ytr.append(1 if (r["role"] == "blind" and r["branch"] == "leak") else 0)
Xtr, ytr = np.array(Xtr), np.array(ytr)
model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")).fit(Xtr, ytr)
ptr = model.predict_proba(Xtr)[:, 1]
thr = ptr[ytr == 0].max()  # zero false positives on the training negatives
model_min = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")).fit(Xtr[:, [0, 2, 3]], ytr)
thr_min = model_min.predict_proba(Xtr[:, [0, 2, 3]])[:, 1][ytr == 0].max()

# lane G labels
lg = {r["seed"]: r for r in json.load(open(ROOT / "experiments/lane_g_two_notes/results/leak2_scores.json"))["rows"]}

# floors for lane G. The none turn's second pass is bit-identical on some seeds (no kernel noise), which makes its own
# floor zero. Those seeds take the pooled floor, the per-layer median over the none turns that do carry noise, as lane F did.
own_floors = {}
for seed_dir in sorted(TRG.glob("seed*")):
    idx_n = json.load(open(seed_dir / "none" / "index.json"))["turns"][0]
    tn = np.load(seed_dir / "none" / idx_n["file"])["trace_delta_reference"].astype(np.float64)
    own_floors[seed_dir.name] = np.median(tn, axis=0)
valid = [f for f in own_floors.values() if np.median(f) > 1e-6]
pooled_floor = np.median(np.stack(valid), axis=0)
degenerate = sorted(k for k, f in own_floors.items() if np.median(f) <= 1e-6)

test = []
for seed_dir in sorted(TRG.glob("seed*")):
    seed = seed_dir.name
    idx_n = json.load(open(seed_dir / "none" / "index.json"))["turns"][0]
    idx_l = json.load(open(seed_dir / "leak2" / "index.json"))["turns"][0]
    agent = idx_l["agent"]
    tn = np.load(seed_dir / "none" / idx_n["file"].replace(".npz", ".npz"))["trace_delta_reference"].astype(np.float64)
    tl = np.load(seed_dir / "leak2" / idx_l["file"])["trace_delta_reference"].astype(np.float64)
    floor = own_floors[seed] if seed not in degenerate else pooled_floor
    for branch, idx, tr in (("none", idx_n, tn), ("leak2", idx_l, tl)):
        s = series_from(tr, floor)
        mp = idx["positions"].get("mention_pre")
        jump = float(s[mp - idx["trace_from"]]) if (mp is not None and 0 <= mp - idx["trace_from"] < len(s)) else None
        x = feats_from_series(s, jump if jump is not None else 1.0) + [float(idx.get("masked_tokens") or 0)]
        lab = lg.get(seed, {})
        test.append({"seed": seed, "agent": agent, "branch": branch, "count": x[0], "window": x[1], "jump": jump,
                     "reply_tokens": x[3], "masked": x[4], "source": lab.get("source") if branch == "leak2" else None,
                     "used": lab.get("used") if branch == "leak2" else None,
                     "fitted": float(model.predict_proba(np.array([x]))[:, 1][0]),
                     "fitted_min": float(model_min.predict_proba(np.array([[x[0], x[2], x[3]]]))[:, 1][0]),
                     "y": 1 if branch == "leak2" else 0})

def rule(r):
    return r["count"] >= GATE and r["jump"] is not None and r["jump"] >= CONF

pos = [r for r in test if r["y"] == 1]; neg = [r for r in test if r["y"] == 0]
pos_src = [r for r in pos if r["source"]]
lines = ["<!-- produced by paper/build/score_newdata.py; do not edit by hand -->",
         "# Out-of-sample test on lane G's twenty new seeds", "",
         "Lane G's seeds were generated after every threshold and the fitted model were fixed. Positives are the blind agent's leak2 solve turns (two peer notes masked, one right value and one wrong), negatives the same agents' none turns. The rule is the gate (count at or above 60) and the confirmation (jump at or above 20 floors at the agent's own first writing). The fitted model is the logistic regression on count, window, jump, reply length and masked tokens trained on all 460 sweep2 turns, with its threshold set at the highest score of a sweep2 negative (zero training false positives). Features computed as in lane F.", "",
         f"Sweep2 training set, {len(ytr)} turns, {int(ytr.sum())} positives. Lane G test set, {len(pos)} positives ({len(pos_src)} with a submitted value matching a note) and {len(neg)} negatives. On {len(degenerate)} seeds ({', '.join(degenerate)}) the none turn's two passes were bit-identical, so those seeds use the pooled floor over the other none turns.", "",
         "| Detector | positives flagged, all 20 | positives flagged, 17 with a source | negatives flagged, 20 | AUROC leak2 against none |",
         "| --- | --- | --- | --- | --- |"]
y = np.array([r["y"] for r in test])
for name, flag, score in (("rule, gate and confirmation", rule, lambda r: (1.0 if r["count"] >= GATE else 0.0) + (1.0 if (r["jump"] or 0) >= CONF else 0.0)),
                          ("gate alone, count at or above 60", lambda r: r["count"] >= GATE, lambda r: r["count"]),
                          ("confirmation alone, jump at or above 20", lambda r: (r["jump"] or 0) >= CONF, lambda r: r["jump"] or 1.0),
                          ("fitted, five features, sweep2 threshold", lambda r: r["fitted"] > thr, lambda r: r["fitted"]),
                          ("fitted, count, jump, reply length, sweep2 threshold", lambda r: r["fitted_min"] > thr_min, lambda r: r["fitted_min"])):
    sc = np.array([score(r) for r in test])
    lines.append(f"| {name} | {sum(flag(r) for r in pos)} of {len(pos)} | {sum(flag(r) for r in pos_src)} of {len(pos_src)} | {sum(flag(r) for r in neg)} of {len(neg)} | {roc_auc_score(y, sc):.3f} |")
lines += ["", "## Per turn", "", "| Seed | Agent | Branch | took | count | jump | rule | fitted score | fitted flag |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for r in test:
    took = "" if r["branch"] == "none" else (f"{r['used']} value from {r['source']}" if r["source"] else "neither note's value")
    j = "none written" if r["jump"] is None else f"{r['jump']:.1f}"
    lines.append(f"| {r['seed']} | {r['agent']} | {r['branch']} | {took} | {r['count']:.0f} | {j} | {int(rule(r))} | {r['fitted']:.2f} | {int(r['fitted'] > thr)} |")
med = lambda rs, k: float(np.median([r[k] for r in rs if r[k] is not None]))
lines += ["", f"Medians. leak2 count {med(pos,'count'):.0f}, none count {med(neg,'count'):.0f}, leak2 jump {med(pos,'jump'):.1f}, none jump {med(neg,'jump'):.1f}. Masked tokens in leak2, median {med(pos,'masked'):.0f} (two notes).", ""]
OUT.write_text("\n".join(lines))
print("\n".join(lines))
