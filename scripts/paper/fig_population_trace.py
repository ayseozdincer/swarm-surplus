"""Figure 2 for the paper. The surplus over the floor along the first 256 reply tokens of the blind agent's solve turn, median and
interquartile band over the 20 sweep2 seeds, leak against the matched harmless tip against the matched none turn, Qwen3-14B.
Definitions are lane F's (scripts/lane_f_scores.py on branch claude/lane-f-fpr-96ebe2). delta is the kept reference's relative residual
difference per token and layer (trace_delta_reference in the slim traces). The floor per layer is the median over the reply tokens of
the same agent's none turn in the same seed. The series is delta over floor, mean over layers 16 to 38. Ticks mark the token where each
leak agent first writes the planted value (positions.planted_pre in index.json). Data, the slim traces lane F left under
data/replay_lane_f/sweep2 in the lane-f worktree (git-ignored, 400 MB), replayed from runs/sweep2 on branch claude/lane-d-sweep2-affc5f.

  .venv/bin/python scripts/paper/fig_population_trace.py <replay_lane_f dir> results/paper/figures/fig2_population_trace.png
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

replay, out = Path(sys.argv[1]) / "sweep2", sys.argv[2]
BAND, FIRST, FACTOR = slice(16, 39), 256, 10
SANCTION, PEER, GREY = "#1f5f8b", "#c2410c", "#6b6a63"

def load_turn(seed_dir: Path, branch: str):
    idx = json.loads((seed_dir / branch / "index.json").read_text())
    for t in idx["turns"]:
        if t["phase"] == "solve" and t.get("blind"):
            z = np.load(seed_dir / branch / Path(t["file"]).with_suffix(".trace.npz").name)
            return t, z["trace_delta_reference"]
    return None, None

series = {b: [] for b in ("leak", "benign_k1", "none")}
rug, counts = [], {b: [] for b in series}
for seed_dir in sorted(replay.glob("seed*")):
    tn, dn = load_turn(seed_dir, "none")
    if dn is None:
        continue
    floor = np.median(dn, axis=0)
    for b in series:
        t, d = load_turn(seed_dir, b)
        if d is None:
            continue
        s = (d / floor[None, :])[:, BAND].mean(axis=1)
        s = s[:FIRST]
        padded = np.full(FIRST, np.nan); padded[:len(s)] = s
        series[b].append(padded); counts[b].append(int((s > FACTOR).sum()))
        if b == "leak":
            p = t["positions"].get("planted_pre")
            if p is not None and 0 <= p - t["trace_from"] < FIRST:
                rug.append(p - t["trace_from"])

fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200)
x = np.arange(FIRST)
for b, col, ls, label in (("none", GREY, ":", "no note, the matched none turn"),
                          ("benign_k1", SANCTION, "--", "a harmless tip from a teammate"),
                          ("leak", PEER, "-", "a peer's note with the withheld answer")):
    arr = np.array(series[b])
    med = np.nanmedian(arr, axis=0); q1 = np.nanpercentile(arr, 25, axis=0); q3 = np.nanpercentile(arr, 75, axis=0)
    ax.fill_between(x, np.maximum(q1, 0.3), q3, color=col, alpha=0.15, lw=0)
    ax.plot(x, med, color=col, lw=1.6, ls=ls, label=f"{label} (n={len(arr)})")
for r in rug:
    ax.plot([r, r], [0.33, 0.45], color=PEER, lw=0.8, alpha=0.8)
ax.text(254, 0.36, "ticks, first writing of the value, one per leak turn", fontsize=6.5, color=PEER, va="bottom", ha="right")
ax.axhline(FACTOR, color="#999", lw=0.8, ls="-."); ax.text(255, 10.8, "10 times the floor", ha="right", va="bottom", fontsize=7, color="#555")
ax.set_yscale("log"); ax.set_ylim(0.3, 200); ax.set_xlim(0, 255)
ax.set_xlabel("reply token, from the first generated token", fontsize=8)
ax.set_ylabel("surplus over the floor, L16 to L38", fontsize=8)
ax.tick_params(labelsize=7); ax.legend(fontsize=7, loc="upper right", frameon=False)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(out); fig.savefig(out.replace(".png", ".pdf"))
print("wrote", out)
for b in counts:
    print(f"{b}: n={len(counts[b])} count median {np.median(counts[b]):.1f}, at or above 60 {sum(c >= 60 for c in counts[b])}")
print("first-writing ticks:", len(rug), "median token", np.median(rug) if rug else None)
