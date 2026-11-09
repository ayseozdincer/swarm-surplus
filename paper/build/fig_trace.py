"""Trace figure for the paper. Seed 5034, sweep2, Qwen3-14B, carol's solve turn in the leak and benign_k1 branches.
Source: experiments/lane_d_sweep2/results/trace_seed5034.csv on branch claude/lane-d-sweep2-affc5f (lane D), placebo column is lane C's designer-span placebo.
Delta over floor is ||h_actual - h_reference|| / ||h_actual|| divided by the kernel floor, mean over L16 to L38."""
import csv, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1]
out = sys.argv[2]
rows = list(csv.DictReader(open(src)))
by = defaultdict(list)
first = {}
for r in rows:
    b = r["branch"]
    if r["marker"] == "first_generated":
        first[b] = int(r["token_index"])
for r in rows:
    b = r["branch"]
    off = int(r["token_index"]) - first[b]
    if 0 <= off < 256:
        by[b].append((off, float(r["delta_over_floor_band_L16_38"]), float(r["placebo_over_floor_band_L16_38"] or 1.0)))

fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=200)
leak = sorted(by["leak"]); ben = sorted(by["benign_k1"])
ax.plot([o for o, _, _ in leak], [d for _, d, _ in leak], color="#c2410c", lw=1.6, label="leak branch, Frank's note gives the answer")
ax.plot([o for o, _, _ in ben], [d for _, d, _ in ben], color="#1f5f8b", lw=1.4, ls="--", label="benign branch, Alice's harmless tip")
ax.plot([o for o, _, _ in leak], [p for _, _, p in leak], color="#6b6a63", lw=1.0, ls=":", label="placebo, a designer span of the note's length masked")
ax.axhline(10, color="#999", lw=0.8, ls="-.")
ax.text(255, 10.8, "10 times the floor", ha="right", va="bottom", fontsize=7, color="#555")
ax.axvline(63, color="#c2410c", lw=0.8, ls=":")
ax.text(65, 160, "Carol first writes -50", fontsize=7, color="#c2410c")
ax.axvline(141, color="#1f5f8b", lw=0.8, ls=":")
ax.text(143, 160, "Carol discusses Alice's tip", fontsize=7, color="#1f5f8b")
ax.set_yscale("log"); ax.set_ylim(0.5, 300); ax.set_xlim(0, 255)
ax.set_xlabel("reply token, from the first generated token", fontsize=8)
ax.set_ylabel("surplus over the kernel floor, L16 to L38", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, loc="upper right", frameon=False)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(out); fig.savefig(out.replace(".png", ".pdf"))
# summary numbers for the caption
def count(v): return sum(1 for _, d, _ in v if d > 10)
print("count above 10 in first 256: leak", count(leak), "benign", count(ben), "placebo", sum(1 for _, _, p in leak if p > 10))
print("leak at mention_pre offset 63:", [d for o, d, _ in leak if o == 63])
