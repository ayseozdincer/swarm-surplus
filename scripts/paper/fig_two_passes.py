"""Figure 1 for the paper. A schematic of the two passes and the mask, with the real per-token surplus of seed 5034 below.
Top, the token sequence of one turn with its provenance tags and the forced reply. Middle, the two attention masks as small grids,
the actual pass causal everywhere, the reference pass with every peer key column blocked for every query. Bottom, the surplus over the
floor along the first 256 reply tokens of seed 5034's leak branch (results/lane_d/trace_seed5034.csv on branch claude/lane-d-sweep2-affc5f,
column delta_over_floor_band_L16_38), with the first writing of the value marked. No other data.

  .venv/bin/python scripts/paper/fig_two_passes.py <trace_seed5034.csv> results/paper/figures/fig1_two_passes.png
"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

src, out = sys.argv[1], sys.argv[2]
SANCTION, PEER, SELF, GREY = "#1f5f8b", "#c2410c", "#6b6a63", "#dcd8cf"

fig = plt.figure(figsize=(7.2, 5.2), dpi=200)
gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.6, 1.9], hspace=0.55, wspace=0.25)

# --- top: the token sequence with provenance tags ---
ax = fig.add_subplot(gs[0, :])
segments = [("designer prompt", 0, 20, SANCTION), ("own earlier turns", 20, 38, SELF), ("peer note", 38, 47, PEER),
            ("task, tool results", 47, 64, SANCTION), ("reply, forced in both passes", 64, 100, SELF)]
for label, a, b, c in segments:
    ax.add_patch(Rectangle((a, 0), b - a, 1, facecolor=c, alpha=0.25 if c != PEER else 0.45, edgecolor=c, lw=1.0,
                           hatch="////" if c == PEER else None))
    ax.text((a + b) / 2, 0.5, label, ha="center", va="center", fontsize=6.5 if c == PEER else 7, color="#1c1b18")
ax.set_xlim(0, 100); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("one turn as served, every span tagged by the orchestrator", fontsize=8, loc="left")

# --- middle: the two masks ---
N = 26
peer_cols = range(9, 13)
reply_from = 17
def draw_mask(ax, blocked, title):
    for q in range(N):
        for k in range(N):
            if k > q:
                continue
            on = not (blocked and k in peer_cols)
            col = PEER if k in peer_cols else (SELF if q >= reply_from else SANCTION)
            ax.add_patch(Rectangle((k, N - 1 - q), 1, 1, facecolor=col if on else "white", alpha=0.35 if on else 1.0,
                                   edgecolor="#eeeae2", lw=0.3))
            if not on:
                ax.add_patch(Rectangle((k, N - 1 - q), 1, 1, facecolor="none", edgecolor=PEER, lw=0.3, hatch="xx"))
    ax.set_xlim(0, N); ax.set_ylim(0, N); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=8, loc="left")
    ax.text(N / 2, -1.2, "keys, the context and reply", ha="center", va="top", fontsize=6.5, color="#605d55")
    ax.text(-1.0, N / 2, "queries", ha="right", va="center", rotation=90, fontsize=6.5, color="#605d55")
    ax.axhline(N - reply_from, color="#605d55", lw=0.5, ls=":")
    ax.text(N + 0.4, N - reply_from, "reply", fontsize=6, va="center", color="#605d55")
axa = fig.add_subplot(gs[1, 0]); draw_mask(axa, False, "actual pass, ordinary causal mask")
axb = fig.add_subplot(gs[1, 1]); draw_mask(axb, True, "reference pass, peer keys blocked for every query")

# --- bottom: the real surplus along the reply, seed 5034 leak branch ---
rows = list(csv.DictReader(open(src)))
first = {r["branch"]: int(r["token_index"]) for r in rows if r["marker"] == "first_generated"}
leak = sorted((int(r["token_index"]) - first["leak"], float(r["delta_over_floor_band_L16_38"]))
              for r in rows if r["branch"] == "leak" and 0 <= int(r["token_index"]) - first["leak"] < 256)
axc = fig.add_subplot(gs[2, :])
axc.plot([o for o, _ in leak], [d for _, d in leak], color=PEER, lw=1.4)
axc.axhline(10, color="#999", lw=0.8, ls="-."); axc.text(255, 10.8, "10 times the floor", ha="right", va="bottom", fontsize=6.5, color="#555")
axc.axhline(1, color="#bbb", lw=0.6); axc.text(255, 1.08, "the floor", ha="right", va="bottom", fontsize=6.5, color="#777")
axc.axvline(63, color=PEER, lw=0.8, ls=":"); axc.text(65, 130, "the value is first written", fontsize=7, color=PEER)
axc.set_yscale("log"); axc.set_ylim(0.5, 300); axc.set_xlim(0, 255)
axc.set_xlabel("reply token", fontsize=7.5); axc.set_ylabel("surplus over the floor", fontsize=7.5)
axc.tick_params(labelsize=6.5)
axc.set_title("the surplus, actual minus reference at every reply token, seed 5034 leak branch, layers 16 to 38", fontsize=8, loc="left")
for s in ("top", "right"): axc.spines[s].set_visible(False)
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("wrote", out)
