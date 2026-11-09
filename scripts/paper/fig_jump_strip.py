"""Figure 3 for the paper. The jump at the first writing, one point per turn, grouped by what the agent did with the value, log scale,
with the confirmation cut at 20 floors. Sources. Lane F's per-turn scores, rows and extra_rows of results/lane_f/lane_f_scores.json on
branch claude/lane-f-fpr-96ebe2 (lane F's definition, L16 to L38, floor from the agent's own none turn, jump at the agent's own first
writing of its submitted answer), results/lane_f/solvable_wrong_note.md on the same branch (the same definition, the jump at the note's
value for the wrong-note readers), and results/paper/tables/jump_per_turn.md for the groups scored under other lanes' definitions, the
shared-file arm (2C, L16 to L38, derived mask), the solvable arm's own-answer jumps (SV, late band) and the command negatives (S2, late
band). Groups under another definition are marked in the legend.

  .venv/bin/python scripts/paper/fig_jump_strip.py <lane_f_scores.json> <solvable_wrong_note.md> results/paper/tables/jump_per_turn.md <out.png>
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

lf, swn, jpt, out = sys.argv[1:5]
d = json.load(open(lf))
rows = d["rows"] + d.get("extra_rows", [])
PEER, SANCTION, GREY, OK = "#c2410c", "#1f5f8b", "#6b6a63", "#2f7d4f"

def md_table(path, first_col_prefix):
    """Rows of the first markdown table whose header starts with first_col_prefix, as dicts."""
    header, out = None, []
    for line in open(path, encoding="utf-8"):
        if not line.startswith("|"):
            if header and out:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            if cells[0].startswith(first_col_prefix):
                header = cells
            continue
        if set(cells[0]) <= set("-: "):
            continue
        if len(cells) == len(header):
            out.append(dict(zip(header, cells)))
    return out

def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

groups = []  # (label, values, colour, marker)
def take(pred, label, col, mk="o"):
    v = [r["jump"] for r in rows if pred(r) and r.get("jump") is not None]
    groups.append((label, v, col, mk))

# positives, lane F definition
take(lambda r: r["role"] == "blind" and r["branch"] == "leak" and r.get("arm") in (None, "sweep2"), "blind, took the leak", PEER)
take(lambda r: r["role"] == "blind" and r["branch"] == "forbidden_leak", "blind, took the forbidden leak", PEER)
take(lambda r: r["role"] == "blind" and r["branch"] == "leak" and r.get("arm") == "sweep2_norule", "blind, took the leak, rule off", PEER)

# other lanes' definitions from jump_per_turn.md
jp = md_table(jpt, "Def")
def jp_group(de, needle):
    return [num(r["Jump"]) for r in jp if r["Def"] == de and needle in r["Group"] and num(r["Jump"]) is not None]
groups.append(("shared file, copied the line (2C)", jp_group("2C", "2c leak, derived mask, first call"), PEER, "s"))
groups.append(("shared file, used the credential (2C)", jp_group("2C", "credential token first written"), PEER, "s"))
# wrong-note readers at the note's value, lane F definition
sw = md_table(swn, "Seed")
note_col = next((c for c in (sw[0].keys() if sw else []) if "note" in c.lower() and "jump" in c.lower()), None)
own_col = next((c for c in (sw[0].keys() if sw else []) if "own" in c.lower() and "jump" in c.lower()), None)
if note_col:
    groups.append(("solvable, read a wrong value, kept own answer, at the note's value", [num(r[note_col]) for r in sw if num(r[note_col]) is not None], SANCTION, "^"))
groups.append(("solvable, submitted the right value, 9 of 13 right without the note (SV)", jp_group("SV", "copied right"), OK, "^"))
# negatives
if own_col:
    groups.append(("solvable, read a wrong value, kept own answer, at own value", [num(r[own_col]) for r in sw if num(r[own_col]) is not None], GREY, "^"))
take(lambda r: r["role"] == "blind" and r["branch"] == "benign_k1", "blind, harmless tip, own answer", GREY)
groups.append(("blind, a peer's command, own answer (S2)", jp_group("S2", "command"), GREY, "D"))
take(lambda r: r["role"] == "reader", "readers of a note not for them, own answer", GREY)
take(lambda r: r["role"] == "nonblind_unmasked", "nothing masked, own answer", GREY)

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
rng = np.random.default_rng(0)
labels = []
for i, (label, v, col, mk) in enumerate(groups):
    v = np.clip(np.array(v, dtype=float), 0.2, None)
    x = i + rng.uniform(-0.22, 0.22, len(v))
    ax.scatter(x, v, s=9, color=col, marker=mk, alpha=0.75, lw=0)
    ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color="#1c1b18", lw=1.0)
    labels.append(f"{label}\n(n={len(v)})")
ax.axvspan(-0.5, 4.5, color=PEER, alpha=0.04, lw=0); ax.axvspan(4.5, 6.5, color=SANCTION, alpha=0.04, lw=0)
ax.text(2, 150, "a value from a peer span", ha="center", fontsize=7, color=PEER)
ax.text(5.5, 150, "origin, not use", ha="center", fontsize=7, color=SANCTION)
ax.text(8.5, 150, "the agent's own value", ha="center", fontsize=7, color=GREY)
ax.axhline(20, color="#555", lw=0.9, ls="-."); ax.text(len(groups) - 0.6, 22, "confirmation cut, 20 floors", ha="right", fontsize=7, color="#333")
ax.axhline(1, color="#bbb", lw=0.6)
ax.set_yscale("log"); ax.set_ylim(0.2, 220); ax.set_xlim(-0.5, len(groups) - 0.5)
ax.set_xticks(range(len(groups))); ax.set_xticklabels(labels, fontsize=5.4, rotation=32, ha="right")
ax.set_ylabel("jump, surplus over the floor at the first writing", fontsize=8); ax.tick_params(axis="y", labelsize=7)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(out); fig.savefig(out.replace(".png", ".pdf"))
print("wrote", out)
for label, v, _, _ in groups:
    v = np.array(v, dtype=float); print(f"{label}: n={len(v)} median {np.median(v):.1f} at or above 20 {int((v >= 20).sum())}")
print("solvable_wrong_note columns:", list(sw[0].keys()) if sw else None, "note col:", note_col, "own col:", own_col)
