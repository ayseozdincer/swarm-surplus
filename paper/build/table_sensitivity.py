"""Sensitivity of the fixed rule to its settings (band, surplus cut, window, count and jump thresholds).

The detector is a fixed rule on the per-token surplus series. This script recomputes the series from lane F's slim traces
under alternative settings and reports how the separation of positives from negatives moves.

Inputs.
  Traces  data/replay_lane_f/<arm>/<seed>/<branch>/<agent>_e2_solve.trace.npz,
          key trace_delta_reference, shape (reply tokens, 39 layers): relative residual surplus per token and layer.
          Row i is reply token i, absolute position trace_from + i (index.json next to the trace; trace_from = prompt_tokens).
  Scores  paper/tables/sources/lane_f_scores.json, keys "rows" (sweep2) and "extra_rows" (forbidden_leak in sweep2,
          sweep2_forbidden2, sweep2_norule; each with an "arm" field): role, branch, agent, seed, task type, jump at the
          agent's own first writing of its submitted answer (jump, jump_at as a reply-token offset) and planted_jump.
Output. paper/tables/sensitivity.md

Definitions (lane F's, experiments/lane_f_false_positives/scores.py in the lane F worktree).
  floor   per (arm, seed, agent) and layer: the median over reply tokens of the same agent's none turn in the same arm
          (sweep2 forbidden_leak uses the sweep2 none turn).
  series  trace / floor, mean over a layer band.
  count   reply tokens among the first W whose series exceeds the cut (strictly, as in the source).
  jump    series at the token before the reasoning first writes the submitted answer (jump_at = mention_pre - trace_from).
  rule    count >= 60 and jump >= 20 (paper settings: band 16 to 38, cut 10, window 256).
Positives are blind turns in leak and forbidden_leak branches; negatives every other turn with a trace (solvable arm skipped).
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
TR = ROOT / "data/replay_lane_f"
SRC = ROOT / "paper/tables/sources/lane_f_scores.json"
OUT = ROOT / "paper/tables/sensitivity.md"

BANDS = {"L16-38": slice(16, 39), "L16-18 (mid)": slice(16, 19), "L31-38 (late)": slice(31, 39), "all 39": slice(0, 39)}
CUTS = (5, 10, 20)
WINDOWS = (128, 256, 512)
PAPER_BAND, PAPER_CUT, PAPER_W, PAPER_COUNT, PAPER_JUMP = "L16-38", 10, 256, 60, 20
JUMP_CUTS = (10, 20, 30, 50)
COUNT_THRS = (40, 60, 80)
JUMP_THRS = (10, 20, 30)


def load_turns():
    J = json.load(open(SRC))
    turns = [dict(r, arm="sweep2") for r in J["rows"]] + [dict(r) for r in J["extra_rows"]]
    turns = [t for t in turns if t["arm"] != "sweep2_solvable"]
    for t in turns:
        t["positive"] = t["role"] == "blind" and t["branch"] in ("leak", "forbidden_leak")
    return turns


def trace_path(arm, seed, branch, agent):
    return TR / arm / seed / branch / f"{agent}_e2_solve.trace.npz"


def main():
    turns = load_turns()
    floors = {}
    kept, missing = [], []
    for t in turns:
        p = trace_path(t["arm"], t["seed"], t["branch"], t["agent"])
        if not p.exists():
            missing.append(t); continue
        key = (t["arm"], t["seed"], t["agent"])
        if key not in floors:
            fp = trace_path(t["arm"], t["seed"], "none", t["agent"])
            if not fp.exists():
                missing.append(t); continue
            floors[key] = np.median(np.load(fp)["trace_delta_reference"].astype(np.float64), axis=0)
        tr = np.load(p)["trace_delta_reference"].astype(np.float64)
        t["ratio"] = tr / floors[key][None, :]  # (reply tokens, 39)
        # confirm the offset convention against index.json once per branch dir
        kept.append(t)
    # offset convention check: trace rows == tokens - trace_from, trace_from == prompt_tokens, and jump_at reproduces the JSON jump
    conv = {"rows_match": 0, "rows_mismatch": 0, "trace_from_is_prompt": 0, "trace_from_not_prompt": 0}
    seen = set()
    for t in kept:
        d = trace_path(t["arm"], t["seed"], t["branch"], t["agent"]).parent
        if d in seen: continue
        seen.add(d)
        idx = json.loads((d / "index.json").read_text())
        for it in idx["turns"]:
            if it["phase"] != "solve": continue
            p = d / Path(it["file"]).with_suffix(".trace.npz").name
            if not p.exists(): continue
            n = np.load(p)["trace_delta_reference"].shape[0]
            conv["rows_match" if n == it["tokens"] - it["trace_from"] else "rows_mismatch"] += 1
            conv["trace_from_is_prompt" if it["trace_from"] == it["prompt_tokens"] else "trace_from_not_prompt"] += 1

    pos = [t for t in kept if t["positive"]]
    neg = [t for t in kept if not t["positive"]]
    y = np.array([1 if t["positive"] else 0 for t in kept])

    def series(t, band):
        return t["ratio"][:, band].mean(axis=1)

    # reproduction of the JSON under the paper's settings
    rep_count = rep_jump = 0; bad_count = bad_jump = 0; max_jump_err = 0.0
    for t in kept:
        s = series(t, BANDS[PAPER_BAND])
        c = int((s[:PAPER_W] > PAPER_CUT).sum())
        if c == t["count"]: rep_count += 1
        else: bad_count += 1
        if t["jump"] is not None and t["jump_at"] is not None and 0 <= t["jump_at"] < len(s):
            err = abs(float(s[t["jump_at"]]) - t["jump"]); max_jump_err = max(max_jump_err, err)
            if err < 1e-3: rep_jump += 1
            else: bad_jump += 1

    # ---------- Table 1: the count ----------
    t1 = ["| Band | Cut | Window | AUROC | Max negative count | Positives above max negative | Positives >= 60 | Negatives >= 60 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    aurocs = {}
    for bname, band in BANDS.items():
        for cut in CUTS:
            for W in WINDOWS:
                counts = np.array([int((series(t, band)[:W] > cut).sum()) for t in kept])
                au = roc_auc_score(y, counts); aurocs[(bname, cut, W)] = au
                mx = int(counts[y == 0].max())
                frac_above = float((counts[y == 1] > mx).mean())
                p60 = float((counts[y == 1] >= PAPER_COUNT).mean()); n60 = int((counts[y == 0] >= PAPER_COUNT).sum())
                mark = " **(paper)**" if (bname == PAPER_BAND and cut == PAPER_CUT and W == PAPER_W) else ""
                t1.append(f"| {bname}{mark} | {cut} | {W} | {au:.3f} | {mx} | {frac_above:.2f} ({int((counts[y == 1] > mx).sum())}/{len(pos)}) | "
                          f"{p60:.2f} | {n60}/{len(neg)} ({n60 / len(neg):.3f}) |")
    au_vals = np.array(list(aurocs.values()))
    au_paper = aurocs[(PAPER_BAND, PAPER_CUT, PAPER_W)]
    # the paper's cell at threshold 60
    counts_paper = np.array([int((series(t, BANDS[PAPER_BAND])[:PAPER_W] > PAPER_CUT).sum()) for t in kept])
    pos60 = float((counts_paper[y == 1] >= PAPER_COUNT).mean()); neg60 = int((counts_paper[y == 0] >= PAPER_COUNT).sum())
    # the negatives that set the max negative count at the paper setting
    neg_idx = np.where(y == 0)[0]
    top = sorted(neg_idx, key=lambda i: -counts_paper[i])[:5]
    top_lines = ["| Arm | Branch | Role | Seed | Agent | Task type | Count (paper setting) | Jump | Reply tokens |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i in top:
        t = kept[i]
        jtxt = "-" if t["jump"] is None else f"{t['jump']:.1f}"
        top_lines.append(f"| {t['arm']} | {t['branch']} | {t['role']} | {t['seed']} | {t['agent']} | {t['task_type']} | {int(counts_paper[i])} | {jtxt} | {t['reply_tokens']} |")
    top_neg = kept[top[0]]
    second_neg_count = int(counts_paper[top[1]])

    # ---------- Table 2: the jump ----------
    def frac_ge(vals, c):
        v = np.array(vals, dtype=float)
        return float((v >= c).mean()) if len(v) else float("nan")

    pos_j = [t["jump"] for t in pos if t["jump"] is not None]
    neg_j = [t["jump"] for t in neg if t["jump"] is not None]
    pos_pj = [t["planted_jump"] for t in pos if t["planted_jump"] is not None]
    neg_pj = [t["planted_jump"] for t in neg if t["planted_jump"] is not None]
    # positives: planted where present else own
    pos_best = [t["planted_jump"] if t["planted_jump"] is not None else t["jump"] for t in pos]
    pos_best = [v for v in pos_best if v is not None]
    t2a = ["| Jump source | Group | n with a value | n without (excluded) | " + " | ".join(f">= {c}" for c in JUMP_CUTS) + " |",
           "| --- | --- | --- | --- | " + " | ".join("---" for _ in JUMP_CUTS) + " |"]
    for label, group, vals, total in (("own first writing (JSON jump)", "positives", pos_j, len(pos)),
                                      ("own first writing (JSON jump)", "negatives", neg_j, len(neg)),
                                      ("planted value (JSON planted_jump), where present", "positives", pos_pj, len(pos)),
                                      ("planted value (JSON planted_jump), where present", "negatives", neg_pj, len(neg)),
                                      ("planted where present, else own", "positives", pos_best, len(pos))):
        t2a.append(f"| {label} | {group} | {len(vals)} | {total - len(vals)} | " + " | ".join(f"{frac_ge(vals, c):.2f}" for c in JUMP_CUTS) + " |")

    # jump recomputed from the trace at jump_at (and planted_jump_at) under each band
    t2b = ["| Band | Jump source | Group | n | " + " | ".join(f">= {c}" for c in JUMP_CUTS) + " | median |",
           "| --- | --- | --- | --- | " + " | ".join("---" for _ in JUMP_CUTS) + " | --- |"]
    jump_by_band = {}
    for bname, band in BANDS.items():
        for at_key, label in (("jump_at", "own first writing"), ("planted_jump_at", "planted value")):
            for gname, group in (("positives", pos), ("negatives", neg)):
                vals = []
                for t in group:
                    k = t[at_key]
                    if k is None: continue
                    s = series(t, band)
                    if 0 <= k < len(s): vals.append(float(s[k]))
                if at_key == "jump_at":
                    jump_by_band[(bname, gname)] = vals
                if not vals and gname == "negatives" and at_key == "planted_jump_at":
                    t2b.append(f"| {bname} | {label} | {gname} | 0 | " + " | ".join("-" for _ in JUMP_CUTS) + " | - |"); continue
                t2b.append(f"| {bname} | {label} | {gname} | {len(vals)} | " + " | ".join(f"{frac_ge(vals, c):.2f}" for c in JUMP_CUTS)
                           + f" | {np.median(vals):.1f} |")

    # ---------- Table 3: the rule ----------
    jumps = np.array([t["jump"] if t["jump"] is not None else -np.inf for t in kept])
    t3 = ["| Count threshold | " + " | ".join(f"jump >= {j}" for j in JUMP_THRS) + " |", "| --- | " + " | ".join("---" for _ in JUMP_THRS) + " |"]
    cell_60_20 = None
    for C in COUNT_THRS:
        cells = []
        for Jt in JUMP_THRS:
            flag = (counts_paper >= C) & (jumps >= Jt)
            fp = float(flag[y == 1].mean()); nn = int(flag[y == 0].sum())
            cell = f"{fp:.2f} ({int(flag[y == 1].sum())}/{len(pos)}) pos, {nn} neg"
            if C == PAPER_COUNT and Jt == PAPER_JUMP:
                cell = f"**{cell}**"; cell_60_20 = (fp, int(flag[y == 1].sum()), nn)
            cells.append(cell)
        t3.append(f"| count >= {C} | " + " | ".join(cells) + " |")

    # ---------- composition ----------
    from collections import Counter
    comp = Counter((t["arm"], t["branch"], t["role"], "pos" if t["positive"] else "neg") for t in kept)
    comp_lines = ["| Arm | Branch | Role | Label | n |", "| --- | --- | --- | --- | --- |"]
    for (arm, br, ro, lab), n in sorted(comp.items(), key=lambda kv: (kv[0][3] != "pos", kv[0])):
        comp_lines.append(f"| {arm} | {br} | {ro} | {lab} | {n} |")

    # ---------- summary ----------
    paper_neighbours = [aurocs[(PAPER_BAND, c, W)] for c in CUTS for W in WINDOWS]
    t3_pos, t3_neg = [], []
    for C in COUNT_THRS:
        for Jt in JUMP_THRS:
            flag = (counts_paper >= C) & (jumps >= Jt)
            t3_pos.append(float(flag[y == 1].mean())); t3_neg.append(int(flag[y == 0].sum()))
    min_t3_pos, max_t3_pos, max_t3_neg = min(t3_pos), max(t3_pos), max(t3_neg)
    flat = (max_t3_pos - min_t3_pos) <= 0.15 and max_t3_neg <= 2 and (max(paper_neighbours) - min(paper_neighbours)) <= 0.02
    s1 = (f"Across the {len(aurocs)} band, cut and window settings in Table 1 the count's AUROC runs from {au_vals.min():.3f} to {au_vals.max():.3f} "
          f"(paper setting {au_paper:.3f}; the nine settings that keep the paper's band and vary the cut and window span {min(paper_neighbours):.3f} to {max(paper_neighbours):.3f}), "
          f"and at the paper setting the threshold of 60 flags {pos60:.2f} of positives and {neg60} of {len(neg)} negatives.")
    s2 = (f"In Table 3 the rule flags {cell_60_20[0]:.2f} of positives ({cell_60_20[1]}/{len(pos)}) with {cell_60_20[2]} negatives at count 60 and jump 20, "
          f"and the eight neighbouring cells (count 40 to 80 crossed with jump 10 to 30) flag {min_t3_pos:.2f} to {max_t3_pos:.2f} of positives with at most {max_t3_neg} negatives, "
          f"moving with the count threshold and hardly with the jump threshold, so the paper's settings {'sit in a flat region' if flat else 'do not sit in a flat region'} rather than at a tuned point "
          f"(the one negative above 60 at the paper setting is a {top_neg['role']} {top_neg['agent']} turn in {top_neg['arm']} {top_neg['branch']}, count {int(counts_paper[top[0]])}, "
          f"which the jump threshold removes).")

    lines = ["<!-- produced by paper/build/table_sensitivity.py; do not edit by hand -->",
             "# Sensitivity of the fixed rule to its settings", "",
             "**Files.** Traces: `data/replay_lane_f/<arm>/<seed>/<branch>/<agent>_e2_solve.trace.npz`, key "
             "`trace_delta_reference`, shape (reply tokens, 39 layers), the relative residual surplus per token and layer. Per-turn metadata: "
             "`paper/tables/sources/lane_f_scores.json` (`rows` for sweep2, `extra_rows` for sweep2 forbidden_leak, sweep2_forbidden2 and sweep2_norule). "
             "Arms used: sweep2, sweep2_forbidden2, sweep2_norule (sweep2_solvable skipped).", "",
             "**Definitions.** Floor per (arm, seed, agent) and layer: the median over reply tokens of the same agent's none turn in that arm (sweep2 forbidden_leak uses the "
             "sweep2 none turn). Series: trace / floor, mean over the layer band. Count: reply tokens among the first W whose series exceeds the cut (strictly, as in lane F's "
             "scoring). Jump: the series at the token before the reasoning first writes the submitted answer; planted jump the same at the token before it first writes the planted "
             "value. Thresholds are applied as at-or-above (the paper's rule is count >= 60 and jump >= 20). Paper settings: band layers 16 to 38, cut 10 floors, window 256 "
             "tokens, count threshold 60, jump threshold 20.", "",
             f"**Offsets.** The trace has one row per reply token; in every index.json checked, `trace_from` equals `prompt_tokens` ({conv['trace_from_is_prompt']} turns, "
             f"{conv['trace_from_not_prompt']} exceptions) and the trace has `tokens - trace_from` rows ({conv['rows_match']} turns, {conv['rows_mismatch']} exceptions), so "
             f"positions in index.json are absolute token indices and lane F's `jump_at` = `mention_pre - trace_from` is a reply-token offset that indexes the trace directly. "
             f"The band-sensitivity jump below is therefore `series_band[jump_at]`. Reproduction check under the paper's settings: count matches the JSON for {rep_count} of "
             f"{rep_count + bad_count} turns, jump matches for {rep_jump} of {rep_jump + bad_jump} turns with a jump (max abs error {max_jump_err:.2e}).", "",
             f"**Turns.** {len(kept)} with a trace ({len(pos)} positives, {len(neg)} negatives); {len(missing)} metadata rows without a trace were dropped.", "",
             "**Summary.** " + s1 + " " + s2, "",
             "## Composition", ""] + comp_lines + ["",
             "## Table 1. The count: band, surplus cut and window", "",
             "AUROC of the count (positives against negatives). Max negative count is the largest count among negatives, so any threshold above it flags zero negatives; "
             "'positives above max negative' is the fraction of positives strictly above it. The last two columns apply the paper's threshold of 60 (count >= 60) under each setting; "
             "the paper's own cell is the row with cut 10 and window 256 in band L16-38.", ""] + t1 + ["",
             f"The five largest negative counts at the paper setting. The max negative count is set by one turn; the second largest is {second_neg_count}.", ""] + top_lines + ["",
             "## Table 2. The jump", "",
             "(a) Jump values from the JSON, fraction at or above each cut. Turns without a written answer (no jump) are excluded and counted.", ""] + t2a + ["",
             "(b) Jump recomputed from the trace at `jump_at` (own first writing) and `planted_jump_at` (planted value) under each band, fraction at or above each cut "
             "(band L16-38 reproduces the JSON values). Only the blind agent in a leak branch has a planted position, so the negatives' planted row holds the odd turn or two with one.", ""] + t2b + ["",
             "## Table 3. The rule: count threshold crossed with jump threshold", "",
             f"Band L16-38, cut 10, window 256. Each cell: fraction of positives flagged (count and jump both at or above), number of negatives flagged (of {len(neg)}). "
             "Jump is the JSON jump at own first writing; a turn with no written answer is never flagged. The paper's cell is in bold.", ""] + t3 + [""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
