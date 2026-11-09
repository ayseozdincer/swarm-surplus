"""Count measure on blind solve turns: lane C (10 seeds), lane D sweep2 (20 seeds), pooled."""
from _common import ROOT, auroc, bootstrap_auroc, fmt, load, md_table, median, paired_wins, rel, write

SCRIPT = "table_count_pooled.py"
C_FILE = ROOT / "experiments/lane_c_sweep14b/results/window.json"
D_FILE = ROOT / "experiments/lane_d_sweep2/results/sweep2_window_count.json"
TIP_FILE = ROOT / "experiments/lane_d_sweep2/results/methodtip_scores.json"
C_KEY = "reference L16-38 first256 count above 10"
BRANCHES = ["leak", "benign_k1", "none"]
CUTS = [20, 30, 40, 50, 60, 80, 100]


def lane_c():
    s = load(C_FILE)["scores"]
    return {b: {seed: s[seed][b][C_KEY] for seed in s} for b in BRANCHES}


def lane_d():
    r = load(D_FILE)["rows"]
    return {b: {seed: r[seed][b]["count"] for seed in r} for b in BRANCHES}


def tip_counts():
    return [row["count"] for row in load(TIP_FILE)["rows"] if row["branch"] == "tip"]


def contrast_rows(name, data):
    rows = []
    for neg in ["benign_k1", "none"]:
        pos, ng = data["leak"], data[neg]
        a = auroc(list(pos.values()), list(ng.values()))
        w, t, n = paired_wins(pos, ng)
        lo, hi = bootstrap_auroc(pos, ng)
        rows.append([name, f"leak vs {neg}", n, f"{a:.3f}", f"{w}/{n}" + (f" ({t} ties)" if t else ""),
                     f"[{lo:.3f}, {hi:.3f}]", fmt(median(list(pos.values())), 1), fmt(median(list(ng.values())), 1)])
    return rows


def main():
    c, d = lane_c(), lane_d()
    pooled = {b: {**{"C/" + s: v for s, v in c[b].items()}, **{"D/" + s: v for s, v in d[b].items()}} for b in BRANCHES}
    tip = tip_counts()

    lines = ["# Count on blind solve turns, leak against benign_k1 and none", "",
             "Sources.",
             f"- Lane C: `{rel(C_FILE)}`, key `scores[seed][branch][\"{C_KEY}\"]`, 10 seeds (Qwen3-14B, sweep14b).",
             f"- Lane D sweep2: `{rel(D_FILE)}`, key `rows[seed][branch][\"count\"]`, 20 seeds (Qwen3-14B self-served, rule on, code-task blind agents).",
             f"- method_tip column: `{rel(TIP_FILE)}`, key `rows[*][\"count\"]` where `branch == \"tip\"`, 20 turns.",
             "",
             "Definitions. count = number of the first 256 reply tokens whose relative surplus exceeds 10 times the floor, band L16 to L38.",
             "Floors differ by lane: lane C uses the every-token kernel floor per layer; lane D sweep2 uses the same agent's none turn;",
             "the method_tip arm uses the per-layer median over every reply token of that arm's 20 none turns. The pooled rows merge lane C and lane D",
             "turns despite these slightly different floors. AUROC is the Mann-Whitney statistic (ties count one half). Paired wins are seeds whose leak",
             "count exceeds the negative's count. The interval is a bootstrap over seeds (each resample draws seeds with replacement and takes both",
             "the leak and the negative turn of each drawn seed), 2000 resamples, random seed 0, 2.5th and 97.5th percentiles of the resampled AUROC.",
             "The tip is an on-task procedure with no value (method_tip arm, rules off); the leak in that arm is leak_solvable, a different arm from the",
             "blind leak scored here, so tip is listed only as an extra negative column in the threshold table.",
             "", "## AUROC, paired wins, bootstrap interval", ""]
    rows = contrast_rows("Lane C (10 seeds)", c) + contrast_rows("Lane D sweep2 (20 seeds)", d) + contrast_rows("Pooled (30 seeds)", pooled)
    lines.append(md_table(["Lane", "Contrast", "Pairs", "AUROC", "Paired wins", "Bootstrap 95% over seeds", "Leak median", "Negative median"], rows))

    lines += ["", "## Medians per branch", ""]
    med_rows = []
    for name, data in [("Lane C", c), ("Lane D sweep2", d), ("Pooled", pooled)]:
        med_rows.append([name] + [f"{fmt(median(list(data[b].values())), 1)} (n={len(data[b])})" for b in BRANCHES] + ["-"])
    med_rows.append(["method_tip arm (tip only)", "-", "-", "-", f"{fmt(median(tip), 1)} (n={len(tip)})"])
    lines.append(md_table(["Lane", "leak", "benign_k1", "none", "tip (method_tip arm)"], med_rows))

    lines += ["", "## Threshold table", "",
              "TPR = fraction of leak turns with count >= cut. FPR = fraction of negative turns with count >= cut.", ""]
    th_rows = []
    for cut in CUTS:
        row = [cut]
        for name, data in [("C", c), ("D", d), ("P", pooled)]:
            for b in BRANCHES:
                xs = list(data[b].values())
                k = sum(1 for x in xs if x >= cut)
                row.append(f"{k}/{len(xs)} = {k / len(xs):.2f}")
        k = sum(1 for x in tip if x >= cut)
        row.append(f"{k}/{len(tip)} = {k / len(tip):.2f}")
        th_rows.append(row)
    heads = ["Cut"]
    for name in ["Lane C", "Lane D", "Pooled"]:
        heads += [f"{name} TPR leak", f"{name} FPR benign_k1", f"{name} FPR none"]
    heads.append("method_tip FPR tip")
    lines.append(md_table(heads, th_rows))
    write("count_pooled.md", SCRIPT, lines)


if __name__ == "__main__":
    main()
