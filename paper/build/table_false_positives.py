"""Every negative turn available for the count, and the jump at own first writing where it exists."""
from _common import ROOT, fmt, load, md_table, rel, summary, write

SCRIPT = "table_false_positives.py"
C_WINDOW = ROOT / "experiments/lane_c_sweep14b/results/window.json"
C_USERS = ROOT / "experiments/lane_c_sweep14b/results/users.json"
D_WINDOW = ROOT / "experiments/lane_d_sweep2/results/sweep2_window_count.json"
D_SCORES = ROOT / "experiments/lane_d_sweep2/results/sweep2_scores.json"
D_ARMS = ROOT / "experiments/lane_d_sweep2/results/arms_scores.json"
D_TIP = ROOT / "experiments/lane_d_sweep2/results/methodtip_scores.json"
E_FILE = ROOT / "experiments/lane_e_shared_workspace/results/planted_scores.json"
C_KEY = "reference L16-38 first256 count above 10"

OFF = "off-task harmless note"
ON = "on-task"
NONE = "no note"


def count_groups():
    g = []
    cs = load(C_WINDOW)["scores"]
    for b, kind in [("none", NONE), ("benign_k1", OFF + " (1 harmless tip)"), ("benign_k3", OFF + " (3 harmless tips)"),
                    ("proposal_k1", ON + " (proposal to accept each other's submissions, no value)"), ("proposal_k3", ON + " (same proposal plus 2 agreements, no value)")]:
        g.append(("Lane C", "blind solve", b, kind, "every-token kernel floor, L16-38", [cs[s][b][C_KEY] for s in cs]))
    dr = load(D_WINDOW)["rows"]
    for b, kind in [("none", NONE), ("benign_k1", OFF + " (1 harmless tip)")]:
        g.append(("Lane D sweep2", "blind solve", b, kind, "same agent's none turn, L16-38", [dr[s][b]["count"] for s in dr]))
    tip = load(D_TIP)["rows"]
    for b, kind in [("none", NONE), ("tip", ON + " (method tip, procedure without the value)")]:
        g.append(("Lane D method_tip", "blind solve", b, kind, "per-layer median over the arm's 20 none turns, L16-38",
                  [r["count"] for r in tip if r["branch"] == b]))
    sv = load(D_ARMS)["solvable"]["rows"]
    g.append(("Lane D solvable", "blind solve (solvable task)", "none", NONE, "same agent's none median, late band",
              [r["plateau"] for r in sv if r["variant"] == "none"]))
    er = load(E_FILE)["rows"]
    for b, kind in [("none", NONE + " (shared file without the line)"), ("benign", OFF + " (harmless line in the shared file)")]:
        g.append(("Lane E 2c", "first call after reading the file, derived mask", b, kind, "same agent's none median, L16-38",
                  [r["derived first"]["plateau"] for r in er if r["branch"] == b]))
    ur = load(C_USERS)["rows"]
    for v, role, kind in [("leak", "non-user", ON + " (leak note seen, value not used)"),
                          ("leak", "user", ON + " (leak note used at its solve turn; positive, listed for reference)"),
                          ("benign_k1", "non-user", OFF + " (harmless tip seen)"),
                          ("benign_k1", "user", OFF + " (blind agent, harmless tip seen)"),
                          ("benign_k1", "other", OFF + " (the leak's author as a free reader)")]:
        g.append(("Lane C verdict", f"verdict turn, role {role}", v, kind, "stage-1 floor, mean over L16-38",
                  [r["first256 count"] for r in ur if r["variant"] == v and r["role"] == role]))
    return g


def jump_groups():
    g = []
    fm = load(D_SCORES)["first_mention"]
    for v, kind in [("none", NONE), ("benign_k1", OFF + " (1 harmless tip)"), ("benign_k3", OFF + " (3 harmless tips)"),
                    ("command_k1", ON + " (1 note commanding reviewers to reject)"), ("command_k3", ON + " (3 such notes)")]:
        g.append(("Lane D sweep2", "blind solve, own answer first written", v, kind, "none turn at the same position, L31-38",
                  [r["jump | late"] for r in fm if r["variant"] == v and r["jump | late"] is not None]))
    return g


def stat_rows(groups, cuts):
    rows = []
    for lane, turn, branch, kind, floor, xs in groups:
        s = summary(xs)
        row = [lane, turn, branch, kind, floor, s["n"], fmt(s["median"], 1), fmt(s["p90"], 1), fmt(s["max"], 1)]
        for c in cuts:
            k = sum(1 for x in xs if x >= c)
            row.append(f"{k}/{len(xs)} = {k / len(xs):.2f}")
        rows.append(row)
    return rows


def main():
    lines = ["# Negative turns: count and jump", "",
             "Sources.",
             f"- Lane C blind solves: `{rel(C_WINDOW)}`, `scores[seed][branch][\"{C_KEY}\"]`.",
             f"- Lane C verdict turns: `{rel(C_USERS)}`, `rows[*][\"first256 count\"]` split by `variant` and `role` (100 rows in the file; the lane C report describes this set as 110 turns).",
             f"- Lane D sweep2 blind solves: `{rel(D_WINDOW)}`, `rows[seed][branch][\"count\"]`; jump from `{rel(D_SCORES)}`, `first_mention[*][\"jump | late\"]`.",
             f"- Lane D method_tip: `{rel(D_TIP)}`, `rows[*][\"count\"]` for branches none and tip.",
             f"- Lane D solvable: `{rel(D_ARMS)}`, `solvable.rows[*][\"plateau\"]` for variant none.",
             f"- Lane E 2c: `{rel(E_FILE)}`, `rows[*][\"derived first\"][\"plateau\"]` for branches none and benign.",
             "",
             "Definitions. count = tokens among the first 256 of the reply with relative surplus above 10 times the floor (the solvable and 2c files call it `plateau`).",
             "jump = surplus over floor at the token before the agent first writes its own submitted answer, in floor units. Floors and bands are per row.",
             "Lane C verdict turns use delta over the stage-1 floor averaged over L16-38, scored over the first 256 reply tokens.",
             "Note type: `off-task harmless note` is a peer note or planted line with no bearing on the task; `on-task` notes concern the task but carry no usable value",
             "(method tip, proposal, command) or carry a value the agent did not use (lane C non-users); `no note` is the none branch.",
             "p90 uses linear interpolation between order statistics. Fractions use >= cut.",
             "", "## Count", ""]
    lines.append(md_table(["Lane", "Turn", "Branch", "Note type", "Floor, band", "n", "median", "p90", "max", ">= 60", ">= 100"],
                          stat_rows(count_groups(), [60, 100])))
    lines += ["", "## Jump at own first writing", "",
              "Only lane D sweep2 has a per-turn jump for negative turns (the agent must write an answer; rows without one are absent from the source).",
              "Lane C negatives at `mention_pre` are summarised in its report only as ranges (benign_k1 1.1 to 1.6, placebo 1.1 to 1.7) and are not in a JSON file, so they are not tabulated here.", ""]
    lines.append(md_table(["Lane", "Turn", "Branch", "Note type", "Floor, band", "n", "median", "p90", "max", ">= 10", ">= 20", ">= 30"],
                          stat_rows(jump_groups(), [10, 20, 30])))
    write("false_positives.md", SCRIPT, lines)


if __name__ == "__main__":
    main()
