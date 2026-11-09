"""Census of run directories: seeds, branch directories with run.json, target turns."""
from _common import LANE_D, LANE_E, ROOT, md_table, rel, write

SCRIPT = "table_census.py"
SWEEPS = [
    ("sweep1", ROOT / "runs/sweep1"),
    ("sweep14b", ROOT / "runs/sweep14b"),
    ("sweep2", LANE_D / "runs/sweep2"),
    ("sweep2_norule", LANE_D / "runs/sweep2_norule"),
    ("sweep2_leak2", LANE_D / "runs/sweep2_leak2"),
    ("sweep2_multi", LANE_D / "runs/sweep2_multi"),
    ("sweep2_solvable", LANE_D / "runs/sweep2_solvable"),
    ("sweep2_forbidden2", LANE_D / "runs/sweep2_forbidden2"),
    ("sweep2_methodtip", LANE_D / "runs/sweep2_methodtip"),
    ("planted", LANE_E / "runs/planted"),
]
# branches that plant a value the target can use
VALUE_BEARING = {"leak", "forbidden_leak", "leak2", "leak_all", "leak_solvable", "leak_solvable_wrong", "credential"}
TARGETS_PER_BRANCH = {"leak_all": 3}


def main():
    per = []
    all_branches = []
    for name, path in SWEEPS:
        seeds = sorted(p for p in path.iterdir() if p.is_dir() and p.name.startswith("seed"))
        counts = {}
        prefix = 0
        for s in seeds:
            for b in sorted(p for p in s.iterdir() if p.is_dir()):
                if not (b / "run.json").exists():
                    continue
                if b.name == "prefix":
                    prefix += 1
                    continue
                counts[b.name] = counts.get(b.name, 0) + 1
        for b in counts:
            if b not in all_branches:
                all_branches.append(b)
        per.append((name, path, len(seeds), counts, prefix))

    lines = ["# Census of run directories", "",
             "Sources: the run directories listed in the table (a seed directory is `seed*`; a branch is a subdirectory of a seed containing `run.json`).",
             "`prefix` directories also contain `run.json` but hold the shared prefix, not a branch; they are counted separately and excluded from branch totals.",
             "Seeds counts every `seed*` directory, including one that holds no branch with `run.json` (sweep14b has `seed3006_failed_prefix`, a failed prefix attempt, so 11 seed directories and 10 complete seeds).",
             "Target turns: 1 per branch, except `leak_all` with 3 targets per branch. Value-bearing branches are those that plant a value the target can use:",
             ", ".join(f"`{b}`" for b in sorted(VALUE_BEARING)) + ". The method_tip, benign, proposal, command and none branches plant no value.",
             "", "## Branch directories per sweep", ""]
    heads = ["Sweep", "Directory", "Seeds"] + all_branches + ["Branches", "prefix dirs", "Target turns", "Value-bearing branches", "Value-bearing target turns"]
    rows = []
    tot = {b: 0 for b in all_branches}
    tot_b = tot_p = tot_t = tot_vb = tot_vt = tot_s = 0
    for name, path, nseeds, counts, prefix in per:
        nb = sum(counts.values())
        nt = sum(v * TARGETS_PER_BRANCH.get(b, 1) for b, v in counts.items())
        vb = sum(v for b, v in counts.items() if b in VALUE_BEARING)
        vt = sum(v * TARGETS_PER_BRANCH.get(b, 1) for b, v in counts.items() if b in VALUE_BEARING)
        rows.append([name, f"`{rel(path)}`", nseeds] + [counts.get(b, 0) for b in all_branches] + [nb, prefix, nt, vb, vt])
        for b in all_branches:
            tot[b] += counts.get(b, 0)
        tot_b += nb; tot_p += prefix; tot_t += nt; tot_vb += vb; tot_vt += vt; tot_s += nseeds
    rows.append(["total", "", tot_s] + [tot[b] for b in all_branches] + [tot_b, tot_p, tot_t, tot_vb, tot_vt])
    lines.append(md_table(heads, rows))
    write("census.md", SCRIPT, lines)


if __name__ == "__main__":
    main()
