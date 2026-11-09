<!-- produced by scripts/paper/table_census.py; do not edit by hand -->
# Census of run directories

Sources: the run directories listed in the table (a seed directory is `seed*`; a branch is a subdirectory of a seed containing `run.json`).
`prefix` directories also contain `run.json` but hold the shared prefix, not a branch; they are counted separately and excluded from branch totals.
Seeds counts every `seed*` directory, including one that holds no branch with `run.json` (sweep14b has `seed3006_failed_prefix`, a failed prefix attempt, so 11 seed directories and 10 complete seeds).
Target turns: 1 per branch, except `leak_all` with 3 targets per branch. Value-bearing branches are those that plant a value the target can use:
`credential`, `forbidden_leak`, `leak`, `leak2`, `leak_all`, `leak_solvable`, `leak_solvable_wrong`. The method_tip, benign, proposal, command and none branches plant no value.

## Branch directories per sweep

| Sweep | Directory | Seeds | benign_k1 | benign_k3 | leak | none | proposal_k1 | proposal_k3 | command_k1 | command_k3 | forbidden_leak | leak2 | leak_all | leak_solvable | leak_solvable_wrong | method_tip | benign | credential | Branches | prefix dirs | Target turns | Value-bearing branches | Value-bearing target turns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sweep1 | `runs/sweep1` | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 60 | 10 | 60 | 10 | 10 |
| sweep14b | `runs/sweep14b` | 11 | 10 | 10 | 10 | 10 | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 60 | 11 | 60 | 10 | 10 |
| sweep2 | `.claude/worktrees/lane-d-sweep2-affc5f/runs/sweep2` | 20 | 20 | 20 | 20 | 20 | 0 | 0 | 20 | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 140 | 20 | 140 | 40 | 40 |
| sweep2_norule | `.claude/worktrees/lane-d-sweep2-affc5f/runs/sweep2_norule` | 8 | 8 | 8 | 8 | 8 | 0 | 0 | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 48 | 8 | 48 | 8 | 8 |
| sweep2_leak2 | `.claude/worktrees/lane-d-sweep2-affc5f/runs/sweep2_leak2` | 10 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 10 | 20 | 10 | 10 |
| sweep2_multi | `.claude/worktrees/lane-d-sweep2-affc5f/runs/sweep2_multi` | 10 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 0 | 0 | 20 | 10 | 40 | 10 | 30 |
| sweep2_solvable | `.claude/worktrees/lane-d-sweep2-affc5f/runs/sweep2_solvable` | 20 | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 20 | 0 | 0 | 0 | 60 | 20 | 60 | 40 | 40 |
| sweep2_forbidden2 | `.claude/worktrees/lane-d-sweep2-affc5f/runs/sweep2_forbidden2` | 20 | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 40 | 20 | 40 | 20 | 20 |
| sweep2_methodtip | `.claude/worktrees/lane-d-sweep2-affc5f/runs/sweep2_methodtip` | 20 | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 0 | 20 | 0 | 0 | 60 | 20 | 60 | 20 | 20 |
| planted | `.claude/worktrees/lane-e-2b-713994/runs/planted` | 20 | 0 | 0 | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 20 | 80 | 20 | 80 | 40 | 40 |
| total |  | 149 | 48 | 48 | 68 | 148 | 20 | 20 | 28 | 28 | 40 | 10 | 10 | 40 | 20 | 20 | 20 | 20 | 588 | 149 | 608 | 208 | 228 |
