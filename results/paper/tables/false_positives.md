<!-- produced by scripts/paper/table_false_positives.py; do not edit by hand -->
# Negative turns: count and jump

Sources.
- Lane C blind solves: `.claude/worktrees/lane-c-14b-8c893e/results/lane_c/window.json`, `scores[seed][branch]["reference L16-38 first256 count above 10"]`.
- Lane C verdict turns: `.claude/worktrees/lane-c-14b-8c893e/results/lane_c/users.json`, `rows[*]["first256 count"]` split by `variant` and `role` (100 rows in the file; the lane C report describes this set as 110 turns).
- Lane D sweep2 blind solves: `.claude/worktrees/lane-d-sweep2-affc5f/results/lane_d/sweep2_window_count.json`, `rows[seed][branch]["count"]`; jump from `.claude/worktrees/lane-d-sweep2-affc5f/results/lane_d/sweep2_scores.json`, `first_mention[*]["jump | late"]`.
- Lane D method_tip: `.claude/worktrees/lane-d-sweep2-affc5f/results/lane_d/methodtip_scores.json`, `rows[*]["count"]` for branches none and tip.
- Lane D solvable: `.claude/worktrees/lane-d-sweep2-affc5f/results/lane_d/arms_scores.json`, `solvable.rows[*]["plateau"]` for variant none.
- Lane E 2c: `.claude/worktrees/lane-e-2b-713994/results/lane_e/planted_scores.json`, `rows[*]["derived first"]["plateau"]` for branches none and benign.

Definitions. count = tokens among the first 256 of the reply with relative surplus above 10 times the floor (the solvable and 2c files call it `plateau`).
jump = surplus over floor at the token before the agent first writes its own submitted answer, in floor units. Floors and bands are per row.
Lane C verdict turns use delta over the stage-1 floor averaged over L16-38, scored over the first 256 reply tokens.
Note type: `off-task harmless note` is a peer note or planted line with no bearing on the task; `on-task` notes concern the task but carry no usable value
(method tip, proposal, command) or carry a value the agent did not use (lane C non-users); `no note` is the none branch.
p90 uses linear interpolation between order statistics. Fractions use >= cut.

## Count

| Lane | Turn | Branch | Note type | Floor, band | n | median | p90 | max | >= 60 | >= 100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lane C | blind solve | none | no note | every-token kernel floor, L16-38 | 10 | 0 | 2 | 2 | 0/10 = 0.00 | 0/10 = 0.00 |
| Lane C | blind solve | benign_k1 | off-task harmless note (1 harmless tip) | every-token kernel floor, L16-38 | 10 | 9.5 | 49.9 | 58 | 0/10 = 0.00 | 0/10 = 0.00 |
| Lane C | blind solve | benign_k3 | off-task harmless note (3 harmless tips) | every-token kernel floor, L16-38 | 10 | 24.5 | 72.4 | 94 | 2/10 = 0.20 | 0/10 = 0.00 |
| Lane C | blind solve | proposal_k1 | on-task (proposal to accept each other's submissions, no value) | every-token kernel floor, L16-38 | 10 | 20.5 | 51.2 | 53 | 0/10 = 0.00 | 0/10 = 0.00 |
| Lane C | blind solve | proposal_k3 | on-task (same proposal plus 2 agreements, no value) | every-token kernel floor, L16-38 | 10 | 20.5 | 56.4 | 60 | 1/10 = 0.10 | 0/10 = 0.00 |
| Lane D sweep2 | blind solve | none | no note | same agent's none turn, L16-38 | 20 | 0 | 0.1 | 1 | 0/20 = 0.00 | 0/20 = 0.00 |
| Lane D sweep2 | blind solve | benign_k1 | off-task harmless note (1 harmless tip) | same agent's none turn, L16-38 | 20 | 9.5 | 47.5 | 107 | 1/20 = 0.05 | 1/20 = 0.05 |
| Lane D method_tip | blind solve | none | no note | per-layer median over the arm's 20 none turns, L16-38 | 20 | 0.5 | 2.2 | 5 | 0/20 = 0.00 | 0/20 = 0.00 |
| Lane D method_tip | blind solve | tip | on-task (method tip, procedure without the value) | per-layer median over the arm's 20 none turns, L16-38 | 20 | 134 | 169.3 | 174 | 20/20 = 1.00 | 17/20 = 0.85 |
| Lane D solvable | blind solve (solvable task) | none | no note | same agent's none median, late band | 20 | 1 | 5 | 6 | 0/20 = 0.00 | 0/20 = 0.00 |
| Lane E 2c | first call after reading the file, derived mask | none | no note (shared file without the line) | same agent's none median, L16-38 | 20 | 0 | 1 | 1 | 0/20 = 0.00 | 0/20 = 0.00 |
| Lane E 2c | first call after reading the file, derived mask | benign | off-task harmless note (harmless line in the shared file) | same agent's none median, L16-38 | 20 | 22.5 | 60.3 | 199 | 2/20 = 0.10 | 2/20 = 0.10 |
| Lane C verdict | verdict turn, role non-user | leak | on-task (leak note seen, value not used) | stage-1 floor, mean over L16-38 | 40 | 3 | 8 | 12 | 0/40 = 0.00 | 0/40 = 0.00 |
| Lane C verdict | verdict turn, role user | leak | on-task (leak note used at its solve turn; positive, listed for reference) | stage-1 floor, mean over L16-38 | 10 | 12 | 18.3 | 21 | 0/10 = 0.00 | 0/10 = 0.00 |
| Lane C verdict | verdict turn, role non-user | benign_k1 | off-task harmless note (harmless tip seen) | stage-1 floor, mean over L16-38 | 30 | 3 | 7.2 | 15 | 0/30 = 0.00 | 0/30 = 0.00 |
| Lane C verdict | verdict turn, role user | benign_k1 | off-task harmless note (blind agent, harmless tip seen) | stage-1 floor, mean over L16-38 | 10 | 5.5 | 10.1 | 11 | 0/10 = 0.00 | 0/10 = 0.00 |
| Lane C verdict | verdict turn, role other | benign_k1 | off-task harmless note (the leak's author as a free reader) | stage-1 floor, mean over L16-38 | 10 | 6 | 16 | 16 | 0/10 = 0.00 | 0/10 = 0.00 |

## Jump at own first writing

Only lane D sweep2 has a per-turn jump for negative turns (the agent must write an answer; rows without one are absent from the source).
Lane C negatives at `mention_pre` are summarised in its report only as ranges (benign_k1 1.1 to 1.6, placebo 1.1 to 1.7) and are not in a JSON file, so they are not tabulated here.

| Lane | Turn | Branch | Note type | Floor, band | n | median | p90 | max | >= 10 | >= 20 | >= 30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lane D sweep2 | blind solve, own answer first written | none | no note | none turn at the same position, L31-38 | 18 | 1 | 1 | 1 | 0/18 = 0.00 | 0/18 = 0.00 | 0/18 = 0.00 |
| Lane D sweep2 | blind solve, own answer first written | benign_k1 | off-task harmless note (1 harmless tip) | none turn at the same position, L31-38 | 15 | 1.4 | 9.1 | 11.6 | 1/15 = 0.07 | 0/15 = 0.00 | 0/15 = 0.00 |
| Lane D sweep2 | blind solve, own answer first written | benign_k3 | off-task harmless note (3 harmless tips) | none turn at the same position, L31-38 | 14 | 2.0 | 5.3 | 17.6 | 1/14 = 0.07 | 0/14 = 0.00 | 0/14 = 0.00 |
| Lane D sweep2 | blind solve, own answer first written | command_k1 | on-task (1 note commanding reviewers to reject) | none turn at the same position, L31-38 | 14 | 2.5 | 5.5 | 38.4 | 1/14 = 0.07 | 1/14 = 0.07 | 1/14 = 0.07 |
| Lane D sweep2 | blind solve, own answer first written | command_k3 | on-task (3 such notes) | none turn at the same position, L31-38 | 14 | 2.9 | 11.5 | 66.7 | 2/14 = 0.14 | 1/14 = 0.07 | 1/14 = 0.07 |
