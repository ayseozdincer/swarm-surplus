<!-- produced by scripts/paper/table_count_pooled.py; do not edit by hand -->
# Count on blind solve turns, leak against benign_k1 and none

Sources.
- Lane C: `.claude/worktrees/lane-c-14b-8c893e/results/lane_c/window.json`, key `scores[seed][branch]["reference L16-38 first256 count above 10"]`, 10 seeds (Qwen3-14B, sweep14b).
- Lane D sweep2: `.claude/worktrees/lane-d-sweep2-affc5f/results/lane_d/sweep2_window_count.json`, key `rows[seed][branch]["count"]`, 20 seeds (Qwen3-14B self-served, rule on, code-task blind agents).
- method_tip column: `.claude/worktrees/lane-d-sweep2-affc5f/results/lane_d/methodtip_scores.json`, key `rows[*]["count"]` where `branch == "tip"`, 20 turns.

Definitions. count = number of the first 256 reply tokens whose relative surplus exceeds 10 times the floor, band L16 to L38.
Floors differ by lane: lane C uses the every-token kernel floor per layer; lane D sweep2 uses the same agent's none turn;
the method_tip arm uses the per-layer median over every reply token of that arm's 20 none turns. The pooled rows merge lane C and lane D
turns despite these slightly different floors. AUROC is the Mann-Whitney statistic (ties count one half). Paired wins are seeds whose leak
count exceeds the negative's count. The interval is a bootstrap over seeds (each resample draws seeds with replacement and takes both
the leak and the negative turn of each drawn seed), 2000 resamples, random seed 0, 2.5th and 97.5th percentiles of the resampled AUROC.
The tip is an on-task procedure with no value (method_tip arm, rules off); the leak in that arm is leak_solvable, a different arm from the
blind leak scored here, so tip is listed only as an extra negative column in the threshold table.

## AUROC, paired wins, bootstrap interval

| Lane | Contrast | Pairs | AUROC | Paired wins | Bootstrap 95% over seeds | Leak median | Negative median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Lane C (10 seeds) | leak vs benign_k1 | 10 | 1.000 | 10/10 | [1.000, 1.000] | 110.5 | 9.5 |
| Lane C (10 seeds) | leak vs none | 10 | 1.000 | 10/10 | [1.000, 1.000] | 110.5 | 0 |
| Lane D sweep2 (20 seeds) | leak vs benign_k1 | 20 | 0.973 | 20/20 | [0.917, 1.000] | 103.5 | 9.5 |
| Lane D sweep2 (20 seeds) | leak vs none | 20 | 1.000 | 20/20 | [1.000, 1.000] | 103.5 | 0 |
| Pooled (30 seeds) | leak vs benign_k1 | 30 | 0.982 | 30/30 | [0.943, 1.000] | 105 | 9.5 |
| Pooled (30 seeds) | leak vs none | 30 | 1.000 | 30/30 | [1.000, 1.000] | 105 | 0 |

## Medians per branch

| Lane | leak | benign_k1 | none | tip (method_tip arm) |
| --- | --- | --- | --- | --- |
| Lane C | 110.5 (n=10) | 9.5 (n=10) | 0 (n=10) | - |
| Lane D sweep2 | 103.5 (n=20) | 9.5 (n=20) | 0 (n=20) | - |
| Pooled | 105 (n=30) | 9.5 (n=30) | 0 (n=30) | - |
| method_tip arm (tip only) | - | - | - | 134 (n=20) |

## Threshold table

TPR = fraction of leak turns with count >= cut. FPR = fraction of negative turns with count >= cut.

| Cut | Lane C TPR leak | Lane C FPR benign_k1 | Lane C FPR none | Lane D TPR leak | Lane D FPR benign_k1 | Lane D FPR none | Pooled TPR leak | Pooled FPR benign_k1 | Pooled FPR none | method_tip FPR tip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | 10/10 = 1.00 | 2/10 = 0.20 | 0/10 = 0.00 | 20/20 = 1.00 | 8/20 = 0.40 | 0/20 = 0.00 | 30/30 = 1.00 | 10/30 = 0.33 | 0/30 = 0.00 | 20/20 = 1.00 |
| 30 | 10/10 = 1.00 | 2/10 = 0.20 | 0/10 = 0.00 | 20/20 = 1.00 | 8/20 = 0.40 | 0/20 = 0.00 | 30/30 = 1.00 | 10/30 = 0.33 | 0/30 = 0.00 | 20/20 = 1.00 |
| 40 | 10/10 = 1.00 | 2/10 = 0.20 | 0/10 = 0.00 | 20/20 = 1.00 | 4/20 = 0.20 | 0/20 = 0.00 | 30/30 = 1.00 | 6/30 = 0.20 | 0/30 = 0.00 | 20/20 = 1.00 |
| 50 | 10/10 = 1.00 | 1/10 = 0.10 | 0/10 = 0.00 | 20/20 = 1.00 | 2/20 = 0.10 | 0/20 = 0.00 | 30/30 = 1.00 | 3/30 = 0.10 | 0/30 = 0.00 | 20/20 = 1.00 |
| 60 | 10/10 = 1.00 | 0/10 = 0.00 | 0/10 = 0.00 | 20/20 = 1.00 | 1/20 = 0.05 | 0/20 = 0.00 | 30/30 = 1.00 | 1/30 = 0.03 | 0/30 = 0.00 | 20/20 = 1.00 |
| 80 | 8/10 = 0.80 | 0/10 = 0.00 | 0/10 = 0.00 | 17/20 = 0.85 | 1/20 = 0.05 | 0/20 = 0.00 | 25/30 = 0.83 | 1/30 = 0.03 | 0/30 = 0.00 | 18/20 = 0.90 |
| 100 | 6/10 = 0.60 | 0/10 = 0.00 | 0/10 = 0.00 | 11/20 = 0.55 | 1/20 = 0.05 | 0/20 = 0.00 | 17/30 = 0.57 | 1/30 = 0.03 | 0/30 = 0.00 | 17/20 = 0.85 |
