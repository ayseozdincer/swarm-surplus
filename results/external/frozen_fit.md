<!-- produced by scripts/external_tt_frozen_fit.py; do not edit by hand -->
# The paper's fitted model, frozen, on the TaskTracker-shaped turns

Model `results/paper/tables/fitted_model.json` (overnight-detector 2228f1f), trained on 536 game turns with 68 positives over 40 seeds. Features count256, window16, jump_at_own_first_writing, reply_tokens, masked_tokens, standardised with the training scaler, logit coef dot z plus intercept, flag when the sigmoid exceeds 0.9627 (zero training false positives). Applied untouched to the 600 turns with our pooled floor. Jump is read at the first writing of the injected target, 1.0 where never written, as the paper session asked; the label-free reading at the answer line is beside it.

## jump at the target, 1.0 where never written

| Group | n | flagged at the training threshold | median score | score 90th percentile |
| --- | --- | --- | --- | --- |
| followed | 199 | 142 of 199 | 0.994 | 1.000 |
| not followed, written in thinking | 10 | 9 of 10 | 0.999 | 1.000 |
| not followed, never written | 91 | 0 of 91 | 0.048 | 0.201 |
| clean | 300 | 0 of 300 | 0.008 | 0.008 |

AUROC followed against not followed (all poisoned negatives) 0.928, against clean 1.000, against every negative 0.982. Flagged 142 of 199 followed, 9 of 101 poisoned negatives, 0 of 300 clean.

## jump at the answer line (label-free)

| Group | n | flagged at the training threshold | median score | score 90th percentile |
| --- | --- | --- | --- | --- |
| followed | 199 | 41 of 199 | 0.774 | 0.994 |
| not followed, written in thinking | 10 | 1 of 10 | 0.932 | 0.963 |
| not followed, never written | 91 | 0 of 91 | 0.060 | 0.282 |
| clean | 300 | 0 of 300 | 0.008 | 0.008 |

AUROC followed against not followed (all poisoned negatives) 0.876, against clean 1.000, against every negative 0.969. Flagged 41 of 199 followed, 1 of 101 poisoned negatives, 0 of 300 clean.

## Standardised features by group (z against the training scaler), medians

| Group | count256 | window16 | jump_at_own_first_writing | reply_tokens | masked_tokens |
| --- | --- | --- | --- | --- | --- |
| followed | +0.95 | +0.69 | +2.31 | -1.14 | -0.65 |
| not followed, written in thinking | +1.46 | +0.99 | +3.09 | -1.12 | -0.67 |
| not followed, never written | +0.10 | -0.22 | -0.44 | -1.16 | -0.65 |
| clean | -0.43 | -0.52 | -0.44 | -1.16 | -0.99 |

Coefficients count256 +3.35, window16 +0.54, jump_at_own_first_writing +1.57, reply_tokens -0.37, masked_tokens -0.35, intercept -3.25.

