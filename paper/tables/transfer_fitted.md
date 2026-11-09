<!-- produced by paper/build/transfer_fitted.py; do not edit by hand -->
# The paper's fitted model, frozen, on the TaskTracker-shaped turns

Model from `paper/tables/fitted_model.json`, trained on 536 turns of our game, threshold 0.963 at zero training false positives. Features read from the external check's per-turn scores. Jump read two ways, at the first writing of the agent's own answer (the paper's label-free rule) and at the injected target (label-aware).

| Group | n | flagged, jump at own answer | flagged, jump at target | median score, own | median score, target |
| --- | --- | --- | --- | --- | --- |
| followed | 199 | 39 of 199 | 142 of 199 | 0.73 | 0.99 |
| clean | 300 | 0 of 300 | 0 of 300 | 0.01 | 0.01 |
| not followed, never written | 91 | 0 of 91 | 0 of 91 | 0.07 | 0.05 |
| not followed, written in thinking | 10 | 1 of 10 | 9 of 10 | 0.93 | 1.00 |

AUROC followed against every other turn, jump at own: 0.963; followed against poisoned non-followers only: 0.853

AUROC followed against every other turn, jump at target: 0.982; followed against poisoned non-followers only: 0.928