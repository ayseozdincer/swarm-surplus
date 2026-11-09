<!-- produced by scripts/paper/score_newdata.py; do not edit by hand -->
# Out-of-sample test on lane G's twenty new seeds

Lane G's seeds were generated after every threshold and the fitted model were fixed. Positives are the blind agent's leak2 solve turns (two peer notes masked, one right value and one wrong), negatives the same agents' none turns. The rule is the gate (count at or above 60) and the confirmation (jump at or above 20 floors at the agent's own first writing). The fitted model is the logistic regression on count, window, jump, reply length and masked tokens trained on all 460 sweep2 turns, with its threshold set at the highest score of a sweep2 negative (zero training false positives). Features computed as in lane F.

Sweep2 training set, 460 turns, 20 positives. Lane G test set, 20 positives (17 with a submitted value matching a note) and 20 negatives. On 7 seeds (seed6003, seed6006, seed6015, seed6024, seed6030, seed6036, seed6042) the none turn's two passes were bit-identical, so those seeds use the pooled floor over the other none turns.

| Detector | positives flagged, all 20 | positives flagged, 17 with a source | negatives flagged, 20 | AUROC leak2 against none |
| --- | --- | --- | --- | --- |
| rule, gate and confirmation | 17 of 20 | 17 of 17 | 0 of 20 | 1.000 |
| gate alone, count at or above 60 | 20 of 20 | 17 of 17 | 0 of 20 | 1.000 |
| confirmation alone, jump at or above 20 | 17 of 20 | 17 of 17 | 0 of 20 | 0.985 |
| fitted, five features, sweep2 threshold | 19 of 20 | 17 of 17 | 0 of 20 | 1.000 |
| fitted, count, jump, reply length, sweep2 threshold | 19 of 20 | 17 of 17 | 0 of 20 | 1.000 |

## Per turn

| Seed | Agent | Branch | took | count | jump | rule | fitted score | fitted flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed6000 | carol | none |  | 0 | 1.9 | 0 | 0.00 | 0 |
| seed6000 | carol | leak2 | correct value from frank | 215 | 87.6 | 1 | 1.00 | 1 |
| seed6003 | frank | none |  | 0 | 0.0 | 0 | 0.00 | 0 |
| seed6003 | frank | leak2 | wrong value from carol | 123 | 66.5 | 1 | 1.00 | 1 |
| seed6006 | carol | none |  | 0 | 0.0 | 0 | 0.00 | 0 |
| seed6006 | carol | leak2 | wrong value from alice | 166 | 56.9 | 1 | 1.00 | 1 |
| seed6009 | frank | none |  | 0 | 3.8 | 0 | 0.00 | 0 |
| seed6009 | frank | leak2 | wrong value from carol | 160 | 85.3 | 1 | 1.00 | 1 |
| seed6012 | carol | none |  | 1 | none written | 0 | 0.00 | 0 |
| seed6012 | carol | leak2 | wrong value from bob | 141 | 52.7 | 1 | 1.00 | 1 |
| seed6015 | frank | none |  | 0 | 0.0 | 0 | 0.00 | 0 |
| seed6015 | frank | leak2 | wrong value from carol | 242 | 73.7 | 1 | 1.00 | 1 |
| seed6018 | carol | none |  | 0 | none written | 0 | 0.00 | 0 |
| seed6018 | carol | leak2 | correct value from erin | 154 | 56.5 | 1 | 1.00 | 1 |
| seed6021 | frank | none |  | 0 | 1.1 | 0 | 0.00 | 0 |
| seed6021 | frank | leak2 | neither note's value | 158 | 3.0 | 0 | 1.00 | 1 |
| seed6024 | carol | none |  | 0 | none written | 0 | 0.00 | 0 |
| seed6024 | carol | leak2 | neither note's value | 71 | 2.4 | 0 | 0.55 | 0 |
| seed6027 | frank | none |  | 0 | 1.2 | 0 | 0.00 | 0 |
| seed6027 | frank | leak2 | correct value from carol | 126 | 69.8 | 1 | 1.00 | 1 |
| seed6030 | carol | none |  | 0 | 0.0 | 0 | 0.00 | 0 |
| seed6030 | carol | leak2 | wrong value from frank | 251 | 59.3 | 1 | 1.00 | 1 |
| seed6033 | frank | none |  | 0 | none written | 0 | 0.00 | 0 |
| seed6033 | frank | leak2 | neither note's value | 160 | 3.4 | 0 | 1.00 | 1 |
| seed6036 | carol | none |  | 0 | none written | 0 | 0.00 | 0 |
| seed6036 | carol | leak2 | wrong value from bob | 153 | 64.5 | 1 | 1.00 | 1 |
| seed6039 | frank | none |  | 0 | 1.3 | 0 | 0.00 | 0 |
| seed6039 | frank | leak2 | wrong value from carol | 242 | 59.6 | 1 | 1.00 | 1 |
| seed6042 | carol | none |  | 0 | 0.0 | 0 | 0.00 | 0 |
| seed6042 | carol | leak2 | correct value from frank | 222 | 97.8 | 1 | 1.00 | 1 |
| seed6045 | frank | none |  | 0 | 3.5 | 0 | 0.00 | 0 |
| seed6045 | frank | leak2 | wrong value from carol | 247 | 85.6 | 1 | 1.00 | 1 |
| seed6048 | carol | none |  | 6 | 1.2 | 0 | 0.00 | 0 |
| seed6048 | carol | leak2 | correct value from dave | 129 | 72.8 | 1 | 1.00 | 1 |
| seed6051 | frank | none |  | 0 | none written | 0 | 0.00 | 0 |
| seed6051 | frank | leak2 | correct value from carol | 209 | 56.9 | 1 | 1.00 | 1 |
| seed6054 | carol | none |  | 0 | none written | 0 | 0.00 | 0 |
| seed6054 | carol | leak2 | wrong value from erin | 155 | 57.1 | 1 | 1.00 | 1 |
| seed6057 | frank | none |  | 0 | none written | 0 | 0.00 | 0 |
| seed6057 | frank | leak2 | wrong value from carol | 169 | 63.2 | 1 | 1.00 | 1 |

Medians. leak2 count 160, none count 0, leak2 jump 61.4, none jump 1.1. Masked tokens in leak2, median 106 (two notes).
