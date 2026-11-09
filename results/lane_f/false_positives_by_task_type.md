# False positives of the per-token trace by task type and role (sweep2, episode 2, Qwen3-14B)

Replay of every agent's solve turn at episode 2 in the branches below (`scripts/pod/replay_lane_f.py`), actual pass against the kept reference (peer messages masked as keys, position ids kept) on one explicit 4D-mask kernel from a shared prefix, the per-token relative surplus of the whole reply. Scores (`scripts/lane_f_scores.py`):

- **floor**, per lens layer: the median over the reply tokens of the same agent's `none` turn in the same seed, whose second pass is the plain kernel (kernel noise only).
- **series**: relative surplus over that floor, mean over the band L16 to L38.
- **count**: reply tokens among the first 256 whose series exceeds 10 (lane C's and lane D's plateau count).
- **window**: the maximum 16-token sliding mean of the series over the whole reply.
- **jump**: the series at the token before the reasoning first writes the submitted answer (`mention_pre`); turns that never write it in the reasoning, or submit nothing, have no jump.
- **planted jump** (blind only): the same at the token before the reasoning first writes the planted value.
- **role**: blind is the leak's target (carol or frank); reader is a non-blind agent whose solve turn holds a peer note (four a seed in k1, five in k3 since an author of one note reads the other two); nonblind_unmasked is a non-blind agent with nothing to mask, the `none` branch or the author of the only note, so its surplus is kernel noise against a floor from another of its own turns.
- **task type** from the agent's specialty: alice and dave records, bob and erin data, carol and frank code. The blind agent is always code on these seeds.

Seeds: 20. Turns per role and branch: blind benign_k1 20, blind leak 20, blind none 20, nonblind_unmasked benign_k1 20, nonblind_unmasked leak 20, nonblind_unmasked none 100, reader benign_k1 80, reader benign_k3 100, reader leak 80.

## Count, first 256 reply tokens (fraction of turns above each cut)

| Role | Task type | Branch | n | count median | p90 | max | > 20 | > 40 | > 60 | > 80 | > 100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blind | all | none | 20 | 0.0 | 1.0 | 1 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| blind | all | benign_k1 | 20 | 11.0 | 46.6 | 107 | 8/20 | 4/20 | 1/20 | 1/20 | 1/20 |
| blind | all | leak | 20 | 110.0 | 210.9 | 234 | 20/20 | 20/20 | 20/20 | 18/20 | 12/20 |
| blind | code | none | 20 | 0.0 | 1.0 | 1 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| blind | code | benign_k1 | 20 | 11.0 | 46.6 | 107 | 8/20 | 4/20 | 1/20 | 1/20 | 1/20 |
| blind | code | leak | 20 | 110.0 | 210.9 | 234 | 20/20 | 20/20 | 20/20 | 18/20 | 12/20 |
| reader | all | benign_k1 | 80 | 3.0 | 10.1 | 35 | 1/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| reader | all | leak | 80 | 3.0 | 11.1 | 26 | 2/80 | 0/80 | 0/80 | 0/80 | 0/80 |
| reader | all | benign_k3 | 100 | 3.0 | 10.1 | 40 | 3/100 | 0/100 | 0/100 | 0/100 | 0/100 |
| reader | records | benign_k1 | 29 | 5.0 | 11.0 | 17 | 0/29 | 0/29 | 0/29 | 0/29 | 0/29 |
| reader | records | leak | 40 | 6.0 | 15.2 | 26 | 2/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| reader | records | benign_k3 | 40 | 6.0 | 12.2 | 23 | 1/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| reader | data | benign_k1 | 33 | 1.0 | 3.8 | 5 | 0/33 | 0/33 | 0/33 | 0/33 | 0/33 |
| reader | data | leak | 40 | 2.0 | 4.1 | 12 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| reader | data | benign_k3 | 40 | 2.0 | 4.1 | 8 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| reader | code | benign_k1 | 18 | 5.0 | 16.2 | 35 | 1/18 | 0/18 | 0/18 | 0/18 | 0/18 |
| reader | code | benign_k3 | 20 | 5.0 | 14.7 | 40 | 2/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| nonblind_unmasked | all | none | 100 | 0.0 | 2.0 | 7 | 0/100 | 0/100 | 0/100 | 0/100 | 0/100 |
| nonblind_unmasked | all | benign_k1 | 20 | 0.0 | 1.1 | 2 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| nonblind_unmasked | all | leak | 20 | 0.0 | 2.1 | 6 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| nonblind_unmasked | records | none | 40 | 0.0 | 1.0 | 3 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| nonblind_unmasked | records | benign_k1 | 11 | 0.0 | 0.0 | 0 | 0/11 | 0/11 | 0/11 | 0/11 | 0/11 |
| nonblind_unmasked | data | none | 40 | 0.0 | 1.1 | 3 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| nonblind_unmasked | data | benign_k1 | 7 | 0.0 | 1.4 | 2 | 0/7 | 0/7 | 0/7 | 0/7 | 0/7 |
| nonblind_unmasked | code | none | 20 | 0.0 | 4.2 | 7 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| nonblind_unmasked | code | benign_k1 | 2 | 1.5 | 1.9 | 2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| nonblind_unmasked | code | leak | 20 | 0.0 | 2.1 | 6 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |

## Jump at the first writing of the submitted answer (fraction above each cut)

| Role | Task type | Branch | n | jump median | p90 | max | > 10 | > 20 | > 30 | > 50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blind | all | none | 18 | 1.7 | 3.8 | 5.7 | 0/18 | 0/18 | 0/18 | 0/18 |
| blind | all | benign_k1 | 16 | 3.9 | 9.5 | 12.3 | 1/16 | 0/16 | 0/16 | 0/16 |
| blind | all | leak | 20 | 56.0 | 63.9 | 65.8 | 17/20 | 17/20 | 17/20 | 17/20 |
| blind | code | none | 18 | 1.7 | 3.8 | 5.7 | 0/18 | 0/18 | 0/18 | 0/18 |
| blind | code | benign_k1 | 16 | 3.9 | 9.5 | 12.3 | 1/16 | 0/16 | 0/16 | 0/16 |
| blind | code | leak | 20 | 56.0 | 63.9 | 65.8 | 17/20 | 17/20 | 17/20 | 17/20 |
| reader | all | benign_k1 | 80 | 2.3 | 8.7 | 32.4 | 5/80 | 3/80 | 1/80 | 0/80 |
| reader | all | leak | 80 | 2.5 | 5.7 | 25.8 | 5/80 | 1/80 | 0/80 | 0/80 |
| reader | all | benign_k3 | 99 | 2.4 | 6.4 | 19.1 | 3/99 | 0/99 | 0/99 | 0/99 |
| reader | records | benign_k1 | 29 | 2.5 | 4.6 | 9.4 | 0/29 | 0/29 | 0/29 | 0/29 |
| reader | records | leak | 40 | 2.5 | 3.8 | 6.1 | 0/40 | 0/40 | 0/40 | 0/40 |
| reader | records | benign_k3 | 39 | 2.8 | 3.9 | 10.5 | 1/39 | 0/39 | 0/39 | 0/39 |
| reader | data | benign_k1 | 33 | 2.2 | 11.4 | 32.4 | 4/33 | 3/33 | 1/33 | 0/33 |
| reader | data | leak | 40 | 2.4 | 10.7 | 25.8 | 5/40 | 1/40 | 0/40 | 0/40 |
| reader | data | benign_k3 | 40 | 2.1 | 6.5 | 19.1 | 2/40 | 0/40 | 0/40 | 0/40 |
| reader | code | benign_k1 | 18 | 1.4 | 4.7 | 12.6 | 1/18 | 0/18 | 0/18 | 0/18 |
| reader | code | benign_k3 | 20 | 1.6 | 8.5 | 8.9 | 0/20 | 0/20 | 0/20 | 0/20 |
| nonblind_unmasked | all | none | 100 | 1.1 | 2.1 | 5.9 | 0/100 | 0/100 | 0/100 | 0/100 |
| nonblind_unmasked | all | benign_k1 | 20 | 1.3 | 2.8 | 4.8 | 0/20 | 0/20 | 0/20 | 0/20 |
| nonblind_unmasked | all | leak | 20 | 1.0 | 1.8 | 13.7 | 1/20 | 0/20 | 0/20 | 0/20 |
| nonblind_unmasked | records | none | 40 | 1.1 | 1.4 | 1.8 | 0/40 | 0/40 | 0/40 | 0/40 |
| nonblind_unmasked | records | benign_k1 | 11 | 1.4 | 2.3 | 3.4 | 0/11 | 0/11 | 0/11 | 0/11 |
| nonblind_unmasked | data | none | 40 | 1.1 | 3.2 | 5.9 | 0/40 | 0/40 | 0/40 | 0/40 |
| nonblind_unmasked | data | benign_k1 | 7 | 1.0 | 3.6 | 4.8 | 0/7 | 0/7 | 0/7 | 0/7 |
| nonblind_unmasked | code | none | 20 | 1.0 | 1.6 | 2.3 | 0/20 | 0/20 | 0/20 | 0/20 |
| nonblind_unmasked | code | benign_k1 | 2 | 1.3 | 1.6 | 1.7 | 0/2 | 0/2 | 0/2 | 0/2 |
| nonblind_unmasked | code | leak | 20 | 1.0 | 1.8 | 13.7 | 1/20 | 0/20 | 0/20 | 0/20 |

## 16-token window maximum

| Role | Task type | Branch | n | window median | p90 | max | > 20 | > 40 | > 60 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blind | all | none | 20 | 6.5 | 8.9 | 77.0 | 1/20 | 1/20 | 1/20 |
| blind | all | benign_k1 | 20 | 48.6 | 85.1 | 185.1 | 19/20 | 16/20 | 4/20 |
| blind | all | leak | 20 | 61.2 | 79.3 | 143.2 | 20/20 | 20/20 | 11/20 |
| blind | code | none | 20 | 6.5 | 8.9 | 77.0 | 1/20 | 1/20 | 1/20 |
| blind | code | benign_k1 | 20 | 48.6 | 85.1 | 185.1 | 19/20 | 16/20 | 4/20 |
| blind | code | leak | 20 | 61.2 | 79.3 | 143.2 | 20/20 | 20/20 | 11/20 |
| reader | all | benign_k1 | 80 | 7.8 | 17.1 | 276.0 | 8/80 | 8/80 | 7/80 |
| reader | all | leak | 80 | 8.0 | 13.3 | 123.8 | 5/80 | 2/80 | 2/80 |
| reader | all | benign_k3 | 100 | 7.9 | 13.8 | 270.8 | 7/100 | 6/100 | 5/100 |
| reader | records | benign_k1 | 29 | 7.9 | 13.0 | 220.2 | 2/29 | 2/29 | 2/29 |
| reader | records | leak | 40 | 8.6 | 11.7 | 26.9 | 1/40 | 0/40 | 0/40 |
| reader | records | benign_k3 | 40 | 8.0 | 12.6 | 69.9 | 1/40 | 1/40 | 1/40 |
| reader | data | benign_k1 | 33 | 6.1 | 9.2 | 227.5 | 1/33 | 1/33 | 1/33 |
| reader | data | leak | 40 | 7.3 | 17.8 | 123.8 | 4/40 | 2/40 | 2/40 |
| reader | data | benign_k3 | 40 | 6.5 | 10.2 | 248.2 | 3/40 | 3/40 | 3/40 |
| reader | code | benign_k1 | 18 | 11.4 | 179.6 | 276.0 | 5/18 | 5/18 | 4/18 |
| reader | code | benign_k3 | 20 | 11.3 | 37.1 | 270.8 | 3/20 | 2/20 | 1/20 |
| nonblind_unmasked | all | none | 100 | 3.4 | 8.2 | 139.5 | 3/100 | 3/100 | 2/100 |
| nonblind_unmasked | all | benign_k1 | 20 | 3.0 | 4.9 | 6.4 | 0/20 | 0/20 | 0/20 |
| nonblind_unmasked | all | leak | 20 | 6.9 | 108.6 | 225.9 | 3/20 | 3/20 | 3/20 |
| nonblind_unmasked | records | none | 40 | 2.2 | 3.8 | 4.4 | 0/40 | 0/40 | 0/40 |
| nonblind_unmasked | records | benign_k1 | 11 | 2.4 | 3.8 | 5.5 | 0/11 | 0/11 | 0/11 |
| nonblind_unmasked | data | none | 40 | 3.8 | 8.8 | 139.5 | 3/40 | 3/40 | 2/40 |
| nonblind_unmasked | data | benign_k1 | 7 | 3.2 | 4.0 | 4.6 | 0/7 | 0/7 | 0/7 |
| nonblind_unmasked | code | none | 20 | 6.3 | 8.8 | 10.1 | 0/20 | 0/20 | 0/20 |
| nonblind_unmasked | code | benign_k1 | 2 | 5.6 | 6.3 | 6.4 | 0/2 | 0/2 | 0/2 |
| nonblind_unmasked | code | leak | 20 | 6.9 | 108.6 | 225.9 | 3/20 | 3/20 | 3/20 |

## Readers against the blind agent, same branch (AUROC blind above readers)

| Branch | Score | Blind median | Reader median | AUROC |
| --- | --- | --- | --- | --- |
| benign_k1 | count | 11.0 | 3.0 | 0.882 |
| benign_k1 | jump | 3.9 | 2.3 | 0.620 |
| benign_k1 | window | 48.6 | 7.8 | 0.905 |
| leak | count | 110.0 | 3.0 | 1.000 |
| leak | jump | 56.0 | 2.5 | 0.903 |
| leak | window | 61.2 | 8.0 | 0.976 |

## Pooled floor (lane D's definition: the median over every none turn of the role) instead of the agent's own

| Role | Branch | n | count median (own floor) | count median (pooled) | > 60 own | > 60 pooled | jump median own | jump median pooled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blind | none | 20 | 0.0 | 0.0 | 0/20 | 0/20 | 1.7 | 1.5 |
| blind | benign_k1 | 20 | 11.0 | 9.5 | 1/20 | 1/20 | 3.9 | 3.6 |
| blind | leak | 20 | 110.0 | 104.0 | 20/20 | 20/20 | 56.0 | 51.7 |
| reader | benign_k1 | 80 | 3.0 | 4.0 | 0/80 | 0/80 | 2.3 | 2.4 |
| reader | leak | 80 | 3.0 | 3.0 | 0/80 | 0/80 | 2.5 | 2.5 |
| reader | benign_k3 | 100 | 3.0 | 4.0 | 0/100 | 0/100 | 2.4 | 2.5 |

Readers in `leak` whose reasoning writes the leaked value: 7 of 80.

