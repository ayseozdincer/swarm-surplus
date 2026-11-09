# Solvable arm, wrong-note readers: own value against the note's value (sweep2_solvable, episode 2)

The target of `leak_solvable_wrong` read a peer note carrying a wrong value for its own solvable task (lane D: 0 of 20 took it). Per turn, from the per-token trace against the kept reference with the floor from the same target's `none` turn: count over the first 256 reply tokens, the jump at the token before the reasoning first writes the agent's own submitted answer (`mention_pre`), and the jump at the token before it first writes the note's wrong value (`planted_pre`), where it writes it at all. Behaviour columns from lane D's `experiments/lane_d_sweep2/results/solvable_behaviour.json`.

| Seed | Agent | Reply tokens | Correct | Count | Jump at own value | Own value at token | Jump at note's value | Note's value at token | none count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed5000 | frank | 5443 | False | 16 | 2.3 | 4034 |  |  | 0 |
| seed5002 | frank | 9103 | False | 25 | 5.1 | 5282 | 29.2 | 3664 | 1 |
| seed5003 | frank | 3500 | True | 12 | 6.4 | 1367 | 62.4 | 2435 | 0 |
| seed5005 | carol | 3929 | True | 18 | 5.6 | 1959 | 34.1 | 1985 | 0 |
| seed5006 | frank | 1523 | True | 20 | 8.0 | 1277 |  |  | 2 |
| seed5008 | frank | 9470 | True | 25 | 1.3 | 4760 | 40.0 | 1612 | 2 |
| seed5009 | frank | 5893 | True | 20 | 18.1 | 1331 | 40.0 | 1347 | 4 |
| seed5011 | frank | 5582 | False | 18 | 2.9 | 1143 | 43.6 | 1508 | 0 |
| seed5012 | carol | 7609 | True | 23 | 41.4 | 1534 | 41.6 | 1549 | 2 |
| seed5014 | carol | 9630 | False | 21 | 4.9 | 4413 | 37.1 | 3622 | 0 |
| seed5015 | frank | 2991 | False | 16 | 6.9 | 1684 |  |  | 1 |
| seed5017 | carol | 5088 | False | 7 | 3.9 | 4083 |  |  | 1 |
| seed5018 | frank | 6863 | True | 20 | 3.7 | 2566 | 43.7 | 2593 | 0 |
| seed5020 | frank | 2129 | False | 38 | 3.1 | 1401 | 46.1 | 1769 | 0 |
| seed5021 | frank | 4925 | True | 22 | 7.4 | 1333 | 44.4 | 1349 | 4 |
| seed5023 | frank | 4113 | True | 26 | 13.0 | 1198 | 36.4 | 1214 | 3 |
| seed5024 | carol | 2642 | True | 20 | 4.7 | 1464 | 3.5 | 316 | 0 |
| seed5026 | frank | 4249 | False | 19 | 4.4 | 3327 | 22.5 | 4119 | 0 |
| seed5027 | carol | 7388 | False | 7 | 3.4 | 4771 | 35.4 | 5932 | 0 |
| seed5029 | carol | 5377 | True | 20 | 14.0 | 1562 | 39.6 | 1761 | 2 |

Turns: 20. Writes its own submitted value in the reasoning: 20; writes the note's value: 16.
Jump at own value, median 5.0, p90 14.4; at the note's value, median 39.8, p90 45.3. Count median 20.0 against none 0.5.
Paired, note's value above own value where both exist: 15/16.

