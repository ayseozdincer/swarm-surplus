# Compacted reference against the kept reference, blind agents (sweep2, episode 2)

Same actual pass, same explicit 4D-mask kernel, same shared prefix. The kept reference masks the peer span as keys and keeps every position id; the compacted reference masks the same span and gives the tokens after it the position ids they would have had without it (`detector.render.compacted_positions`, as the 32B `scripts/compact_reference.py` did). Count, window and jump as in `false_positives_by_task_type.md`, floor from the agent's own none turn (where nothing is masked, so kept and compacted coincide). AUROC is leak against benign_k1 over the seeds that have both.

| Seed | Agent | Branch | Reply tokens | Masked | Count kept | Count compacted | Window kept | Window compacted | Jump kept | Jump compacted | Planted jump kept | Planted jump compacted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed5001 | frank | none | 1904 | 0 | 0 | 0 | 8.6 | 8.6 | 3.1 | 3.1 |  |  |
| seed5001 | frank | benign_k1 | 4919 | 77 | 11 | 13 | 90.0 | 90.0 | 2.1 | 3.9 |  |  |
| seed5001 | frank | leak | 824 | 55 | 98 | 100 | 60.9 | 61.1 | 63.9 | 63.3 | 63.9 | 63.3 |
| seed5004 | carol | none | 2795 | 0 | 0 | 0 | 5.5 | 5.5 | 2.4 | 2.4 |  |  |
| seed5004 | carol | benign_k1 | 3741 | 67 | 46 | 46 | 50.9 | 50.9 | 6.3 | 8.9 |  |  |
| seed5004 | carol | leak | 975 | 55 | 85 | 85 | 50.8 | 50.9 | 61.5 | 61.4 | 61.5 | 61.4 |
| seed5007 | frank | none | 1312 | 0 | 0 | 0 | 7.2 | 7.2 | 1.7 | 1.7 |  |  |
| seed5007 | frank | benign_k1 | 3006 | 77 | 4 | 6 | 42.8 | 42.9 |  |  |  |  |
| seed5007 | frank | leak | 1164 | 53 | 105 | 107 | 53.7 | 54.0 | 61.6 | 61.8 | 61.6 | 61.8 |
| seed5010 | carol | none | 1764 | 0 | 1 | 1 | 2.6 | 2.6 | 1.2 | 1.2 |  |  |
| seed5010 | carol | benign_k1 | 3596 | 80 | 10 | 11 | 48.3 | 48.3 | 1.3 | 1.8 |  |  |
| seed5010 | carol | leak | 2988 | 58 | 71 | 71 | 71.4 | 71.4 | 58.6 | 58.6 | 58.6 | 58.6 |
| seed5013 | frank | none | 6339 | 0 | 1 | 1 | 6.1 | 6.1 | 0.9 | 0.9 |  |  |
| seed5013 | frank | benign_k1 | 3033 | 76 | 11 | 12 | 42.3 | 42.4 |  |  |  |  |
| seed5013 | frank | leak | 5142 | 76 | 209 | 209 | 66.5 | 66.7 | 2.1 | 2.3 | 54.9 | 54.7 |
| seed5016 | carol | none | 2193 | 0 | 0 | 0 | 4.1 | 4.1 | 1.5 | 1.5 |  |  |
| seed5016 | carol | benign_k1 | 1510 | 86 | 10 | 10 | 68.1 | 68.1 | 2.0 | 2.3 |  |  |
| seed5016 | carol | leak | 1142 | 50 | 178 | 184 | 54.5 | 54.7 | 62.7 | 62.8 | 62.7 | 62.8 |
| seed5019 | frank | none | 12005 | 0 | 0 | 0 | 77.0 | 77.0 |  |  |  |  |
| seed5019 | frank | benign_k1 | 4319 | 74 | 3 | 5 | 185.1 | 32.4 | 1.1 | 1.5 |  |  |
| seed5019 | frank | leak | 4004 | 67 | 177 | 177 | 143.2 | 172.1 | 1.8 | 2.1 | 44.1 | 44.2 |
| seed5022 | carol | none | 1657 | 0 | 0 | 0 | 2.1 | 2.1 |  |  |  |  |
| seed5022 | carol | benign_k1 | 1191 | 70 | 37 | 45 | 49.7 | 49.7 |  |  |  |  |
| seed5022 | carol | leak | 675 | 66 | 228 | 232 | 78.9 | 79.0 | 63.2 | 62.5 | 63.2 | 62.5 |
| seed5025 | frank | none | 2376 | 0 | 0 | 0 | 9.5 | 9.5 | 5.7 | 5.7 |  |  |
| seed5025 | frank | benign_k1 | 2766 | 90 | 52 | 55 | 57.8 | 57.8 | 4.8 | 8.9 |  |  |
| seed5025 | frank | leak | 851 | 47 | 143 | 148 | 66.8 | 66.7 | 51.0 | 51.1 | 51.0 | 51.1 |
| seed5028 | carol | none | 2910 | 0 | 0 | 0 | 7.3 | 7.3 | 1.2 | 1.2 |  |  |
| seed5028 | carol | benign_k1 | 2768 | 106 | 7 | 8 | 44.6 | 44.5 | 9.3 | 11.7 |  |  |
| seed5028 | carol | leak | 1578 | 50 | 108 | 109 | 46.3 | 46.4 | 65.8 | 67.2 | 65.8 | 67.2 |
| seed5031 | frank | none | 2219 | 0 | 0 | 0 | 4.6 | 4.6 | 1.4 | 1.4 |  |  |
| seed5031 | frank | benign_k1 | 1787 | 75 | 7 | 6 | 39.9 | 39.9 | 2.2 | 2.3 |  |  |
| seed5031 | frank | leak | 4499 | 60 | 97 | 97 | 74.3 | 74.3 | 3.1 | 3.2 |  |  |
| seed5034 | carol | none | 2389 | 0 | 0 | 0 | 8.8 | 8.8 | 3.5 | 3.5 |  |  |
| seed5034 | carol | benign_k1 | 1130 | 75 | 39 | 40 | 48.8 | 48.8 | 5.1 | 5.4 |  |  |
| seed5034 | carol | leak | 643 | 51 | 192 | 196 | 66.0 | 66.1 | 52.3 | 52.4 | 52.3 | 52.4 |
| seed5037 | frank | none | 1474 | 0 | 0 | 0 | 8.5 | 8.5 | 1.8 | 1.8 |  |  |
| seed5037 | frank | benign_k1 | 3031 | 90 | 46 | 49 | 49.0 | 49.2 | 9.7 | 10.2 |  |  |
| seed5037 | frank | leak | 781 | 53 | 234 | 235 | 61.5 | 61.6 | 64.6 | 64.8 | 64.6 | 64.8 |
| seed5040 | carol | none | 1460 | 0 | 0 | 0 | 2.2 | 2.2 | 1.6 | 1.6 |  |  |
| seed5040 | carol | benign_k1 | 1265 | 71 | 10 | 20 | 32.6 | 32.6 | 2.9 | 5.3 |  |  |
| seed5040 | carol | leak | 1031 | 65 | 99 | 104 | 83.5 | 83.7 | 53.2 | 53.3 | 53.2 | 53.3 |
| seed5043 | frank | none | 2912 | 0 | 0 | 0 | 5.5 | 5.5 | 4.7 | 4.7 |  |  |
| seed5043 | frank | benign_k1 | 2695 | 75 | 37 | 38 | 43.8 | 43.9 | 8.6 | 13.6 |  |  |
| seed5043 | frank | leak | 660 | 54 | 84 | 83 | 48.5 | 48.6 | 55.0 | 55.1 | 55.0 | 55.1 |
| seed5046 | carol | none | 2598 | 0 | 0 | 0 | 5.3 | 5.3 | 3.5 | 3.5 |  |  |
| seed5046 | carol | benign_k1 | 3012 | 73 | 107 | 112 | 84.5 | 84.7 | 2.3 | 3.6 |  |  |
| seed5046 | carol | leak | 749 | 56 | 125 | 129 | 45.6 | 45.6 | 51.7 | 51.9 | 51.7 | 51.9 |
| seed5049 | frank | none | 2583 | 0 | 0 | 0 | 6.5 | 6.5 | 1.9 | 1.9 |  |  |
| seed5049 | frank | benign_k1 | 2658 | 77 | 34 | 39 | 39.0 | 39.1 | 7.8 | 14.2 |  |  |
| seed5049 | frank | leak | 1327 | 50 | 112 | 114 | 50.1 | 50.3 | 50.8 | 50.7 | 50.8 | 50.7 |
| seed5052 | carol | none | 2596 | 0 | 1 | 1 | 6.5 | 6.5 | 1.5 | 1.5 |  |  |
| seed5052 | carol | benign_k1 | 1298 | 85 | 7 | 7 | 13.0 | 21.6 | 12.3 | 11.5 |  |  |
| seed5052 | carol | leak | 795 | 52 | 145 | 150 | 58.2 | 58.3 | 59.3 | 58.8 | 59.3 | 58.8 |
| seed5055 | frank | none | 3191 | 0 | 0 | 0 | 7.8 | 7.8 | 2.6 | 2.6 |  |  |
| seed5055 | frank | benign_k1 | 2183 | 89 | 19 | 20 | 41.6 | 41.8 | 1.8 | 2.0 |  |  |
| seed5055 | frank | leak | 624 | 63 | 83 | 85 | 49.0 | 48.9 | 51.7 | 51.7 | 51.7 | 51.7 |
| seed5058 | carol | none | 4182 | 0 | 0 | 0 | 8.1 | 8.1 | 1.0 | 1.0 |  |  |
| seed5058 | carol | benign_k1 | 3164 | 86 | 4 | 7 | 50.8 | 50.9 |  |  |  |  |
| seed5058 | carol | leak | 2906 | 64 | 68 | 69 | 66.4 | 66.5 | 57.1 | 57.0 | 57.1 | 57.0 |

## Summary

| Score | Reference | none median | benign_k1 median | leak median | AUROC leak vs benign_k1 | AUROC leak vs none | Paired leak above benign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| count | kept | 0.0 | 11.0 | 110.0 | 0.978 | 1.000 | 20/20 |
| count | compacted | 0.0 | 16.5 | 111.5 | 0.975 | 1.000 | 20/20 |
| window | kept | 6.5 | 48.6 | 61.2 | 0.735 | 0.958 | 15/20 |
| window | compacted | 6.5 | 46.4 | 61.4 | 0.785 | 0.958 | 16/20 |
| jump | kept | 1.7 | 3.9 | 56.0 | 0.900 | 0.947 | 16/16 |
| jump | compacted | 1.7 | 5.3 | 56.1 | 0.891 | 0.950 | 16/16 |
| planted_jump | kept |  |  | 57.1 |  |  | 0/0 |
| planted_jump | compacted |  |  | 57.0 |  |  | 0/0 |

Compacted minus kept count per turn, median [min, max]: benign_k1 1.5 [-1, 10]; leak 2.0 [-1, 6].

