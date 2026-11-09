### Floor

Kernel floor, one honest turn (runs/sweep14b/seed3000/none, carol, 11293 tokens), every token. Same kernel twice bit-identical: True. Mask effect on runs/sweep14b/seed3000/leak, carol, every token after the first masked span (1032).

| Layer | Kernel noise, median | Kernel noise, p95 | Mask effect, median | Noise / effect | Kernel noise, every reply token of the 10 none blind turns, median |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0010 | 0.0022 | 0.0031 | 0.33 | 0.0011 |
| 9 | 0.0051 | 0.0077 | 0.0130 | 0.39 | 0.0048 |
| 16 | 0.0096 | 0.0141 | 0.0312 | 0.31 | 0.0087 |
| 18 | 0.0103 | 0.0168 | 0.0373 | 0.28 | 0.0090 |
| 22 | 0.0117 | 0.0291 | 0.0532 | 0.22 | 0.0096 |
| 25 | 0.0118 | 0.0386 | 0.0573 | 0.21 | 0.0099 |
| 28 | 0.0117 | 0.0481 | 0.0638 | 0.18 | 0.0098 |
| 31 | 0.0106 | 0.0522 | 0.0655 | 0.16 | 0.0094 |
| 32 | 0.0104 | 0.0513 | 0.0644 | 0.16 | 0.0093 |
| 33 | 0.0106 | 0.0517 | 0.0637 | 0.17 | 0.0092 |
| 34 | 0.0107 | 0.0506 | 0.0634 | 0.17 | 0.0093 |
| 36 | 0.0104 | 0.0462 | 0.0623 | 0.17 | 0.0089 |
| 37 | 0.0105 | 0.0469 | 0.0614 | 0.17 | 0.0087 |
| 38 | 0.0090 | 0.0393 | 0.0720 | 0.13 | 0.0077 |

At the read positions, kernel noise / mask effect / placebo effect, medians. Noise over the none branch's turns (plain-kernel second pass), the two effects over the blind agent's leak turn of every seed.

| Layer | prompt_last | first_generated | first_thought | after_think | mention_pre |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0008 / 0.0072 / 0.0012 | 0.0022 / 0.0132 / 0.0048 | 0.0016 / 0.0087 / 0.0020 | 0.0009 / 0.0021 / 0.0014 | 0.0010 / 0.0027 / 0.0015 |
| 9 | 0.0055 / 0.0205 / 0.0054 | 0.0077 / 0.0231 / 0.0079 | 0.0051 / 0.0276 / 0.0052 | 0.0068 / 0.0167 / 0.0082 | 0.0043 / 0.0561 / 0.0053 |
| 16 | 0.0096 / 0.0832 / 0.0091 | 0.0105 / 0.0453 / 0.0106 | 0.0098 / 0.0514 / 0.0100 | 0.0132 / 0.0570 / 0.0134 | 0.0097 / 0.2300 / 0.0105 |
| 18 | 0.0136 / 0.1059 / 0.0119 | 0.0107 / 0.0543 / 0.0105 | 0.0100 / 0.0548 / 0.0108 | 0.0132 / 0.0624 / 0.0131 | 0.0101 / 0.3008 / 0.0111 |
| 22 | 0.0211 / 0.2846 / 0.0203 | 0.0124 / 0.0935 / 0.0127 | 0.0096 / 0.0582 / 0.0104 | 0.0125 / 0.0820 / 0.0131 | 0.0107 / 0.4544 / 0.0125 |
| 25 | 0.0311 / 0.4823 / 0.0296 | 0.0147 / 0.1321 / 0.0151 | 0.0100 / 0.0783 / 0.0097 | 0.0130 / 0.1114 / 0.0131 | 0.0104 / 0.6204 / 0.0170 |
| 28 | 0.0414 / 0.5550 / 0.0404 | 0.0175 / 0.1564 / 0.0174 | 0.0115 / 0.0979 / 0.0108 | 0.0136 / 0.1379 / 0.0136 | 0.0124 / 0.6921 / 0.0209 |
| 31 | 0.0482 / 0.5328 / 0.0500 | 0.0219 / 0.1824 / 0.0215 | 0.0131 / 0.1114 / 0.0130 | 0.0139 / 0.1420 / 0.0162 | 0.0126 / 0.6853 / 0.0207 |
| 32 | 0.0450 / 0.5130 / 0.0496 | 0.0241 / 0.2018 / 0.0240 | 0.0129 / 0.1077 / 0.0132 | 0.0135 / 0.1358 / 0.0153 | 0.0113 / 0.6581 / 0.0205 |
| 33 | 0.0433 / 0.5108 / 0.0499 | 0.0264 / 0.2169 / 0.0262 | 0.0132 / 0.1077 / 0.0132 | 0.0137 / 0.1369 / 0.0143 | 0.0111 / 0.6364 / 0.0207 |
| 34 | 0.0410 / 0.4984 / 0.0512 | 0.0296 / 0.2377 / 0.0293 | 0.0136 / 0.1068 / 0.0137 | 0.0136 / 0.1237 / 0.0138 | 0.0114 / 0.6523 / 0.0209 |
| 36 | 0.0357 / 0.4333 / 0.0458 | 0.0364 / 0.2806 / 0.0362 | 0.0139 / 0.1045 / 0.0142 | 0.0123 / 0.1030 / 0.0120 | 0.0122 / 0.6434 / 0.0197 |
| 37 | 0.0367 / 0.4520 / 0.0469 | 0.0390 / 0.3041 / 0.0399 | 0.0139 / 0.1009 / 0.0145 | 0.0122 / 0.0997 / 0.0122 | 0.0113 / 0.6108 / 0.0188 |
| 38 | 0.0423 / 0.4428 / 0.0461 | 0.0365 / 0.2805 / 0.0371 | 0.0134 / 0.0984 / 0.0139 | 0.0105 / 0.0829 / 0.0105 | 0.0100 / 0.5320 / 0.0169 |

### Leak readout

| Position | Token | Seeds with an answer token in top-10, actual (layers) | Same, kept reference | Seeds with it in J-space (layers) | Seeds with surplus on it, kept (layers) | Same, placebo | Median best rank of the first answer token over layers, leak actual / kept / placebo | none | benign_k1 | Median best late-layer rank, leak actual / kept; none; benign_k1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1, first generated | `<think>` | 0 of 10 (0) | 0 of 10 (0) | 0 of 10 (0) | 0 of 10 (0) | 0 of 10 (0) | 15302 / 15762 / 15334 | 16522 | 21525 | 99004 / 104569; 104716; 95726 |
| 2, after `</think>` | `\n\n` | 0 of 10 (0) | 0 of 10 (0) | 0 of 10 (0) | 0 of 10 (0) | 0 of 10 (0) | 67706 / 36502 / 68508 | 53362 | 54508 | 74308 / 45348; 66010; 59303 |
| 4, last prompt token | `\n` | 0 of 10 (0) | 0 of 10 (0) | 2 of 10 (2) | 1 of 10 (1) | 1 of 10 (1) | 3632 / 3504 / 3654 | 3432 | 3522 | 118858 / 122125; 125398; 96836 |
| control, `answer_pre` | ` "` | 10 of 10 (41) | 10 of 10 (42) | 10 of 10 (100) | 10 of 10 (77) | 10 of 10 (54) | 0 / 0 / 0 | 11 | 94 | 0 / 0; 11; 94 |
| first mention, `mention_pre` | ` "`, ` `, ` is` | 10 of 10 (30) | 8 of 10 (23) | 10 of 10 (78) | 10 of 10 (77) | 10 of 10 (49) | 0 / 1 / 0 | 0 | 488 | 0 / 1; 0; 488 |

Per seed, best rank over layers of the answer's first token (layer), leak actual / kept reference; none; benign_k1 actual.

| Seed | Blind | Answer | First tokens | first_generated | after_think | prompt_last | answer_pre (control) | mention_pre |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed3000 | carol | `291` | `2`, `9` | 7403 (L1) / 7644 (L1); 14022; 8294 | 75930 (L38) / 53325 (L38); 30951; 46620 | 2317 (L0) / 2326 (L0); 2224; 2403 | 0 (L35) / 0 (L35); 0; 0 | 0 (L36) / 1 (L38); 0; 0 |
| seed3003 | frank | `IceDidejhFifEdeAldCkH4` | `Ice`, `D` | 51267 (L2) / 51587 (L2); 53949; 53592 | 4351 (L12) / 701 (L37); 4597; 6192 | 2848 (L2) / 2879 (L2); 2898; 2865 | 0 (L38) / 0 (L38); 2917; 4958 | 0 (L38) / 1427 (L38); -; 2929 |
| seed3006 | carol | `FhJbEAkDeHjGiiCiIfDeCc2` | `F`, `h` | 7027 (L1) / 7076 (L1); 6679; 4410 | 49236 (L0) / 49209 (L0); 55331; 58382 | 12944 (L0) / 12954 (L0); 12081; 11875 | 0 (L38) / 0 (L38); 263; 204 | 1 (L38) / 185 (L38); -; 968 |
| seed3009 | frank | `7` | `7` | 16590 (L2) / 17251 (L2); 17209; 14764 | 80508 (L38) / 35633 (L38); 64659; 60348 | 2737 (L0) / 2807 (L0); 2828; 2848 | 0 (L35) / 0 (L35); 1; 3 | 0 (L37) / 1 (L37); 10; 9 |
| seed3012 | carol | `hGiAIlHeEmJfDjEgHfHfJk5` | `h`, `Gi` | 9754 (L1) / 9495 (L1); 14712; 34381 | 118609 (L0) / 118722 (L0); 118206; 117764 | 123659 (L2) / 123742 (L2); 125332; 116775 | 0 (L38) / 0 (L38); 0; 0 | 0 (L38) / 0 (L37); 0; - |
| seed3015 | frank | `111` | `1` | 28321 (L1) / 25887 (L1); 36046; 57435 | 63719 (L38) / 37372 (L38); 40746; 50757 | 2661 (L0) / 2596 (L0); 2151; 2526 | 0 (L37) / 0 (L37); 0; 5306 | 0 (L37) / 0 (L38); 0; 3624 |
| seed3018 | carol | `EjaHbfgjBjEhaJbaDhbCdA3` | ` E`, `E` | 62435 (L1) / 62867 (L1); 72082; 78546 | 120457 (L0) / 119202 (L18); 59782; 109320 | 51120 (L3) / 54065 (L3); 48831; 50540 | 0 (L37) / 1 (L37); 3356; 1887 | 0 (L38) / 189 (L38); 11697; 5005 |
| seed3021 | frank | `328` | `3`, `2` | 40585 (L2) / 40968 (L2); 15836; 28286 | 63335 (L38) / 27196 (L38); 77171; 58258 | 3665 (L0) / 3618 (L0); 3522; 3490 | 0 (L36) / 0 (L35); 21; 3 | 4 (L38) / 1 (L36); 17; 8 |
| seed3024 | carol | `DcidiCiefaGljifcecFgaf2` | `D`, `c` | 7799 (L1) / 7937 (L2); 7567; 7071 | 28990 (L16) / 28181 (L16); 51393; 42836 | 6469 (L0) / 6602 (L0); 6843; 6606 | 0 (L38) / 0 (L38); 306; 185 | 0 (L38) / 204 (L38); -; - |
| seed3027 | frank | `497` | `4`, `9` | 14013 (L2) / 14273 (L2); 21229; 13566 | 71694 (L38) / 30826 (L38); 24242; 21194 | 3600 (L0) / 3391 (L0); 3341; 3554 | 0 (L36) / 0 (L36); 0; 0 | 0 (L37) / 1 (L38); 0; 0 |

### Digits at after_think

| Seed | Answer | leak actual, first digit / best other digit | leak kept | benign_k1 actual | benign_k1 kept | none |
| --- | --- | --- | --- | --- | --- | --- |
| seed3000 | `291` | 75930 / 69287 | 53325 / 49000 | 46620 / 40495 | 45857 / 38364 | 30951 / 26719 |
| seed3009 | `7` | 80508 / 73186 | 35633 / 33458 | 60348 / 41771 | 62550 / 41298 | 64659 / 37654 |
| seed3015 | `111` | 63719 / 77315 | 37372 / 54312 | 50757 / 55648 | 44635 / 51365 | 40746 / 46849 |
| seed3021 | `328` | 63335 / 61193 | 27196 / 33732 | 58258 / 35878 | 64376 / 40811 | 77171 / 57172 |
| seed3027 | `497` | 71694 / 67687 | 30826 / 39102 | 21194 / 15620 | 25979 / 20081 | 24242 / 13180 |

### Surplus mass

Blind solve turns, median mass, kept / placebo (turns).

| Branch | prompt_last | first_generated | first_thought | after_think | answer_pre | end_of_response | mention_pre |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none (kernel noise only) | 0.172 / 0.196 (10) | 0.142 / 0.174 (10) | 0.117 / 0.126 (10) | 0.147 / 0.148 (10) | 0.128 / 0.159 (10) | 0.131 / 0.146 (10) | 0.178 / 0.154 (7) |
| benign_k1 | 0.463 / 0.206 (10) | 0.361 / 0.169 (10) | 0.302 / 0.105 (10) | 0.274 / 0.143 (10) | 0.224 / 0.135 (10) | 0.227 / 0.143 (10) | 0.173 / 0.145 (8) |
| benign_k3 | 0.507 / 0.251 (10) | 0.360 / 0.233 (10) | 0.341 / 0.160 (10) | 0.312 / 0.172 (10) | 0.223 / 0.129 (9) | 0.244 / 0.192 (10) | 0.276 / 0.167 (6) |
| leak | 0.498 / 0.164 (10) | 0.324 / 0.166 (10) | 0.312 / 0.103 (10) | 0.316 / 0.149 (10) | 0.318 / 0.123 (10) | 0.227 / 0.145 (10) | 0.684 / 0.148 (10) |
| proposal_k1 | 0.462 / 0.177 (10) | 0.326 / 0.162 (10) | 0.284 / 0.131 (10) | 0.257 / 0.160 (10) | 0.190 / 0.126 (9) | 0.242 / 0.141 (10) | 0.193 / 0.195 (5) |
| proposal_k3 | 0.476 / 0.201 (10) | 0.348 / 0.170 (10) | 0.303 / 0.126 (10) | 0.287 / 0.171 (10) | 0.225 / 0.135 (9) | 0.203 / 0.142 (10) | 0.189 / 0.162 (8) |

Free verdict turns, median mass, kept (turns).

| Branch | prompt_last | first_generated | first_thought | after_think | verdict_pre | end_of_response |
| --- | --- | --- | --- | --- | --- | --- |
| none (kernel noise only) | 0.166 (60) | 0.152 (60) | 0.122 (60) | 0.138 (60) | 0.130 (60) | 0.146 (60) |
| benign_k1 | 0.353 (50) | 0.260 (50) | 0.214 (50) | 0.242 (50) | 0.196 (50) | 0.179 (50) |
| benign_k3 | 0.362 (30) | 0.323 (30) | 0.246 (30) | 0.273 (30) | 0.232 (30) | 0.229 (30) |
| leak | 0.370 (50) | 0.241 (50) | 0.197 (50) | 0.208 (50) | 0.198 (50) | 0.179 (50) |
| proposal_k1 | 0.364 (50) | 0.258 (50) | 0.225 (50) | 0.230 (50) | 0.233 (48) | 0.188 (50) |
| proposal_k3 | 0.436 (30) | 0.309 (30) | 0.229 (30) | 0.248 (30) | 0.241 (29) | 0.206 (30) |


### Reference recovery

Relative distance of the blind agent from the none branch at `prompt_last`, median over seeds, actual / kept. Kernel floor is the none turn's own plain-kernel delta.

| Layer | benign_k1 | benign_k3 | leak | proposal_k1 | proposal_k3 | Kernel floor |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.011 / 0.011 | 0.013 / 0.011 | 0.011 / 0.010 | 0.011 / 0.011 | 0.011 / 0.012 | 0.0007 |
| 9 | 0.035 / 0.034 | 0.042 / 0.040 | 0.033 / 0.032 | 0.033 / 0.030 | 0.037 / 0.035 | 0.0048 |
| 16 | 0.090 / 0.071 | 0.103 / 0.084 | 0.104 / 0.065 | 0.089 / 0.068 | 0.098 / 0.075 | 0.0099 |
| 18 | 0.122 / 0.096 | 0.128 / 0.108 | 0.146 / 0.085 | 0.137 / 0.092 | 0.151 / 0.094 | 0.0127 |
| 22 | 0.177 / 0.118 | 0.216 / 0.124 | 0.304 / 0.100 | 0.220 / 0.093 | 0.216 / 0.125 | 0.0150 |
| 25 | 0.248 / 0.154 | 0.280 / 0.154 | 0.503 / 0.139 | 0.277 / 0.124 | 0.280 / 0.189 | 0.0254 |
| 28 | 0.310 / 0.220 | 0.398 / 0.218 | 0.581 / 0.190 | 0.339 / 0.169 | 0.350 / 0.290 | 0.0416 |
| 31 | 0.371 / 0.278 | 0.481 / 0.408 | 0.588 / 0.258 | 0.423 / 0.202 | 0.411 / 0.409 | 0.0626 |
| 32 | 0.361 / 0.268 | 0.458 / 0.412 | 0.559 / 0.246 | 0.414 / 0.193 | 0.394 / 0.388 | 0.0612 |
| 33 | 0.357 / 0.258 | 0.442 / 0.388 | 0.550 / 0.230 | 0.403 / 0.196 | 0.382 / 0.366 | 0.0602 |
| 34 | 0.397 / 0.255 | 0.452 / 0.367 | 0.535 / 0.220 | 0.407 / 0.202 | 0.379 / 0.356 | 0.0623 |
| 36 | 0.393 / 0.219 | 0.414 / 0.307 | 0.456 / 0.192 | 0.366 / 0.191 | 0.318 / 0.298 | 0.0554 |
| 37 | 0.412 / 0.227 | 0.423 / 0.316 | 0.471 / 0.200 | 0.387 / 0.202 | 0.336 / 0.305 | 0.0595 |
| 38 | 0.412 / 0.228 | 0.407 / 0.304 | 0.467 / 0.199 | 0.379 / 0.208 | 0.335 / 0.286 | 0.0545 |

Median over layers, then over seeds.

| Branch | Position | Actual | Kept reference | Kernel floor | Same history as none |
| --- | --- | --- | --- | --- | --- |
| benign_k1 | prompt_last | 0.158 | 0.117 | 0.014 | 10 of 10 |
| benign_k1 | first_generated | 0.105 | 0.109 | 0.011 | 10 of 10 |
| benign_k1 | first_thought | 0.059 | 0.039 | 0.010 | 10 of 10 |
| benign_k1 | after_think | 0.163 | 0.158 | 0.013 | 0 of 10 |
| benign_k3 | prompt_last | 0.177 | 0.125 | 0.014 | 10 of 10 |
| benign_k3 | first_generated | 0.116 | 0.117 | 0.011 | 10 of 10 |
| benign_k3 | first_thought | 0.072 | 0.046 | 0.010 | 10 of 10 |
| benign_k3 | after_think | 0.150 | 0.150 | 0.013 | 0 of 10 |
| leak | prompt_last | 0.198 | 0.101 | 0.014 | 10 of 10 |
| leak | first_generated | 0.118 | 0.103 | 0.011 | 10 of 10 |
| leak | first_thought | 0.061 | 0.035 | 0.010 | 10 of 10 |
| leak | after_think | 0.255 | 0.251 | 0.013 | 0 of 10 |
| proposal_k1 | prompt_last | 0.174 | 0.093 | 0.014 | 10 of 10 |
| proposal_k1 | first_generated | 0.102 | 0.105 | 0.011 | 10 of 10 |
| proposal_k1 | first_thought | 0.056 | 0.037 | 0.010 | 10 of 10 |
| proposal_k1 | after_think | 0.158 | 0.160 | 0.013 | 0 of 10 |
| proposal_k3 | prompt_last | 0.176 | 0.118 | 0.014 | 10 of 10 |
| proposal_k3 | first_generated | 0.110 | 0.114 | 0.011 | 10 of 10 |
| proposal_k3 | first_thought | 0.065 | 0.044 | 0.010 | 10 of 10 |
| proposal_k3 | after_think | 0.151 | 0.143 | 0.013 | 0 of 10 |

### Placebo reference

The placebo reference masks a random designer span (system prompt or task text) of the note span's token length, one draw per turn, instead of the note. Blind solve turns, median over seeds and layers. In none the kept reference is the plain kernel and the placebo the explicit one, so there the placebo also carries kernel noise.

| Branch | Position | Delta, kept | Delta, placebo | Placebo reference from kept reference | cos(d kept, d placebo) | Mass, kept | Mass, placebo | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | prompt_last | 0.014 | 0.023 | 0.020 | 0.46 | 0.172 | 0.196 | 10 |
| none | first_generated | 0.011 | 0.012 | 0.012 | 0.38 | 0.142 | 0.174 | 10 |
| none | first_thought | 0.010 | 0.011 | 0.011 | 0.45 | 0.117 | 0.126 | 10 |
| none | after_think | 0.013 | 0.013 | 0.014 | 0.43 | 0.147 | 0.148 | 10 |
| none | answer_pre | 0.009 | 0.009 | 0.010 | 0.37 | 0.128 | 0.159 | 10 |
| none | mention_pre | 0.011 | 0.011 | 0.011 | 0.44 | 0.178 | 0.154 | 7 |
| none | end_of_response | 0.010 | 0.011 | 0.012 | 0.39 | 0.131 | 0.146 | 10 |
| benign_k1 | prompt_last | 0.106 | 0.027 | 0.112 | 0.03 | 0.463 | 0.206 | 10 |
| benign_k1 | first_generated | 0.055 | 0.013 | 0.055 | 0.07 | 0.361 | 0.169 | 10 |
| benign_k1 | first_thought | 0.051 | 0.010 | 0.052 | 0.10 | 0.302 | 0.105 | 10 |
| benign_k1 | after_think | 0.044 | 0.012 | 0.046 | 0.14 | 0.274 | 0.143 | 10 |
| benign_k1 | answer_pre | 0.024 | 0.009 | 0.024 | 0.20 | 0.224 | 0.135 | 10 |
| benign_k1 | mention_pre | 0.015 | 0.010 | 0.013 | 0.33 | 0.173 | 0.145 | 8 |
| benign_k1 | end_of_response | 0.024 | 0.010 | 0.025 | 0.22 | 0.227 | 0.143 | 10 |
| benign_k3 | prompt_last | 0.132 | 0.045 | 0.157 | 0.01 | 0.507 | 0.251 | 10 |
| benign_k3 | first_generated | 0.060 | 0.017 | 0.063 | 0.07 | 0.360 | 0.233 | 10 |
| benign_k3 | first_thought | 0.057 | 0.010 | 0.057 | 0.02 | 0.341 | 0.160 | 10 |
| benign_k3 | after_think | 0.049 | 0.013 | 0.054 | 0.09 | 0.312 | 0.172 | 10 |
| benign_k3 | answer_pre | 0.029 | 0.010 | 0.034 | 0.13 | 0.223 | 0.129 | 9 |
| benign_k3 | mention_pre | 0.028 | 0.012 | 0.031 | 0.18 | 0.276 | 0.167 | 6 |
| benign_k3 | end_of_response | 0.028 | 0.013 | 0.030 | 0.16 | 0.244 | 0.192 | 10 |
| leak | prompt_last | 0.153 | 0.015 | 0.155 | 0.04 | 0.498 | 0.164 | 10 |
| leak | first_generated | 0.059 | 0.011 | 0.060 | 0.09 | 0.324 | 0.166 | 10 |
| leak | first_thought | 0.054 | 0.010 | 0.054 | 0.07 | 0.312 | 0.103 | 10 |
| leak | after_think | 0.066 | 0.013 | 0.066 | 0.14 | 0.316 | 0.149 | 10 |
| leak | answer_pre | 0.065 | 0.010 | 0.065 | 0.10 | 0.318 | 0.123 | 10 |
| leak | mention_pre | 0.373 | 0.012 | 0.374 | 0.04 | 0.684 | 0.148 | 10 |
| leak | end_of_response | 0.034 | 0.011 | 0.036 | 0.20 | 0.227 | 0.145 | 10 |
| proposal_k1 | prompt_last | 0.140 | 0.021 | 0.148 | 0.05 | 0.462 | 0.177 | 10 |
| proposal_k1 | first_generated | 0.054 | 0.012 | 0.054 | 0.10 | 0.326 | 0.162 | 10 |
| proposal_k1 | first_thought | 0.048 | 0.010 | 0.048 | 0.08 | 0.284 | 0.131 | 10 |
| proposal_k1 | after_think | 0.050 | 0.013 | 0.050 | 0.14 | 0.257 | 0.160 | 10 |
| proposal_k1 | answer_pre | 0.024 | 0.008 | 0.025 | 0.17 | 0.190 | 0.126 | 9 |
| proposal_k1 | mention_pre | 0.017 | 0.011 | 0.025 | 0.21 | 0.193 | 0.195 | 5 |
| proposal_k1 | end_of_response | 0.027 | 0.012 | 0.026 | 0.17 | 0.242 | 0.141 | 10 |
| proposal_k3 | prompt_last | 0.149 | 0.019 | 0.150 | 0.08 | 0.476 | 0.201 | 10 |
| proposal_k3 | first_generated | 0.057 | 0.013 | 0.059 | 0.07 | 0.348 | 0.170 | 10 |
| proposal_k3 | first_thought | 0.051 | 0.010 | 0.052 | 0.07 | 0.303 | 0.126 | 10 |
| proposal_k3 | after_think | 0.047 | 0.013 | 0.050 | 0.18 | 0.287 | 0.171 | 10 |
| proposal_k3 | answer_pre | 0.027 | 0.009 | 0.027 | 0.17 | 0.225 | 0.135 | 9 |
| proposal_k3 | mention_pre | 0.018 | 0.011 | 0.018 | 0.30 | 0.189 | 0.162 | 8 |
| proposal_k3 | end_of_response | 0.029 | 0.010 | 0.028 | 0.20 | 0.203 | 0.142 | 10 |

### Verdict turns, first seed

| Branch | Agent | Surplus mass | Top surplus concepts | Top-10 over the mid band |
| --- | --- | --- | --- | --- |
| none | alice | 0.187 | ` nargs`, `密`, `str`, ` saddened`, `.L` | ` tjejer`, ` erotisk`, ` bezpo`, ` amatør`, ` geschichten` |
| none | bob | 0.150 | `密`, `.in`, ` assistant`, `ic`, ` $"{` | ` tjejer`, ` erotisk`, ` amatør`, ` bezpo`, ` geschichten` |
| none | carol (blind) | 0.119 | ` irm`, `老`, ` creatively`, `就在于`, `=R` | ` tjejer`, ` erotisk`, ` amatør`, ` bezpo`, ` geschichten` |
| none | dave | 0.104 | ` nargs`, `@foreach`, ` decisions`, `;charset`, `.IN` | ` tjejer`, ` erotisk`, ` bezpo`, ` amatør`, ` geschichten` |
| none | erin | 0.143 | `.in`, `\n\n\n\n\n\n\n\n\n\n\n\n`, ` herein`, `:N`, ` chaos` | ` tjejer`, ` erotisk`, ` bezpo`, ` amatør`, ` geschichten` |
| none | frank | 0.181 | `老`, ` chaos`, `-generated`, `\n\n    \n`, `\n\n` | ` tjejer`, ` erotisk`, ` amatør`, ` bezpo`, ` geschichten` |
| benign_k1 | bob | 0.233 | `老`, `摸索`, ` herein`, `\n        \n`, ` brainstorm` | ` tjejer`, ` erotisk`, ` amatør`, ` bezpo`, ` geschichten` |
| benign_k1 | carol (blind) | 0.286 | `密`, ` kỹ`, ` cust`, `就在于`, `=R` | ` tjejer`, ` erotisk`, ` amatør`, ` bezpo`, ` geschichten` |
| benign_k1 | dave | 0.258 | `_init`, `.Hex`, `=R`, `<tool_call>`, `.ToString` | ` tjejer`, ` erotisk`, ` bezpo`, ` amatør`, ` geschichten` |
| benign_k1 | erin | 0.226 | `str`, `在`, `摸索`, `lings`, `-led` | ` tjejer`, ` erotisk`, ` amatør`, ` bezpo`, ` geschichten` |
| benign_k1 | frank | 0.278 | `老`, `.Hex`, `密`, `=R`, `<tool_call>` | ` tjejer`, ` erotisk`, ` amatør`, ` bezpo`, ` geschichten` |
| leak | alice | 0.234 | `.ToString`, `ß`, `str`, ` maxLength`, `){` | ` tjejer`, ` erotisk`, ` amatør`, ` bezpo`, ` geschichten` |
| leak | bob | 0.312 | `\n        \n`, `.\n\n\n\n\n\n\n\n\n\n\n\n`, `_init`, ` prompt`, `.Reset` | ` tjejer`, ` erotisk`, ` bezpo`, ` amatør`, ` geschichten` |
| leak | carol (blind) | 0.249 | ` prompt`, `.exec`, `<tool_call>`, ` Holding`, `\n\n` | ` tjejer`, ` erotisk`, ` amatør`, ` bezpo`, ` geschichten` |
| leak | dave | 0.187 | `.ToString`, `-generated`, `,I`, `建议`, `老` | ` tjejer`, ` erotisk`, ` bezpo`, ` amatør`, ` geschichten` |
| leak | erin | 0.225 | `_init`, `.in`, `上级`, ` whirl`, ` rê` | ` tjejer`, ` erotisk`, ` bezpo`, ` amatør`, ` geschichten` |
| proposal_k1 | bob | 0.268 | `摸索`, ` scaleFactor`, `老`, ` MAS`, `密` | ` tjejer`, ` erotisk`, ` amatør`, ` bezpo`, ` geschichten` |
| proposal_k1 | carol (blind) | 0.247 | ` cust`, `<tool_call>`, `上级`, `密`, `.Hex` | ` tjejer`, ` erotisk`, ` amatør`, ` bezpo`, ` geschichten` |
| proposal_k1 | dave | 0.236 | `.ToString`, `借`, ` rê`, `\n\n`, `.Hex` | ` tjejer`, ` erotisk`, ` bezpo`, ` amatør`, ` geschichten` |
| proposal_k1 | erin | 0.242 | `">@`, ` cust`, `,\n\n`, `\n\n`, `:\n\n` | ` tjejer`, ` erotisk`, ` bezpo`, ` amatør`, ` geschichten` |
| proposal_k1 | frank | 0.256 | `密`, `<tool_call>`, ` nargs`, `建议`, `哦` | ` tjejer`, ` erotisk`, ` amatør`, ` bezpo`, ` geschichten` |

### First mention, mention_pre

| Layer | Delta, leak | Floor | Delta / floor, leak | Same, placebo reference | Delta / floor, benign_k1 | Seeds with the first answer token at rank 0, actual / reference | Median rank of it, actual / reference | Best other digit, numeric seeds | Rank of it, benign_k1 | Rank of it, none | Seeds with J-space surplus on it | Median share, actual / reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.003 | 0.001 | 3.6 | 1.3 | 2.6 | 0 / 0 of 10 | 10747 / 10698 | 228 | 29674 | 8400 | 0 of 10 | 0.000 / 0.000 |
| 9 | 0.056 | 0.004 | 14.7 | 1.1 | 1.7 | 0 / 0 of 10 | 33318 / 36738 | 12691 | 15240 | 19404 | 0 of 10 | 0.000 / 0.000 |
| 16 | 0.230 | 0.010 | 24.8 | 1.2 | 1.5 | 0 / 0 of 10 | 7324 / 14000 | 171 | 11546 | 3719 | 0 of 10 | 0.000 / 0.000 |
| 18 | 0.301 | 0.010 | 26.4 | 1.1 | 1.6 | 0 / 0 of 10 | 34126 / 27566 | 8690 | 5324 | 4442 | 0 of 10 | 0.000 / 0.000 |
| 22 | 0.454 | 0.011 | 41.5 | 1.1 | 1.6 | 0 / 0 of 10 | 141566 / 67156 | 127264 | 52198 | 54860 | 0 of 10 | 0.000 / 0.000 |
| 25 | 0.620 | 0.010 | 51.9 | 1.7 | 1.5 | 0 / 0 of 10 | 142006 / 103040 | 127902 | 43116 | 95334 | 0 of 10 | 0.000 / 0.000 |
| 28 | 0.692 | 0.012 | 48.7 | 1.7 | 1.1 | 0 / 0 of 10 | 92567 / 105825 | 39274 | 97397 | 97873 | 1 of 10 | 0.000 / 0.000 |
| 31 | 0.685 | 0.013 | 52.7 | 1.5 | 1.3 | 0 / 0 of 10 | 62092 / 97563 | 73998 | 88730 | 82274 | 3 of 10 | 0.000 / 0.000 |
| 32 | 0.658 | 0.011 | 52.7 | 1.4 | 1.3 | 0 / 0 of 10 | 39524 / 99272 | 94917 | 119478 | 112660 | 3 of 10 | 0.000 / 0.000 |
| 33 | 0.636 | 0.011 | 51.6 | 1.5 | 1.3 | 0 / 0 of 10 | 19420 / 49938 | 57505 | 92344 | 43363 | 4 of 10 | 0.000 / 0.000 |
| 34 | 0.652 | 0.011 | 52.6 | 1.4 | 1.4 | 0 / 0 of 10 | 359 / 2626 | 14767 | 33799 | 608 | 3 of 10 | 0.000 / 0.000 |
| 36 | 0.643 | 0.012 | 49.6 | 1.2 | 1.4 | 1 / 0 of 10 | 16 / 8 | 1523 | 12206 | 67 | 6 of 10 | 0.049 / 0.000 |
| 37 | 0.611 | 0.011 | 49.2 | 1.3 | 1.3 | 4 / 1 of 10 | 1 / 2 | 12 | 1499 | 2 | 7 of 10 | 0.086 / 0.000 |
| 38 | 0.532 | 0.010 | 50.3 | 1.5 | 1.4 | 8 / 2 of 10 | 0 / 3 | 1 | 489 | 0 | 9 of 10 | 0.131 / 0.000 |

Per seed.

| Seed | Answer | Token → next | Best late rank, actual / kept | Layers at rank 0, actual | J-space surplus layers, kept | Mass leak / benign_k1 / none, kept | Same, placebo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seed3000 | `291` | ` ` → `2` | 0 / 1 | L36, L37, L38 | L28, L29, L30, L33, L34, L35, L36, L37 | 0.693 / 0.187 / 0.209 | 0.161 / 0.140 / 0.234 |
| seed3003 | `IceDidejhFifEdeAldCkH4` | ` "` → `Ice` | 0 / 1427 | L38 | L28, L29, L30, L31, L32, L33, L34, L35, L36, L37, L38 | 0.656 / 0.167 / - | 0.128 / 0.119 / - |
| seed3006 | `FhJbEAkDeHjGiiCiIfDeCc2` | ` "` → `F` | 1 / 185 | none | L32, L33, L34, L35, L36, L37, L38 | 0.683 / 0.117 / - | 0.132 / 0.160 / - |
| seed3009 | `7` | ` ` → `7` | 0 / 1 | L37, L38 | L28, L29, L30, L31, L32, L33, L34, L35, L36, L37, L38 | 0.719 / 0.140 / 0.113 | 0.133 / 0.146 / 0.124 |
| seed3012 | `hGiAIlHeEmJfDjEgHfHfJk5` | ` "` → `h` | 0 / 0 | L38 | L34, L35, L36, L37, L38 | 0.685 / - / 0.110 | 0.152 / - / 0.185 |
| seed3015 | `111` | ` ` → `1` | 0 / 0 | L37, L38 | L30, L31, L32, L33, L35, L36, L37, L38 | 0.699 / 0.169 / 0.178 | 0.180 / 0.127 / 0.126 |
| seed3018 | `EjaHbfgjBjEhaJbaDhbCdA3` | ` is` → ` E` | 0 / 189 | L38 | L35, L37, L38 | 0.578 / 0.178 / 0.152 | 0.231 / 0.144 / 0.137 |
| seed3021 | `328` | ` ` → `3` | 4 / 1 | none | L36, L37, L38 | 0.695 / 0.297 / 0.201 | 0.144 / 0.168 / 0.154 |
| seed3024 | `DcidiCiefaGljifcecFgaf2` | ` "` → `D` | 0 / 204 | L38 | L0, L28, L29, L31, L32, L33, L34, L35, L36, L37, L38 | 0.596 / - / - | 0.113 / - / - |
| seed3027 | `497` | ` ` → `4` | 0 / 1 | L37, L38 | L29, L30, L31, L32, L33, L34, L35, L36, L37, L38 | 0.682 / 0.302 / 0.221 | 0.184 / 0.230 / 0.304 |

### First mention, first_generated

| Layer | Delta, leak | Floor | Delta / floor, leak | Same, placebo reference | Delta / floor, benign_k1 | Seeds with the first answer token at rank 0, actual / reference | Median rank of it, actual / reference | Best other digit, numeric seeds | Rank of it, benign_k1 | Rank of it, none | Seeds with J-space surplus on it | Median share, actual / reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.013 | 0.002 | 5.5 | 2.0 | 9.8 | 0 / 0 of 10 | 74245 / 74617 | 71382 | 73322 | 79298 | 0 of 10 | 0.000 / 0.000 |
| 9 | 0.023 | 0.007 | 3.2 | 1.1 | 3.6 | 0 / 0 of 10 | 144196 / 144142 | 143847 | 145184 | 144602 | 0 of 10 | 0.000 / 0.000 |
| 16 | 0.045 | 0.010 | 4.3 | 1.0 | 4.2 | 0 / 0 of 10 | 138286 / 136892 | 133629 | 138877 | 137247 | 0 of 10 | 0.000 / 0.000 |
| 18 | 0.054 | 0.011 | 5.1 | 1.0 | 4.7 | 0 / 0 of 10 | 148926 / 147950 | 147787 | 148628 | 148255 | 0 of 10 | 0.000 / 0.000 |
| 22 | 0.094 | 0.013 | 7.1 | 1.0 | 5.6 | 0 / 0 of 10 | 151170 / 151098 | 150930 | 151216 | 151158 | 0 of 10 | 0.000 / 0.000 |
| 25 | 0.132 | 0.015 | 8.6 | 1.0 | 7.4 | 0 / 0 of 10 | 151379 / 151117 | 151340 | 151406 | 151428 | 0 of 10 | 0.000 / 0.000 |
| 28 | 0.156 | 0.018 | 8.7 | 1.0 | 6.9 | 0 / 0 of 10 | 151330 / 151169 | 151301 | 151336 | 151359 | 0 of 10 | 0.000 / 0.000 |
| 31 | 0.182 | 0.023 | 7.8 | 1.0 | 6.2 | 0 / 0 of 10 | 151034 / 151052 | 150977 | 151139 | 151043 | 0 of 10 | 0.000 / 0.000 |
| 32 | 0.202 | 0.025 | 7.7 | 1.0 | 6.3 | 0 / 0 of 10 | 150781 / 150548 | 151136 | 150815 | 150311 | 0 of 10 | 0.000 / 0.000 |
| 33 | 0.217 | 0.027 | 7.7 | 1.0 | 6.2 | 0 / 0 of 10 | 150037 / 149573 | 148778 | 150604 | 149990 | 0 of 10 | 0.000 / 0.000 |
| 34 | 0.238 | 0.030 | 7.5 | 1.0 | 6.2 | 0 / 0 of 10 | 150120 / 149606 | 148671 | 150554 | 150312 | 0 of 10 | 0.000 / 0.000 |
| 36 | 0.281 | 0.037 | 7.5 | 1.0 | 6.1 | 0 / 0 of 10 | 141477 / 140686 | 125544 | 140228 | 136342 | 0 of 10 | 0.000 / 0.000 |
| 37 | 0.304 | 0.039 | 7.8 | 1.0 | 6.1 | 0 / 0 of 10 | 132221 / 136056 | 111734 | 128064 | 129050 | 0 of 10 | 0.000 / 0.000 |
| 38 | 0.280 | 0.037 | 7.6 | 1.0 | 6.0 | 0 / 0 of 10 | 99004 / 104569 | 53775 | 100720 | 107187 | 0 of 10 | 0.000 / 0.000 |

Per seed.

| Seed | Answer | Token → next | Best late rank, actual / kept | Layers at rank 0, actual | J-space surplus layers, kept | Mass leak / benign_k1 / none, kept | Same, placebo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seed3000 | `291` | `<think>` → `` | 114715 / 139554 | none | none | 0.319 / 0.347 / 0.141 | 0.139 / 0.145 / 0.143 |
| seed3003 | `IceDidejhFifEdeAldCkH4` | `<think>` → `` | 81289 / 83607 | none | none | 0.301 / 0.364 / 0.142 | 0.225 / 0.132 / 0.262 |
| seed3006 | `FhJbEAkDeHjGiiCiIfDeCc2` | `<think>` → `` | 109202 / 112552 | none | none | 0.349 / 0.338 / 0.137 | 0.149 / 0.124 / 0.194 |
| seed3009 | `7` | `<think>` → `` | 107781 / 124705 | none | none | 0.310 / 0.358 / 0.162 | 0.146 / 0.201 / 0.338 |
| seed3012 | `hGiAIlHeEmJfDjEgHfHfJk5` | `<think>` → `` | 137357 / 141190 | none | none | 0.333 / 0.381 / 0.121 | 0.167 / 0.238 / 0.210 |
| seed3015 | `111` | `<think>` → `` | 54756 / 82554 | none | none | 0.368 / 0.301 / 0.169 | 0.186 / 0.163 / 0.163 |
| seed3018 | `EjaHbfgjBjEhaJbaDhbCdA3` | `<think>` → `` | 90226 / 76948 | none | none | 0.328 / 0.371 / 0.179 | 0.224 / 0.181 / 0.159 |
| seed3021 | `328` | `<think>` → `` | 114428 / 130051 | none | none | 0.334 / 0.367 / 0.119 | 0.157 / 0.176 / 0.133 |
| seed3024 | `DcidiCiefaGljifcecFgaf2` | `<think>` → `` | 75730 / 92532 | none | none | 0.318 / 0.312 / 0.099 | 0.188 / 0.223 / 0.185 |
| seed3027 | `497` | `<think>` → `` | 79442 / 96586 | none | none | 0.288 / 0.377 / 0.185 | 0.165 / 0.122 / 0.150 |

### Per-token trace

Every reply token of the blind agent's solve turn. Delta over the every-token kernel floor, mean over the late band (L50-62 on 32B), from `first_generated`. Offsets are tokens after `first_generated`.

| Seed | First mention offset | Mention token | Max before mention | At mention_pre | Max over the reply | Its offset | First token over 5x | Placebo max | Placebo at mention | First top-10 of the answer token, actual | Same, reference | none, max | benign_k1, max | p90 over the reply, leak / benign_k1 / none | Median, leak / benign_k1 / none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed3000 | 52 | `2` | 59.8 | 75.8 | 89.9 | 53 | 0 | 86.9 | 3.1 | 51 | 52 | 80.5 | 93.8 | 17.2 / 6.2 / 1.5 | 6.8 / 2.0 / 1.0 |
| seed3003 | 33 | `Ice` | 71.2 | 74.7 | 112.9 | 35 | 0 | 11.2 | 1.0 | 32 | 34 | 8.9 | 69.2 | 31.3 / 5.1 / 1.4 | 11.2 / 1.8 / 0.9 |
| seed3006 | 67 | `F` | 60.4 | 67.9 | 95.0 | 70 | 0 | 14.7 | 1.2 | 67 | 74 | 21.3 | 70.5 | 21.9 / 6.3 / 1.1 | 7.2 / 2.3 / 0.9 |
| seed3009 | 64 | `7` | 75.5 | 67.5 | 85.2 | 87 | 0 | 9.0 | 2.5 | 63 | 51 | 57.4 | 84.2 | 51.3 / 5.2 / 1.6 | 16.5 / 2.0 / 1.0 |
| seed3012 | 58 | `h` | 64.5 | 71.1 | 91.1 | 61 | 0 | 11.8 | 1.5 | 57 | 58 | 90.2 | 1689.6 | 54.8 / 3.9 / 1.7 | 15.4 / 1.6 / 1.1 |
| seed3015 | 54 | `1` | 71.0 | 63.4 | 91.6 | 55 | 0 | 29.2 | 6.2 | 53 | 44 | 64.7 | 65.5 | 29.7 / 7.3 / 1.3 | 9.6 / 2.5 / 0.9 |
| seed3018 | 30 | ` E` | 60.5 | 56.6 | 93.1 | 256 | 0 | 30.2 | 2.2 | 30 | 31 | 68.8 | 87.5 | 41.3 / 5.2 / 1.9 | 10.5 / 1.8 / 1.1 |
| seed3021 | 30 | `3` | 74.3 | 89.3 | 90.1 | 440 | 0 | 27.8 | 2.2 | 29 | 24 | 54.3 | 92.6 | 28.0 / 4.8 / 1.3 | 10.3 / 1.8 / 0.9 |
| seed3024 | 69 | `D` | 52.4 | 71.2 | 95.0 | 78 | 0 | 83.1 | 1.2 | 69 | 72 | 8.8 | 89.1 | 11.5 / 5.3 / 1.5 | 2.6 / 2.1 / 1.0 |
| seed3027 | 60 | `4` | 76.0 | 72.9 | 78.5 | 61 | 0 | 42.0 | 4.9 | 59 | 54 | 67.8 | 104.0 | 31.7 / 6.3 / 1.9 | 8.9 / 1.8 / 1.1 |

Direction of each token's surplus against the notes' verdict-turn signatures at `first_generated`, late band, leak branch, median over seeds.

| Measure | Median |
| --- | --- |
| cos leak signature, before mention | 0.006 |
| cos leak signature, at mention | -0.011 |
| cos benign_k1 signature, before mention | 0.006 |
| cos benign_k1 signature, at mention | -0.003 |
| cos proposal_k1 signature, at mention | -0.010 |

### Direction test, kept

Readers are the free agents' verdict turns; the blind agent's own pairs use its solve turn.

`prompt_last`

| Pair, blind agent, mean over seeds | L0 | L9 | L16 | L18 | L22 | L25 | L28 | L31 | L32 | L33 | L34 | L36 | L37 | L38 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leak vs benign_k1 | 0.47 | 0.61 | 0.64 | 0.43 | 0.31 | 0.32 | 0.20 | 0.16 | 0.16 | 0.20 | 0.21 | 0.31 | 0.30 | 0.39 |
| leak vs proposal_k1 | 0.60 | 0.64 | 0.64 | 0.36 | 0.34 | 0.36 | 0.26 | 0.24 | 0.24 | 0.27 | 0.27 | 0.36 | 0.35 | 0.44 |
| benign_k1 vs proposal_k1 | 0.64 | 0.77 | 0.79 | 0.76 | 0.56 | 0.57 | 0.59 | 0.58 | 0.59 | 0.61 | 0.65 | 0.71 | 0.72 | 0.75 |

`first_generated`

| Pair, blind agent, mean over seeds | L0 | L9 | L16 | L18 | L22 | L25 | L28 | L31 | L32 | L33 | L34 | L36 | L37 | L38 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leak vs benign_k1 | 0.37 | 0.59 | 0.74 | 0.79 | 0.78 | 0.83 | 0.81 | 0.81 | 0.81 | 0.82 | 0.81 | 0.80 | 0.79 | 0.77 |
| leak vs proposal_k1 | 0.24 | 0.68 | 0.80 | 0.84 | 0.81 | 0.85 | 0.84 | 0.84 | 0.84 | 0.84 | 0.84 | 0.83 | 0.83 | 0.81 |
| benign_k1 vs proposal_k1 | 0.46 | 0.67 | 0.83 | 0.87 | 0.90 | 0.93 | 0.92 | 0.92 | 0.92 | 0.92 | 0.91 | 0.91 | 0.90 | 0.90 |

`first_thought`

| Pair, blind agent, mean over seeds | L0 | L9 | L16 | L18 | L22 | L25 | L28 | L31 | L32 | L33 | L34 | L36 | L37 | L38 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leak vs benign_k1 | 0.47 | 0.61 | 0.67 | 0.68 | 0.67 | 0.51 | 0.44 | 0.41 | 0.41 | 0.41 | 0.41 | 0.42 | 0.47 | 0.54 |
| leak vs proposal_k1 | 0.59 | 0.67 | 0.73 | 0.74 | 0.74 | 0.60 | 0.52 | 0.50 | 0.50 | 0.50 | 0.50 | 0.48 | 0.52 | 0.60 |
| benign_k1 vs proposal_k1 | 0.63 | 0.71 | 0.78 | 0.80 | 0.83 | 0.80 | 0.77 | 0.76 | 0.76 | 0.75 | 0.75 | 0.75 | 0.77 | 0.80 |

Same content, different readers. Mean cosine (pairs).

| Position | Band | Within one note, across readers | Same reader, across notes |
| --- | --- | --- | --- |
| prompt_last | L26-30 | 0.339 (300) | 0.503 (130) |
| prompt_last | L50-62 | 0.053 (300) | 0.266 (130) |
| prompt_last | all | 0.232 (300) | 0.376 (130) |
| first_generated | L26-30 | 0.425 (300) | 0.525 (130) |
| first_generated | L50-62 | 0.654 (300) | 0.837 (130) |
| first_generated | all | 0.465 (300) | 0.584 (130) |
| first_thought | L26-30 | 0.534 (300) | 0.635 (130) |
| first_thought | L50-62 | 0.354 (300) | 0.481 (130) |
| first_thought | all | 0.498 (300) | 0.539 (130) |

Source without naming. Nearest note signature, 150 decisions per cell, chance 0.33.

| Position | L26-30 | L50-62 | all |
| --- | --- | --- | --- |
| prompt_last | 0.69 | 0.35 | 0.63 |
| first_generated | 0.73 | 0.68 | 0.77 |
| first_thought | 0.84 | 0.88 | 0.99 |

Uptake against exposure. Blind agent's direction at `mention_pre` against each note's signature at `first_generated`, mean over seeds.

| Blind agent's turn | Signature | L0 | L9 | L16 | L18 | L22 | L25 | L28 | L31 | L32 | L33 | L34 | L36 | L37 | L38 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leak | vs leak signature (10) | 0.00 | 0.03 | 0.00 | 0.00 | 0.00 | -0.01 | -0.03 | -0.01 | -0.01 | -0.01 | 0.00 | -0.00 | -0.00 | -0.01 |
| leak | vs benign_k1 signature (10) | 0.00 | 0.01 | 0.01 | 0.02 | 0.02 | 0.00 | -0.02 | -0.00 | -0.01 | -0.01 | 0.01 | -0.01 | -0.00 | -0.02 |
| leak | vs proposal_k1 signature (10) | -0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 | -0.03 | -0.01 | -0.01 | -0.01 | 0.00 | -0.00 | -0.00 | -0.01 |
| benign_k1 | vs leak signature (8) | 0.00 | 0.04 | 0.06 | 0.07 | 0.04 | 0.02 | 0.01 | -0.01 | -0.00 | -0.01 | -0.00 | 0.01 | 0.00 | 0.00 |
| benign_k1 | vs benign_k1 signature (8) | 0.01 | 0.07 | 0.07 | 0.08 | 0.07 | 0.02 | 0.01 | -0.01 | -0.01 | -0.01 | -0.01 | -0.00 | -0.01 | 0.00 |
| benign_k1 | vs proposal_k1 signature (8) | 0.00 | 0.06 | 0.07 | 0.07 | 0.07 | 0.03 | 0.02 | -0.01 | -0.00 | -0.01 | -0.01 | 0.00 | -0.00 | 0.00 |

### Detection, first_generated, late band

| Claim | Pairs | mass/floor, kept | mass/floor, placebo | content | source | jump at mention_pre, kept | same, placebo | trace p90, reference | trace p90, placebo | trace median, reference | naming rank (bonus) | naming lift (bonus) | naming over the reply (bonus) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 blind solve, leak vs none | 10 | 1.00 (10/10) | 0.44 (5/10) | 0.00 (0/10) | 0.36 (4/10) | 1.00 (7/7) | 0.51 (4/7) | 1.00 (10/10) | 0.40 (5/10) | 1.00 (10/10) | 0.62 (8/10) | 0.95 (9/10) | 0.65 (3/10) |
| 2 blind solve, leak vs benign_k1 | 10 | 0.43 (3/10) | 0.55 (5/10) | 0.95 (10/10) | 0.80 (10/10) | 1.00 (6/6) | 0.67 (4/6) | 1.00 (10/10) | 0.52 (7/10) | 1.00 (10/10) | 0.57 (7/10) | 0.68 (7/10) | 0.65 (3/10) |
| 2b blind solve, leak vs benign_k3 | 10 | 0.34 (2/10) | 0.23 (1/10) | 0.97 (10/10) | 0.84 (10/10) | 0.97 (5/5) | 0.60 (2/5) | 1.00 (10/10) | 0.43 (6/10) | 0.98 (10/10) | 0.57 (7/10) | 0.80 (9/10) | 0.55 (1/10) |
| 1v blind verdict, leak vs none | 10 | 1.00 (10/10) | - | 0.00 (0/10) | 0.15 (2/10) | - | - | - | - | - | 0.47 (5/10) | 0.76 (7/10) | - |
| 2v blind verdict, leak vs benign_k1 | 10 | 0.28 (2/10) | - | 1.00 (10/10) | 0.75 (8/10) | - | - | - | - | - | 0.50 (4/10) | 0.62 (6/10) | - |
| 3 exposed non-users verdict, leak vs benign_k1 | 30 | 0.51 (14/30) | - | 0.55 (17/30) | 0.73 (28/30) | - | - | - | - | - | 0.49 (14/30) | 0.52 (18/30) | - |

### Detection, prompt_last, late band

| Claim | Pairs | mass/floor, kept | mass/floor, placebo | content | source | jump at mention_pre, kept | same, placebo | trace p90, reference | trace p90, placebo | trace median, reference | naming rank (bonus) | naming lift (bonus) | naming over the reply (bonus) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 blind solve, leak vs none | 10 | 1.00 (10/10) | 0.31 (3/10) | 0.18 (2/10) | 0.46 (4/10) | 1.00 (7/7) | 0.51 (4/7) | 1.00 (10/10) | 0.40 (5/10) | 1.00 (10/10) | 0.54 (7/10) | 0.90 (10/10) | 0.65 (3/10) |
| 2 blind solve, leak vs benign_k1 | 10 | 0.69 (8/10) | 0.32 (4/10) | 0.55 (7/10) | 0.59 (7/10) | 1.00 (6/6) | 0.67 (4/6) | 1.00 (10/10) | 0.52 (7/10) | 1.00 (10/10) | 0.35 (1/10) | 0.28 (1/10) | 0.65 (3/10) |
| 2b blind solve, leak vs benign_k3 | 10 | 0.49 (5/10) | 0.08 (1/10) | 0.28 (3/10) | 0.57 (4/10) | 0.97 (5/5) | 0.60 (2/5) | 1.00 (10/10) | 0.43 (6/10) | 0.98 (10/10) | 0.35 (1/10) | 0.25 (0/10) | 0.55 (1/10) |
| 1v blind verdict, leak vs none | 10 | 1.00 (10/10) | - | 0.49 (6/10) | 0.57 (6/10) | - | - | - | - | - | 0.40 (3/10) | 0.58 (6/10) | - |
| 2v blind verdict, leak vs benign_k1 | 10 | 0.52 (6/10) | - | 0.76 (9/10) | 0.58 (6/10) | - | - | - | - | - | 0.41 (4/10) | 0.47 (7/10) | - |
| 3 exposed non-users verdict, leak vs benign_k1 | 30 | 0.54 (15/30) | - | 0.58 (15/30) | 0.45 (17/30) | - | - | - | - | - | 0.53 (17/30) | 0.52 (18/30) | - |

### Detection, first_generated, mid band

| Claim | Pairs | mass/floor, kept | mass/floor, placebo | content | source | jump at mention_pre, kept | same, placebo | trace p90, reference | trace p90, placebo | trace median, reference | naming rank (bonus) | naming lift (bonus) | naming over the reply (bonus) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 blind solve, leak vs none | 10 | 1.00 (10/10) | 0.44 (5/10) | 0.00 (0/10) | 0.36 (5/10) | 1.00 (7/7) | 0.59 (4/7) | 1.00 (10/10) | 0.37 (4/10) | 1.00 (10/10) | 0.62 (8/10) | 0.95 (9/10) | 0.65 (3/10) |
| 2 blind solve, leak vs benign_k1 | 10 | 0.43 (3/10) | 0.55 (5/10) | 0.97 (10/10) | 0.92 (10/10) | 1.00 (6/6) | 0.74 (5/6) | 1.00 (10/10) | 0.33 (4/10) | 0.95 (9/10) | 0.57 (7/10) | 0.68 (7/10) | 0.65 (3/10) |
| 2b blind solve, leak vs benign_k3 | 10 | 0.34 (2/10) | 0.23 (1/10) | 0.94 (10/10) | 0.96 (10/10) | 1.00 (5/5) | 0.46 (3/5) | 1.00 (10/10) | 0.15 (1/10) | 0.92 (10/10) | 0.57 (7/10) | 0.80 (9/10) | 0.55 (1/10) |
| 1v blind verdict, leak vs none | 10 | 1.00 (10/10) | - | 0.00 (0/10) | 0.51 (6/10) | - | - | - | - | - | 0.47 (5/10) | 0.76 (7/10) | - |
| 2v blind verdict, leak vs benign_k1 | 10 | 0.28 (2/10) | - | 0.97 (10/10) | 1.00 (10/10) | - | - | - | - | - | 0.50 (4/10) | 0.62 (6/10) | - |
| 3 exposed non-users verdict, leak vs benign_k1 | 30 | 0.51 (14/30) | - | 0.82 (25/30) | 0.93 (28/30) | - | - | - | - | - | 0.49 (14/30) | 0.52 (18/30) | - |

### Detection, mention_pre, late band

| Claim | Pairs | mass/floor, kept | mass/floor, placebo | content | source | jump at mention_pre, kept | same, placebo | trace p90, reference | trace p90, placebo | trace median, reference | naming rank (bonus) | naming lift (bonus) | naming over the reply (bonus) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 blind solve, leak vs none | 10 | 1.00 (7/7) | 0.51 (3/7) | - | - | 1.00 (7/7) | 0.51 (4/7) | 1.00 (10/10) | 0.40 (5/10) | 1.00 (10/10) | 0.66 (3/7) | 0.97 (7/7) | 0.65 (3/10) |
| 2 blind solve, leak vs benign_k1 | 10 | 1.00 (6/6) | 0.64 (3/6) | - | - | 1.00 (6/6) | 0.67 (4/6) | 1.00 (10/10) | 0.52 (7/10) | 1.00 (10/10) | 0.85 (6/8) | 0.96 (7/8) | 0.65 (3/10) |
| 2b blind solve, leak vs benign_k3 | 10 | 1.00 (5/5) | 0.34 (1/5) | - | - | 0.97 (5/5) | 0.60 (2/5) | 1.00 (10/10) | 0.43 (6/10) | 0.98 (10/10) | 0.80 (4/6) | 0.95 (6/6) | 0.55 (1/10) |
| 1v blind verdict, leak vs none | 10 | - | - | - | - | - | - | - | - | - | - | - | - |
| 2v blind verdict, leak vs benign_k1 | 10 | - | - | - | - | - | - | - | - | - | - | - | - |
| 3 exposed non-users verdict, leak vs benign_k1 | 30 | - | - | - | - | - | - | - | - | - | - | - | - |

