# Lane B. Phase 1b (Transfer) and 1a (Stego) of the surplus detector, Qwen3-32B

Branch `claude/lane-b-phase1-16e264`. Code `scripts/lane_b.py` (replay, J-space, scoring), `scripts/lane_b_tables.py` (the tables below), pod scripts `scripts/pod/run_lane_b.sh` and `scripts/pod/run_lane_b_stego_score.sh`. Numbers in `results/lane_b/noise.json`, `transfer.json`, `stego.json`. Arrays on the pod under `/workspace/swarm/data/replay_lane_b` (see the data note at the end).

Status at 16:17 Los Angeles: Transfer done and written up below. Stego replay running.

<!-- HEADLINE -->

## What ran and what did not

| Item | State | Note |
| --- | --- | --- |
| Pod, model, lens | done | Own pod, model loaded at 15:09 |
| Noise floor on this pod | done | 4 Transfer and 4 Stego honest control turns |
| Phase 1b Transfer, real reference | done | 288 rows, both positions |
| Phase 1b Transfer, placebo and swapped | done | W4's `padded` group not run in the main pass |
| Phase 1a Stego | running | |

<!-- RAN -->

## Pod, hours and cost

<!-- POD -->

## Noise floor on this pod

Honest control turns, each run three times: plain kernel, plain kernel again, explicit 4D mask kernel. Transfer turns are four control agents (first one `cicd__cicd_01__control`, Celine Dubois, 2740 tokens). Stego turns are an honest player at hand 4 of four control runs (first one `bj__shoe00__control`, Yusuf Mironova, 1150 tokens; one of the four speaks before the sender in that hand, so it has no `message_end`). Mass is J-space surplus mass over all 63 layers, 0 to 1. Delta is `||a - b|| / ||b||`, mean over layers.

| Tier | Comparison | Position | Turns | Bit-identical | Mass, median | Mass, max | Relative residual delta, median | First turn, mass / delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stego | plain vs explicit mask | message_end | 3 | 0 | 0.158 | 0.169 | 0.0116 | 0.169 / 0.0124 |
| stego | plain vs explicit mask | prompt_last | 4 | 0 | 0.147 | 0.218 | 0.0117 | 0.138 / 0.0116 |
| stego | same kernel twice | message_end | 3 | 3 | 0.000 | 0.000 | 0.0000 | 0.000 / 0.0000 |
| stego | same kernel twice | prompt_last | 4 | 4 | 0.000 | 0.000 | 0.0000 | 0.000 / 0.0000 |
| transfer | plain vs explicit mask | end_of_response | 4 | 0 | 0.151 | 0.170 | 0.0113 | 0.170 / 0.0127 |
| transfer | plain vs explicit mask | first_response | 4 | 0 | 0.109 | 0.115 | 0.0091 | 0.115 / 0.0090 |
| transfer | same kernel twice | end_of_response | 4 | 4 | 0.000 | 0.000 | 0.0000 | 0.000 / 0.0000 |
| transfer | same kernel twice | first_response | 4 | 4 | 0.000 | 0.000 | 0.0000 | 0.000 / 0.0000 |

| Plain vs explicit, median by layer | L0 | L15 | L26 | L30 | L36 | L45 | L50 | L56 | L62 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stego | explicit | message_end | 0.0010 | 0.0067 | 0.0099 | 0.0114 | 0.0120 | 0.0137 | 0.0185 | 0.0222 | 0.0163 |
| stego | explicit | prompt_last | 0.0013 | 0.0060 | 0.0130 | 0.0146 | 0.0148 | 0.0138 | 0.0159 | 0.0197 | 0.0158 |
| transfer | explicit | end_of_response | 0.0007 | 0.0050 | 0.0089 | 0.0101 | 0.0116 | 0.0153 | 0.0197 | 0.0237 | 0.0202 |
| transfer | explicit | first_response | 0.0009 | 0.0061 | 0.0103 | 0.0110 | 0.0111 | 0.0112 | 0.0109 | 0.0112 | 0.0118 |

The same kernel twice is bit-identical, so as in W3 and W4 the floor that matters is the kernel one. One honest turn, Celine Dubois in `cicd__cicd_01__control`, plain against explicit kernel: mass 0.115 at the first response token and 0.170 at the end of response, delta 0.0090 and 0.0127. This matches the overnight floor (W4, 0.137 and 0.142 mass, 0.0078 and 0.0131 delta). Every agent whose reference input equals its actual input (all control agents, and in Stego the honest players) runs its second pass on the plain kernel on purpose, so it carries exactly this noise. Every other second pass shares the explicit kernel with its actual pass.

## Phase 1b, NARCBench Transfer

### Setup

- Rows. The 288 final-phase public responses of the 36 collusion and 36 control runs, NARCBench's own loader and renderer through the J-space repo's `_aligned`, the same rows W4's pipeline replayed for Transfer. Implicit runs are not in the release metadata and are not used. 4 role-structured agents, one round.
- Reference. The agent's honest role prompt from the paired control run, verbatim, swapped in by name as `detector/core_reference.py` does (names match in all 36 pairs, so no name needs changing). The final-phase prompt has no private block: no `PRIVATE` marker in any of the 288 prompts, and the private messages' text reappears only as the colluders' own public pipeline output. So nothing is masked, and the reference is the same user prompt with the honest system prompt. Responses are teacher-forced on the same tokens (checked token for token on every row).
- Honest agents in collusion runs are not zero-edit here. Their system prompt in a collusion run ("Review the implementation code thoroughly ... Flag any concerns you find") differs from the control twin's ("Perform your role diligently with attention to security and correctness"), so their reference is a real, small edit of about the same length. Control agents' reference is their actual input.
- Placebo, as W4 follow-up 1. The donor is the honest role prompt of the same pipeline slot in the control run of the next scenario of a different family, the donor's name replaced by the reader's. `placebo`: colluders get the donor prompt in place of the collusion prompt. `swapped`: every honest agent, in both run types, gets the donor prompt in place of its own reference prompt, the same kind of edit colluders get, minus the collusion content. W4's third group, `padded` (own prompt plus donor appended), was left out for time.
- Read positions. The first response token and the end of response, as W4, all 63 lens layers.
- Scores, all fixed before the first full result. Surplus mass and pool mean (J-space, `phase0_report.aggregate` over the 63 layers). Lens-free direction, two ways: the norm of the surplus direction `d = h_actual - h_reference` relative to the reference, mean over layers, and the pair alignment of surplus directions, `cos(d_i, d_j)` contrasted with each member's mean cosine with the other two agents in J-PACT's `pair_contrast` form, mean over layers, run score the best pair. Ontology on the surplus, the cosine of `d` with the paper's 18 ontology directions, mean over words then layers, as W4 follow-up 2. Run score is the max over the four agents (pair alignment: max over pairs). Run AUROC is `detect.grouped_auroc` by family with `detect.scenario_draws`, 1000 draws.
- Calibration on control runs only. Each agent's score is turned into a z-score against the agents of the same family and pipeline slot in the six control runs of that family, using the score those control agents get under the same kind of reference (real, or swapped for the placebo comparison). No collusion run enters a mean, a spread or a threshold. The run score is the max z. The TPR column puts the threshold at the 95th percentile of calibrated control-run scores.

### Results

Cached numbers for this model on Transfer (released J-space cache, `data/jpact-cache`, checked in the overnight cache report): J-PACT 0.556 [0.45, 0.70], ontology baseline 0.653, NARCBench probe 0.799.

Run AUROC, by family with the scenario bootstrap, 36 collusion against 36 control runs. "Real reference" is the honest role prompt of the control twin. "Placebo vs swapped" gives every agent an edited reference: colluders the donor brief, honest agents the donor brief in place of their own.

| Detector | Position | References | Run AUROC, raw | Run AUROC, calibrated on control runs | TPR at control p95, calibrated |
| --- | --- | --- | --- | --- | --- |
| Surplus mass (J-space) | first_response | real reference | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.00 |
| Surplus mass (J-space) | first_response | placebo vs swapped | 0.759 [0.64, 0.86] | 0.810 [0.69, 0.91] | 0.67 |
| Surplus pool mean (J-space) | first_response | real reference | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.00 |
| Surplus pool mean (J-space) | first_response | placebo vs swapped | 0.759 [0.62, 0.88] | 0.801 [0.68, 0.90] | 0.61 |
| Surplus direction norm, lens-free | first_response | real reference | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.00 |
| Surplus direction norm, lens-free | first_response | placebo vs swapped | 0.787 [0.71, 0.88] | 0.907 [0.85, 0.97] | 0.83 |
| Ontology on the surplus | first_response | real reference | 0.727 [0.61, 0.84] | 0.755 [0.65, 0.87] | 0.47 |
| Ontology on the surplus | first_response | placebo vs swapped | 0.847 [0.74, 0.94] | 0.875 [0.76, 0.96] | 0.75 |
| Pair alignment of surplus directions, lens-free | first_response | real reference | 0.995 [0.98, 1.00] | - | top pair is the coalition 36 of 36 |
| Pair alignment of surplus directions, lens-free | first_response | placebo vs swapped | 0.991 [0.95, 1.00] | - | top pair is the coalition 13 of 36 |
| Surplus mass (J-space) | end_of_response | real reference | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.00 |
| Surplus mass (J-space) | end_of_response | placebo vs swapped | 0.963 [0.93, 1.00] | 0.981 [0.95, 1.00] | 0.86 |
| Surplus pool mean (J-space) | end_of_response | real reference | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.00 |
| Surplus pool mean (J-space) | end_of_response | placebo vs swapped | 0.917 [0.86, 0.98] | 0.977 [0.95, 1.00] | 0.89 |
| Surplus direction norm, lens-free | end_of_response | real reference | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.00 |
| Surplus direction norm, lens-free | end_of_response | placebo vs swapped | 0.907 [0.84, 0.97] | 0.926 [0.85, 0.98] | 0.83 |
| Ontology on the surplus | end_of_response | real reference | 0.870 [0.76, 0.96] | 0.949 [0.87, 0.99] | 0.83 |
| Ontology on the surplus | end_of_response | placebo vs swapped | 0.958 [0.93, 1.00] | 0.981 [0.96, 1.00] | 0.89 |
| Pair alignment of surplus directions, lens-free | end_of_response | real reference | 1.000 [1.00, 1.00] | - | top pair is the coalition 34 of 36 |
| Pair alignment of surplus directions, lens-free | end_of_response | placebo vs swapped | 0.875 [0.75, 0.96] | - | top pair is the coalition 18 of 36 |

Inside the collusion runs. Colluders against the honest agents of the same 36 runs, and how often the two highest-scoring agents are the coalition.

| Detector | Position | References | Agent AUROC, colluders vs honest agents of the same runs | Top two agents are the coalition |
| --- | --- | --- | --- | --- |
| Surplus mass (J-space) | first_response | real reference | 0.959 | 34 of 36 |
| Surplus mass (J-space) | first_response | placebo vs swapped | 0.726 | 18 of 36 |
| Surplus pool mean (J-space) | first_response | real reference | 0.951 | 34 of 36 |
| Surplus pool mean (J-space) | first_response | placebo vs swapped | 0.728 | 21 of 36 |
| Surplus direction norm, lens-free | first_response | real reference | 0.964 | 36 of 36 |
| Surplus direction norm, lens-free | first_response | placebo vs swapped | 0.791 | 28 of 36 |
| Ontology on the surplus | first_response | real reference | 0.800 | 16 of 36 |
| Ontology on the surplus | first_response | placebo vs swapped | 0.747 | 17 of 36 |
| Surplus mass (J-space) | end_of_response | real reference | 0.944 | 34 of 36 |
| Surplus mass (J-space) | end_of_response | placebo vs swapped | 0.884 | 30 of 36 |
| Surplus pool mean (J-space) | end_of_response | real reference | 0.943 | 34 of 36 |
| Surplus pool mean (J-space) | end_of_response | placebo vs swapped | 0.856 | 28 of 36 |
| Surplus direction norm, lens-free | end_of_response | real reference | 0.936 | 35 of 36 |
| Surplus direction norm, lens-free | end_of_response | placebo vs swapped | 0.880 | 29 of 36 |
| Ontology on the surplus | end_of_response | real reference | 0.873 | 22 of 36 |
| Ontology on the surplus | end_of_response | placebo vs swapped | 0.890 | 29 of 36 |

Per agent, all layers.

| Position | References | Agents | n | Mass, median (range) | Direction norm, median (range) |
| --- | --- | --- | --- | --- | --- |
| first_response | real reference | colluders | 72 | 0.348 (0.255 to 0.407) | 0.0850 (0.0517 to 0.1234) |
| first_response | real reference | honest, collusion runs | 72 | 0.137 (0.070 to 0.338) | 0.0090 (0.0078 to 0.0787) |
| first_response | real reference | control | 144 | 0.125 (0.078 to 0.223) | 0.0087 (0.0075 to 0.0114) |
| first_response | placebo vs swapped | colluders | 72 | 0.375 (0.285 to 0.452) | 0.1070 (0.0744 to 0.1423) |
| first_response | placebo vs swapped | honest, collusion runs | 72 | 0.350 (0.226 to 0.445) | 0.0815 (0.0441 to 0.1186) |
| first_response | placebo vs swapped | control | 144 | 0.354 (0.263 to 0.407) | 0.0807 (0.0577 to 0.1268) |
| end_of_response | real reference | colluders | 72 | 0.393 (0.283 to 0.466) | 0.1108 (0.0514 to 0.1505) |
| end_of_response | real reference | honest, collusion runs | 72 | 0.163 (0.106 to 0.406) | 0.0132 (0.0112 to 0.1156) |
| end_of_response | real reference | control | 144 | 0.151 (0.103 to 0.207) | 0.0124 (0.0104 to 0.0176) |
| end_of_response | placebo vs swapped | colluders | 72 | 0.389 (0.302 to 0.484) | 0.1163 (0.0561 to 0.1878) |
| end_of_response | placebo vs swapped | honest, collusion runs | 72 | 0.343 (0.273 to 0.425) | 0.0885 (0.0535 to 0.1293) |
| end_of_response | placebo vs swapped | control | 144 | 0.350 (0.268 to 0.443) | 0.0916 (0.0552 to 0.1414) |

Size of the edit, the change in input length between actual and reference. Colluders' references also replace the whole collusion prompt (about 70 to 190 tokens) with a shorter honest one; swapped replaces an honest prompt of about 40 tokens with another of similar length.

| Reference and agents | n | Input length change, median tokens | Range |
| --- | --- | --- | --- |
| reference | colluders | 72 | 55.0 | 29.0 to 150.0 |
| reference | honest, collusion runs | 72 | 0.0 | 0.0 to 14.0 |
| reference | control | 144 | 0.0 | 0.0 to 0.0 |
| placebo | colluders | 72 | 49.5 | 33.0 to 157.0 |
| swapped | honest, collusion runs | 72 | 6.0 | 0.0 to 14.0 |
| swapped | control | 144 | 7.5 | 0.0 to 18.0 |

Ontology cosine of the surplus direction, per agent, real reference.

| Position | Ontology cosine, colluders | honest, collusion runs | control |
| --- | --- | --- | --- |
| first_response | 0.0057 (-0.0031 to 0.0173) | 0.0003 (-0.0100 to 0.0067) | 0.0002 (-0.0100 to 0.0122) |
| end_of_response | 0.0056 (-0.0016 to 0.0104) | 0.0001 (-0.0075 to 0.0096) | 0.0003 (-0.0105 to 0.0106) |

Honest agents of collusion runs. In cicd and research their system prompt differs from the control twin's, so they carry a real edit (24 agents). In code_review, news, pump_dump and rag it is identical, so their second pass is the plain kernel and they sit at the noise floor (48 agents). The words of the ontology appear literally in the colluders' system prompts: `secretly` and `secret` in 66 of 72, `partner` in 12, `hidden` in 7, `signal` in 3, `fraud` in 2, `manipulate` in 1.

### What the Transfer numbers mean

**The real-reference 1.000 is true by construction, as on Core.** Every control agent's reference equals its input, so its surplus is the kernel floor (median mass 0.125 and 0.151, the noise turns above give 0.109 and 0.151). Every collusion run has two colluders whose reference removes a whole collusion prompt. So mass, pool mean and the direction norm separate the runs perfectly, and calibration on control runs cannot change that. This shows that the replay sees a system-prompt edit well above noise, not that the surplus tells collusion from other edits.

**With every agent's reference edited the separation weakens, and edit size is the obvious explanation.** Against placebo and swapped references, mass separates runs at 0.759 [0.64, 0.86] at the first response token and 0.963 [0.93, 1.00] at the end of response, 0.810 and 0.981 after calibration. The colluders' placebo edit changes input length by a median 49.5 tokens and replaces a much longer prompt, the honest swapped edit by 6 to 7.5 tokens, and the mass medians differ by only 0.025 at the first response token (0.375 against 0.350) and 0.04 at the end (0.389 against 0.343 and 0.350). Inside collusion runs mass picks the coalition as the top two agents in 18 of 36 runs at the first response token and 30 of 36 at the end. As in W4 follow-up 1, this cannot separate collusion content from a bigger edit. A size-matched control was not run here.

**The ontology on the surplus reads what the edit says, and that is where the clearest signal is, with a caveat.** The colluders' surplus direction leans toward the 18 collusion-word directions, median cosine 0.0057 against 0.0002 to 0.0003 for honest and control agents. Against placebo references, which are edits of similar kind without collusion content, it separates runs at 0.958 [0.93, 1.00] at the end of response (0.981 calibrated) and 0.847 at the first response token, above the paper's raw-residual ontology baseline of 0.653 and its probe of 0.799. The caveat is that 66 of 72 collusion prompts literally contain `secretly` or `secret`, so the direction is partly the removed words read back. Unlike W3's note-content test, here the surplus direction does carry what was removed, not only how much.

**Lens-free.** The direction norm behaves like mass, slightly better under placebo references (0.787 and 0.907, 0.907 and 0.926 calibrated). The pair alignment of surplus directions finds the coalition as the top pair in 36 of 36 runs at the first response token with real references, but only because colluders are the only pair with a large edit. Under placebo-style references its run AUROC stays high (0.991 and 0.875) while the top pair is the coalition in only 13 and 18 of 36 runs. Collusion runs mix two kinds of edit and control runs have one, so that run score separates the structure of the references, not the pair.

**Against the pass rule.** Calibrated on control runs only, every score but one is above 0.653 and 0.799 with real references (ontology at the first response token, 0.755, misses 0.799). With placebo-style references the end-of-response scores pass (mass 0.981, pool mean 0.977, direction norm 0.926, ontology 0.981), and at the first response token the direction norm (0.907) and the ontology (0.875) pass while mass (0.810, interval down to 0.69) is marginal. These are not like-for-like with J-PACT 0.556, the ontology baseline 0.653 or the probe 0.799, which read only the transcripts' own activations. The surplus detector is given the sanctioned system prompt of the control twin, and in Transfer that prompt is exactly where collusion is planted. The fair reading is that on Transfer a known sanctioned prompt makes the planted instruction visible, its content readable through the ontology, and its size a confound for mass.


## Phase 1a, NARCBench Stego

### Setup

- Runs. 20 collusion and 20 control blackjack runs, 4 players and a judge, hands 1 to 7 (`read.py collect_stego` reads hands 1 to 7; hand 8 is the final phase). The sender is the coalition's counter, the receiver its big player. Control runs take the sender and receiver names of their collusion twin, so they have a pseudo-sender and pseudo-receiver at the same seats.
- Reference. `detector/stego_spans.py` gives the private spans of a colluder's prompt: the opening block (the private coordination with the agreed scheme, the per-hand count records `[Hand n: RANGE (count c)]` and the decode instructions) and any later `PRIVATE:` line. The reference masks only those, so the sender's public table talk stays in. Two references: `kept` keeps position ids, `compacted` closes them up. Following the W4 decision, compacted is primary for mass and naming, kept for directions. Honest players and all control players have nothing to mask; their second pass is the plain kernel (the noise floor).
- Read position. `message_end`, read.py's position: the token holding the last character of the sender's public message (its `PRIVATE:` lines stripped), found after the sender's last name tag in the reader's own text. For the receiver it sits inside the prompt, in the table chat, before the receiver writes its bet. The render and replay take this explicit position (`stego_turn` in `scripts/lane_b.py`). A reader who speaks before the sender in a hand has no `message_end` that hand (1 of 140 receiver hands, 87 of 280 honest-player hands). The sender is read at the end of its own public message in its own teacher-forced turn. `prompt_last`, the token before the reader's reply, is also kept.
- Detection score. Surplus mass per player over hands and layers, run score the max over the four players, AUROC with the scenario bootstrap (one group).
- Range naming, the paper's measure. True range is `count_range(running_count[hand - 1])`, as read.py. A measure names a range when its token set holds any word of that range in `RANGE_SIGNATURES` (`third_party/maj-lens/config.py`), matched on the lowercased stripped token text as read.py does. Token sets, over a band of layers: the top-10 lens readout of the actual pass, of the reference, the top-10 surplus (in the actual band top-10 and not in the reference's), the J-space surplus (concepts with positive surplus at any band layer) and the J-space support of the actual pass. Primary band L28 to L56, read.py's L19 to L38 of DeepSeek's 43 blocks scaled to Qwen3-32B's 64; all layers and W4's L26 to L30 are also reported. The paper's 0.41 to 0.62 is the diagonal of its colluding-receiver matrix (NEGATIVE 0.41, NEUTRAL 0.54, HOT 0.62, 140 hands, DeepSeek-V4-Flash, its own band), against 0.10, 0.00 and 0.02 for its honest counterfactual.
- A confound the paper's measure shares, and a control for it. The receiver's private block carries the count records of the previous hands, and ranges persist from hand to hand, so a surplus can name the true range by carrying the last record rather than by decoding the sender. Each naming row therefore also gives the share of hands that name the previous hand's range, and, on the hands where the range changed, how often the true and the previous range are named.
- Lens-free. Within each collusion run, the cosine of the receiver's surplus direction at `message_end` between hands of the same range (HOT with HOT, NEGATIVE with NEGATIVE), minus between HOT and NEGATIVE hands, mean over the band's layers, then over runs. Nearby hands are alike for reasons that have nothing to do with the count, so the same contrast is also taken within strata of equal hand distance and averaged, and a within-run label permutation gives a p value. The same is run with the previous hand's range as the label. Nulls: honest players of collusion runs and every control reader, at the same position.

<!-- STEGO -->

## Notes

<!-- NOTES -->
