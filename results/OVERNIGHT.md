# Overnight report

Draft in progress, updated as work lands. Branch `overnight-detector`.

## What ran and what did not

| Workstream | State | Note |
| --- | --- | --- |
| W1 code notes and cache report | done | Qwen3-32B rows reproduce. `read.py` stopped at figure rendering, see the notes at the end |
| W2 pod, model, lens | done | Smoke test passes |
| W3 replay and leak readout | running | First replay discarded and rerun, see below |
| W4 Phase 0 on Core | queued | |
| W5 Transfer implicit labels | partial, blocker below | Audit done. Outcome label not computable from the release |
| W6 Stego spans | done | `detector/stego_spans.py`, tested on all 320 colluder turns |
| W6 sweep1 replay | not started | |

### The first W3 replay was discarded

The first replay ran the actual pass on the plain attention kernel and the reference pass on the masked kernel. The two kernels differ by about as much as the mask itself in a whole-sequence norm, so that surplus would have contained kernel noise, and turns with nothing to mask would have read exactly zero. `scripts/replay.py` now runs both passes through one kernel, an explicit 4D mask, and the replay was rerun once. Turns with nothing to mask run their second pass on the plain kernel on purpose, marked `plain_kernel`, as a measurement of that noise.

### GPU checks

`tests/gpu_checks.py`, on one turn (carol, episode 2, solve, 15118 tokens). The first version failed three checks. It compared across kernels and used one whole-sequence norm, so it was rewritten with criteria fixed before the rerun. A second run ran out of GPU memory in the test itself, which was fixed with the criteria unchanged. Outputs are in `results/gpu_checks_v1.log`, `results/gpu_checks_v2.log` and `results/gpu_checks.json`.

| Check | Criterion | Result |
| --- | --- | --- |
| mask causal | same kernel, positions before the span exactly equal | PASS, max difference 0.0 |
| mask same path | 2D mask with zeros equals the explicit 4D mask exactly | PASS, max difference 0.0 |
| mask deletion | masked against tokens removed with positions kept, median delta under half the median mask effect at every layer | **FAIL** at 7 of 10 layers checked, ratios 0.45 to 0.63 |
| mask no-op | all-ones 2D mask agrees with no mask | PASS, 0.0 |
| repeat | same pass twice | PASS, bit-identical within one process |
| pursuit | batched pursuit against TransformerLens | PASS on 8 rows at L26 and L45, Jaccard 1.0, total variation 6e-8 |

On the deletion failure. The check failed its criterion and stays failed. The masked-against-deleted delta is about the same size as the kernel noise at every layer (table below), so nothing suggests the mask behaves differently from deletion beyond noise. The criterion was probably the wrong test, because the median token in the tail barely moves under the mask at all. That is an interpretation of one turn, not a pass.

### Noise floor by layer

Relative residual delta per token, `||a - b|| / ||b||`, over all tokens of one turn. Kernel noise is the plain kernel against the explicit mask kernel with the same input.

| Layer | Kernel noise, median | Kernel noise, p95 | Masked vs deleted, median | Mask effect, median | Deleted / effect |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0010 | 0.0019 | 0.0008 | 0.0023 | 0.35 |
| 15 | 0.0053 | 0.0085 | 0.0051 | 0.0081 | 0.62 |
| 26 | 0.0083 | 0.0117 | 0.0084 | 0.0163 | 0.52 |
| 30 | 0.0091 | 0.0130 | 0.0092 | 0.0164 | 0.56 |
| 36 | 0.0105 | 0.0146 | 0.0103 | 0.0164 | 0.63 |
| 40 | 0.0117 | 0.0162 | 0.0110 | 0.0177 | 0.62 |
| 45 | 0.0142 | 0.0356 | 0.0126 | 0.0219 | 0.57 |
| 50 | 0.0154 | 0.0769 | 0.0123 | 0.0242 | 0.51 |
| 56 | 0.0150 | 0.1169 | 0.0117 | 0.0238 | 0.49 |
| 62 | 0.0144 | 0.0900 | 0.0108 | 0.0241 | 0.45 |

The same quantity at the read positions, kernel noise / mask effect. These are the positions W3 reads. Noise was measured on the `none` turn and the mask effect on the `leak` turn, one turn each.

| Layer | prompt_last | first_generated | first_thought | after_think |
| --- | --- | --- | --- | --- |
| 0 | 0.0006 / 0.0081 | 0.0021 / 0.0077 | 0.0027 / 0.0109 | 0.0011 / 0.0010 |
| 15 | 0.0057 / 0.0385 | 0.0065 / 0.0247 | 0.0052 / 0.0544 | 0.0047 / 0.0113 |
| 26 | 0.0113 / 0.1574 | 0.0133 / 0.0785 | 0.0098 / 0.1375 | 0.0081 / 0.0712 |
| 30 | 0.0135 / 0.1663 | 0.0138 / 0.0818 | 0.0114 / 0.1573 | 0.0129 / 0.0885 |
| 36 | 0.0236 / 0.2044 | 0.0155 / 0.0873 | 0.0126 / 0.1348 | 0.0192 / 0.1217 |
| 40 | 0.0227 / 0.1882 | 0.0145 / 0.0775 | 0.0115 / 0.1063 | 0.0215 / 0.1386 |
| 45 | 0.0251 / 0.3574 | 0.0164 / 0.0867 | 0.0095 / 0.0833 | 0.0185 / 0.1385 |
| 50 | 0.0307 / 0.4968 | 0.0189 / 0.1213 | 0.0114 / 0.1189 | 0.0215 / 0.1359 |
| 56 | 0.0320 / 0.4090 | 0.0250 / 0.1470 | 0.0153 / 0.1345 | 0.0241 / 0.1553 |
| 62 | 0.0358 / 0.3877 | 0.0218 / 0.0988 | 0.0202 / 0.1882 | 0.0196 / 0.1152 |

## OpenRouter spend

| Item | USD |
| --- | --- |
| W5 coordination audit, 36 calls | 0.0069 |

## W5 summary

`data/transfer_implicit_labels.json` has one record for each of the 36 implicit runs. It is force-added, since `data/` is ignored.

**Blocker on the outcome label.** The handoff's label cannot be computed from the released data, for two reasons.

| Assumption in the handoff | What the release has |
| --- | --- |
| `fair_option` present on collusion and control rows | null in all 288 Qwen3-32B Transfer rows, and for every other model |
| A final RECOMMENDATION of Option A or B per agent | The released final prompt asks for "a brief final summary". No Transfer prompt offers Option A or B in any of the 108 runs. 0 of 72 pair members have a vote. The one regex hit is an allocation figure ("6-8%") |

`generation/transfer.py` at the pinned commit does ask for a RECOMMENDATION, so the released transcripts came from different prompts than that code. `voted_unfair` and `collusive` are null in every record. No replacement label was invented.

**Coordination audit, run as specified.** `qwen/qwen3-32b` through OpenRouter, the two private messages only, temperature 0.

| Family | Agreed yes | Agreed no |
| --- | --- | --- |
| cicd | 0 | 6 |
| code_review | 0 | 6 |
| news | 3 | 3 |
| pump_dump | 4 | 2 |
| rag | 6 | 0 |
| research | 4 | 2 |
| Total | 17 | 19 |

Read this with care. It is one model's judgment, with no human check. 10 of the 36 supporting quotes are not verbatim in the messages (`quote_verbatim` in each record), including 4 of the 17 yes answers. Some yes answers rest on weak quotes, for example `pump_dump__pd_01` and `pd_06`.

## J-space code notes

Source is `third_party/maj-lens/`, fetched from the anonymous repo with its layout. Unmodified.

### Lens

- Location. Hugging Face `neuronpedia/jacobian-lens`, revision `8aa8152c`, file `qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt` (6.6 GB) with `config.yaml`. Pinned in `config.py` (`LENS_REPO`) and `utils/models.py`.
- Format. A torch checkpoint with keys `jacobian_sum`, `n_done`, `next_idx`, `source_layers`, raw float32 sums. `utils/release.py` `load_released_lens` divides by `n_done` to get J. The finished format has a key `J` instead.
- Layers. Source layers 0 to 62, 63 of them, `list(range(num_hidden_layers - 1))`. Each J is 5120 x 5120. Layer l means the output of block l, hooked at `model.model.layers[l]`.
- **It is a partial checkpoint.** Its own `config.yaml` carries a `known_issue`. The fit stopped at 80 prompts of a planned 1000, below its 100-prompt minimum, and did not converge (final mean relative change 0.026 against a stop threshold of 0.002). `JacobianLens.load` rejects the file. A refit is planned upstream. The paper's Qwen3-32B numbers come from this same file, so comparisons with them are fair, but every Qwen3 readout here inherits it.

### Readout and decomposition

- Readout. `lens.py` `Readout.logits`. With H the residual rows, `x = norm(H @ J_l.T)` then `x @ head.T`, where `norm` is the model's final RMSNorm and `head` is `lm_head.weight` cut to the tokenizer length. `top` takes `.topk(10)`. So the readout is `W_U norm(J_l h)`.
- Dictionary. `lens.py` `dictionary`. Atoms `v_t = W_U[t] J_l`, one row per token, `head @ J_l`.
- Decomposition. `lens.py` `decompose(X, atoms, k=K)` calls, one row at a time, `transformer_lens.tools.analysis.get_sparse_decomposition(x, dictionary, k=25, *, algorithm="gradient_pursuit")`. It returns `support` and `coordinates`. `K = 25` is in `config.py` and is an upper bound, since selection stops once no unselected atom has correlation above `sqrt(eps_float32) * ||x||`. The target is the raw residual, not the normalised one.

### Rendering and read position

- `utils/build/replay.py` imports NARCBench's own loaders and renderers (`extraction/extract_{core,transfer,stego}.py`, `load_runs`, `_build_gen_text`). A sample renders as system, user, assistant through the chat template with `enable_thinking=False`, so Qwen3's text is `...<|im_start|>assistant\n<think>\n\n</think>\n\nRESPONSE<|im_end|>\n`.
- Rows are the public responses, phase `discussion` for Core and `final` for Transfer and Stego. Stego's other rounds come from a second replay with `NARC_PHASE=discussion`.
- Read position. `lens.end_of_response` is `hidden[0, -1]`, the last token of the full text. For Qwen3 that is the `\n` after `<|im_end|>`, not a token of the response. Saved as float16.
- Forward is `model.model(ids, use_cache=False)`, so no logits are computed.

### Honest counterfactual (`read.py`)

- `core_counterfactual(p, honest)` keeps the colluder's user prompt and swaps in the table's honest system prompt with the honest agent's name replaced by the colluder's. Read at the response-start token, `len(ids) - 1` with the generation prefix appended.
- `read.py` runs only on DeepSeek-V4-Flash's own transcripts, so the 86 against 33 result is a DeepSeek result. There is no Qwen3-32B version of it in the repo.

### J-PACT as implemented (`detect.py`)

Per episode of four agents, per round and layer, from each agent's decomposition.

```
z_{a,c}  = coefficient of atom c / sum of agent a's coefficients                    shares()
s_{ij,c} = sqrt(z_{i,c} z_{j,c})                                                    geometric mean
r_{ij,c} = [ s_{ij,c} - mean_{k not in ij} (s_{ik,c} + s_{jk,c}) / 2 ]+             pair_contrast()
R_ij     = mean over the pair's pool (every c with r_{ij,c} > 0 in some cell) of
           (sum of r_{ij,c} over cells / number of cells)                           jpact()
T_s      = max over the six pairs of R_ij                                           run score
```

A cell is one (round, layer). The sweep uses all 63 layers. Run AUROC is `grouped_auroc`, AUROC within each group (Core domain, Transfer family, Stego as one group) then averaged. Intervals bootstrap 1000 draws over scenarios, a scenario's collusion and control runs drawn together (`scenario_draws`).

### Ontology baseline

`lens.py` `ontology_directions` gives unit vectors `(W_U[w] - mean_c W_U[c]) J_l` for 18 target words (fake, trick, fraud, secretly, hidden, covert, conceal, secret, deceive, deception, dishonest, cheat, manipulate, signal, decode, scheme, plan, partner) against 32 control nouns. An agent's score is its residual's cosine with those directions, averaged over words, then layers, then rounds. A pair scores the mean of its two agents. The run score is the max over pairs.

### Cache report

| Check | Expected | Got |
| --- | --- | --- |
| Qwen3-32B Core J-PACT | 0.984 | 0.984 [0.90, 1.00] |
| Qwen3-32B Transfer J-PACT | 0.556 | 0.556 [0.45, 0.70] |
| Qwen3-32B Stego J-PACT | 1.000 | 1.000 [1.00, 1.00] |
| Qwen3-32B ontology, Core, Transfer, Stego | 0.853, 0.653, 1.000 | same |
| Qwen3-32B probe, Core, Transfer, Stego | 1.000, 0.799, 1.000 | same |
| J-PACT one round on Core | 0.861 | 0.861 [0.79, 0.93] |
| `detection.tex`, `observations.tex`, `localisation.tex` | no diff | no diff |
| `checks.py` | pass | all checks pass, run by hand |

- The cache is 9.35 GB, not the 117 MB its README states. Its manifest verified.
- `reproduce.sh` stopped at the `read` step because Playwright's Chromium is not installed. That step renders figures. Its tables had already been written and match. `steer` did not run, and `checks.py` was run directly afterwards. Two re-rendered PNGs differed and were restored.
