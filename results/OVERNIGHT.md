# Overnight report

Draft in progress, updated as work lands. Branch `overnight-detector`.

## What ran and what did not

| Workstream | State | Note |
| --- | --- | --- |
| W1 code notes and cache report | done | Qwen3-32B rows reproduce. `read.py` stopped at figure rendering, see notes |
| W2 pod, model, lens | done | Smoke test passes |
| W3 replay and leak readout | running | |
| W4 Phase 0 on Core | not started | |
| W5 Transfer implicit labels | partial, blocker below | Audit done. Outcome label not computable from the release |
| W6 Stego spans, sweep1 replay | not started | |

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
