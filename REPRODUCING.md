# Reproducing the results

Three levels, from a minute on a laptop to about 40 card hours on rented GPUs. Each level checks itself against what is committed.

Python 3.11 or later. All commands run from the repository root.

```bash
make setup
source .venv/bin/activate
```

## Level 1, no data and no GPU

```bash
make test
make mock
make check
```

`make test` runs the offline tests. A few skip because they need NARCBench, torch or a tokenizer, and they say so.

`make mock` runs the whole environment on a mock model and prints the behaviour table, which shows the harness, the plants, the provenance tags and the labels working without an API key.

`make check` rebuilds the tables that come from per-turn scores committed under `paper/tables/sources/` and `experiments/*/results/`, and fails if any of them differs from the committed version.

| Table | Script |
| --- | --- |
| `count_pooled.md` | `paper/build/table_count_pooled.py` |
| `false_positives.md` | `paper/build/table_false_positives.py` |
| `jump_per_turn.md` | `paper/build/table_jump_per_turn.py` |
| `heldout_rule.md`, Table 2 | `paper/build/table_heldout_rule.py` |
| `transfer_fitted.md` | `paper/build/transfer_fitted.py` |

## Level 2, the data bundle and no GPU

The bundle holds every transcript as served with its provenance tags (`runs/`) and the per-token traces the paper reads (`data/`). It is about 1 GB.

```bash
scripts/fetch_data.sh
make check-all
make rescore
```

`make check-all` adds the tables that read transcripts or traces.

| Table | Script |
| --- | --- |
| `census.md` | `paper/build/table_census.py` |
| `text_baseline.md` | `paper/build/table_text_baseline.py` |
| `fitted_detector.md`, `fitted_model.json` | `paper/build/fit_detector.py` |
| `sensitivity.md`, Table D10 | `paper/build/table_sensitivity.py` |
| `newdata_test.md`, Table D7 | `paper/build/score_newdata.py` |

`make rescore` recomputes per-experiment scores from those inputs and fails if the outputs differ from what is committed. Every one of these was checked and reproduces byte for byte.

The insider-trading scores also recompute from the bundle, with `python experiments/external_insider_trading/scores.py --lp-cut 2.0`. They need the Qwen3-14B tokenizer from Hugging Face, and the only difference from the committed file is the local path recorded under `fitted_model`.

### What the bundle does not hold

The full replay arrays are far too large to ship, about 170 MB per turn and tens of GB per experiment. The bundle keeps slim traces instead, which hold the per-token relative surplus. Scores that need the full residuals, such as direction and J-space shares, are committed under `experiments/*/results/` as computed, and recomputing them needs a replay at Level 3.

NARCBench's data is not ours and is not in the bundle. `scripts/pod/inputs.sh` fetches it with the other pinned inputs through the J-space code's own step.

## Level 3, regenerating

This needs a GPU with 80 GB of memory and, for some experiments, an OpenRouter key.

```bash
cp .env.example .env
```

Fill in `OPENROUTER_API_KEY` and `RUNPOD_API_KEY`. Then set up a card as described in [scripts/pod/README.md](scripts/pod/README.md), and follow the README of the experiment you want. The `jobs/` folder of each experiment holds the exact commands as they ran, with their seeds and flags.

Every experiment has the same three steps.

1. **Generate.** `scripts/run_plants.py` runs the swarm against a model and writes `runs/<name>/seed<S>/<branch>/` with every call as served and its provenance tags.
2. **Replay.** `scripts/replay.py`, or an experiment's own replay script, runs each turn twice on one explicit attention kernel and writes arrays under `data/`.
3. **Score.** The experiment's scripts read those arrays and write `experiments/<name>/results/`.

Costs as spent for the paper, from Appendix F.

| Experiment | Card | Hours | USD |
| --- | --- | --- | --- |
| phase 0 and controls, 32B | A100 80 GB | 9.15 | 16.8 |
| lane B, Transfer and Stego, 32B | A100 80 GB | 2.78 | 4.98 |
| lane C, 14B replay of sweep14b | A100 80 GB, three cards | 4.91 | 8.77 |
| lane D, sweep2 generation, replay, extra arms | H200 and A100, five cards | 18.1 | 52.3 |
| lane E, phase 2b and 2c | OpenRouter, one A100 | 3.72 | 6.66 plus 2.9 |
| lane F, reader replay, compacted check, 48 extra positives | A100 80 GB, two cards | about 1.3 | 2.16 |
| lane G, twenty leak2 seeds, generation and replay | H200 141 GB | 1.95 | about 7.0 |
| TaskTracker-shaped set, 600 turns | A100 80 GB | 0.5 | 0.88 |
| insider-trading eval, 357 turns | A100 80 GB, OpenRouter judge | 0.47 | about 0.96 |
| sweep1 and sweep14b generation | OpenRouter | | about 8 |

### What to expect from a rerun

Generation samples at temperature 0.7, so a rerun gives different transcripts and the counts will move within sampling noise. Replay of a given transcript is deterministic only on one kernel, the same pass twice is bit-identical, while two attention kernels differ by 0.1 to 5 percent of the residual by layer. That is why both passes run on one explicit mask kernel, and why floors are measured rather than assumed (Appendix C of the paper).

Levels 1 and 2 start from the transcripts as served, so they reproduce the paper exactly.
