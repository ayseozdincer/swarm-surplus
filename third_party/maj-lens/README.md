# Reproduction

Every table and figure in `results/` is written by the code here.

```bash
pip install -r env/requirements.txt
hf download throwaway-reproducibility-354364563/jpact-cache --repo-type dataset --local-dir <cache>   # the cache, 117 MB
CACHE_DIR=<cache> bash reproduce.sh
```

The cache holds only what this code computes (and DeepSeek-V4-Flash's generated transcripts); NARCBench's release
metadata is fetched from its pinned dataset, and model and lens weights are not needed for the report steps.

`CACHE_DIR` becomes `artifacts/`. If `<cache>/MANIFEST.sha256` verifies, only the report steps run, on CPU; `git diff
results/` is then empty. Otherwise every step runs from public inputs on a GPU and fills `<cache>`; a step whose outputs
are present reuses them. `checks.py` then recomputes the headline numbers independently and compares them with the
tables.

Replayed activations are not bit-identical across runs, even on the same GPU, so the detection numbers of a run from
public inputs can differ from `results/` (most for GPT-OSS-20B's Core split). The report steps reproduce `results/`
exactly from the released cache.

## Layout

The scientific content is in the top-level files; `utils/` holds I/O, table and figure drawing, model loading and
NARCBench's code; `utils/build/` replays NARCBench's transcripts through each model and stages the resulting activations.

| File | Content |
| --- | --- |
| `config.py` | every setting the paper states |
| `lens.py` | the J-lens readout, the residual it reads, the sparse J-space decomposition, the ontology directions |
| `detect.py` | J-PACT, the ontology baseline, NARCBench's probe, AUROC and bootstrap intervals, localisation |
| `read.py` | what the colluders hold: the workspace band, Core colluders and their counterfactuals, the Stego receiver |
| `steer.py` | the lens-coordinate swap in the Stego receiver and its random-pair control |
| `cache.py` | the cache directory and its manifest |
| `checks.py` | independent recomputation of the headline numbers |
| `reproduce.sh` | the order of the steps |

## Claims to code

| Paper | Output | Code |
| --- | --- | --- |
| Ontology table (appendix) | written in the paper | words: `ontology.json`; directions: `lens.py` `ontology_directions`; score: `detect.py` `sweep` (the ontology baseline) |
| Detection table | `results/main/detection.tex` | `detect.py`: `sweep`, `jpact`, `pair_contrast`, `shares`, `probe_job`, `report` |
| Observations table | `results/main/observations.tex` | `detect.py`: `sweep`, `report` |
| Localisation table | `results/main/localisation.tex` | `detect.py`: `localisation` |
| Workspace band | `results/interp/fig_band.png` | `read.py`: `collect_band` |
| Read (Core, Stego) | `fig_quant.png`, `fig_core_transcript.png`, `fig_stego_decoding.png`, `core_commitment.tex`, `core_pairs.tex`, `stego_decoding.tex` | `read.py` |
| Steer | `stego_swap.tex`, `fig_steer.png` | `steer.py` |
| Figure 1 | `fig_overview.png` | `read.py`, `steer.py`; drawn in `utils/figures.py` |

## Inputs (pinned in `config.py` and `utils/models.py`)

NARCBench's code and dataset; the released Jacobian lenses (Neuronpedia); Anthropic's jacobian-lens code; the four
model checkpoints. NARCBench's LR probe and aggregators are imported unchanged (`utils/narcbench.py`); TransformerLens's
`get_sparse_decomposition` is the decomposition (`lens.py`). DeepSeek-V4-Flash's own transcripts are generated with
NARCBench's generation code at temperature 0.7 (`utils/generate_cohort.sh`, vLLM 0.30.0) and are in the cache, since
resampling gives a different sample.
