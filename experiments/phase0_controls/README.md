# Phase 0 and controls, Qwen3-32B

The first GPU run. It asks whether the J-space paper's J-PACT score reproduces from public inputs, what the surplus measures on NARCBench Core once placebo and size-matched controls are added, and whether a planted leak is read at the positions prior work reads. It also holds the runs of the pilot and of sweep1.

In the paper, Appendix B (the released benchmarks, run and not claimed) and Appendix C (kernel noise, the floor and the two references).

The main findings. J-PACT reproduces from public inputs (Core 1.000 against 0.984 cached). Surplus mass on Core is mostly edit size, a size-matched honest removal brings its AUROC to 0.448 at the end of a response. The leak is not read at the response start but at its first mention in the reasoning. Two attention kernels differ by as much as the mask itself, so both passes must run on one kernel.

## Rerun

Generate the pilot and sweep1 through OpenRouter, about 4 USD for sweep1.

```bash
python scripts/run_plants.py --seed-list 1000 --out runs/pilot
python scripts/run_plants.py --seed-list $(seq 2000 2009) --k 1 3 --out runs/sweep1
```

The blind agent has rotated with the seed since sweep14b, so these seeds no longer put a code specialist blind in every run as they did for sweep1. To keep that design, take seeds as sweep14b does, every third one.

Then, on a card set up with `scripts/pod/setup.sh` and `scripts/pod/inputs.sh`, the jobs in the order they ran.

| Job | What it does |
| --- | --- |
| `jobs/w3.sh` | replays episode 2 of the pilot with the kernel pinned, decomposes it and reads the leak |
| `jobs/w4_jspace.sh` | runs the J-space pipeline from public inputs |
| `jobs/w4_surplus.sh` | runs our reference pass on Core, and the noise floor |
| `jobs/w4_followups.sh` | placebo references, ontology on the surplus, the edge test |
| `jobs/w6.sh`, `jobs/w6_sweep.sh` | the first-mention position on the pilot, then the sweep1 replay |
| `jobs/w4_matched.sh` | honest references with a size-matched removal |
| `jobs/w7.sh` | the direction test, part B on sweep1, the direction edge on Core |
| `jobs/w4_dose.sh` | key-sentence dose references and token edits |

`transfer_labels.py` audits NARCBench Transfer's implicit runs through OpenRouter. It costs under a cent. Each script names its inputs and outputs in its docstring.
