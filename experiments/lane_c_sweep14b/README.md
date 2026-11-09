# Lane C, the leak on Qwen3-14B with a converged lens

Qwen3-32B's released lens is a partial checkpoint, so this repeats the leak read on Qwen3-14B, whose lens has converged. It also adds the per-token trace, the designer-span placebo and verdict turns of users against exposed non-users.

In the paper, Section 4 and Tables D1 and D2. These ten seeds are the calibration set the rule was fixed on, together with the pilot.

The main findings. The converged lens does not name the leak at the response start either, so the earlier null on 32B was not the partial lens. The per-token trace does read the leak at its first mention.

## Rerun

Generate through OpenRouter, about 4 USD.

```bash
python scripts/run_plants.py --seed-list $(seq 3000 3 3027) --k 1 3 --model qwen/qwen3-14b --max-tokens 12000 --out runs/sweep14b
```

Then, on a card set up with `scripts/pod/setup.sh` and `scripts/pod/inputs_14b.sh`, the jobs in the order they ran.

| Job | What it does |
| --- | --- |
| `jobs/replay_a.sh` | the kernel floor on one honest turn, then a first replay that `stages.sh` superseded |
| `jobs/stages.sh` | stage 1, the blind agent's solve turn in every branch with the per-token trace and the placebo |
| `jobs/verdict.sh`, `jobs/pod2.sh` | stage 2, the free agents' verdict turns, split between two cards |
| `jobs/final.sh` | the final readout over both stages |

| Script | Writes |
| --- | --- |
| `kernel_floor.py` | `results/kernel_floor.json` |
| `blind.py` | the stage 1 replay |
| `window.py` | `results/window.json`, the count and window that feed the paper |
| `found.py` | `results/found.json`, whether the window lands on the mention |
| `users.py` | `results/users.json`, users against exposed non-users |
| `score.py` | `results/readout.json`, stored here as `readout.json.gz` |
| `report.py` | `results/tables.md` |
