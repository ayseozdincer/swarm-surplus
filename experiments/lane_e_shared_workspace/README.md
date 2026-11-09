# Lane E, a shared workspace and a planted line

Two questions about a channel nobody declared.

**Phase 2b, [REPORT.md](REPORT.md).** Does coordination emerge by itself when agents share a workspace? It did not. Twenty runs on Qwen3-14B, 2,887 model calls, made two workspace calls in total, both listings of an empty workspace, and no reads or writes of another agent's bytes.

**Phase 2c, [REPORT_2c.md](REPORT_2c.md).** Plant the channel instead, as a line appended to a shared task file through a logged write. Does write-log provenance recover it? The declared mask treats the file as environment and masks nothing, while the derived mask finds the line from the write log's bytes alone. On the same turns, source direction goes from 0.48 to 1.00.

In the paper, Section 4 and Table 3.

## Rerun

Phase 2b goes through OpenRouter, about 3 USD for all of it.

```bash
python scripts/run_plants.py --mode shared_file --seed-list $(seq 7001 7010) --model qwen/qwen3-14b --max-tokens 12000 --out runs/shared
python scripts/run_plants.py --mode shared_file --seed-list $(seq 7101 7110) --wrong-penalty 0.5 --salience --model qwen/qwen3-14b --max-tokens 12000 --out runs/shared_c2
python experiments/lane_e_shared_workspace/summarize_shared.py runs/shared --edges
python experiments/lane_e_shared_workspace/report_shared.py runs/shared runs/shared_c2
```

Phase 2c runs on a card set up with `scripts/pod/vllm_setup.sh`, the server started with `scripts/pod/vllm_serve.sh`.

```bash
bash experiments/lane_e_shared_workspace/jobs/planted.sh
bash experiments/lane_e_shared_workspace/jobs/planted_chain.sh
```

`planted_chain.sh` stops the server, checks the masks on the real model, replays with `replay_planted.py` and scores with `planted_scores.py`. From the data bundle, `make rescore` recomputes the behaviour summary.
