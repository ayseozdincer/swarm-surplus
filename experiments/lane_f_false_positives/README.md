# Lane F, false positives by task type

Does the rule fire on agents that read a note not meant for them? It also checks the compacted reference, wrong-note readers and 48 more positive turns, and it leaves the slim traces that the fitted detector and the sensitivity table read.

In the paper, Section 4, Tables 2, D3 and D10.

The main findings. None of the 260 reader turns passes the gate, and the confirmation fires on 4 of 259, all data readers. The 16-token window is the score with false positives.

## Rerun without a GPU

From the data bundle.

```bash
python experiments/lane_f_false_positives/scores.py
```

This writes `results/lane_f_scores.json` and the tables beside it. `make rescore` runs it and compares the result with what is committed.

## Rerun from scratch

On a card set up with `scripts/pod/replay_setup.sh`, over the transcripts of `runs/sweep2`.

```bash
SWARM_MODEL=qwen3_14b SWARM_ARTIFACTS=/root/artifacts python experiments/lane_f_false_positives/replay_traces.py runs --out-root data/replay_lane_f --units blind nonblind_a nonblind_b solvable
bash experiments/lane_f_false_positives/jobs/extra_positives.sh
```
