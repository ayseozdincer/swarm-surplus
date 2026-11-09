# Lane G, which of two notes was used, pre-registered

Lane D found on ten turns that the surplus direction names the note an agent used, at a position chosen after several had been seen. This fixes the position in advance and adds twenty new seeds, generated after every threshold and the fitted model were fixed.

In the paper, Section 4 and Table D7. The same seeds are the out-of-sample test of the rule, 17 of 17 and 0 of 20.

The result. Lane D's 10 of 10 does not replicate at that rate. The pre-registered test gives 11 of 17 new turns, 21 of 27 pooled.

**The pre-registration** is the first section of [REPORT.md](REPORT.md). It was committed as `595d9a1` before any label or replay array of the new seeds existed, and git keeps that commit and its timestamp in this repository's history.

## Rerun

On a generation card set up with `scripts/pod/vllm_setup.sh`.

```bash
bash experiments/lane_g_two_notes/jobs/generate.sh
bash experiments/lane_g_two_notes/jobs/replay.sh
```

`generate.sh` writes `runs/leak2_name` and `runs/leak2_broad`, and `replay.sh` merges them into `runs/lane_g/sweep2_leak2` with `merge_runs.py`, replays them with lane D's `replay_sweep2.py` and scores them with `score_leak2.py`.

Scoring needs the full replay arrays, which are not in the data bundle. The bundle's `data/replay_lane_g/` holds what `slim_traces.py` keeps, enough for `paper/build/score_newdata.py`.
