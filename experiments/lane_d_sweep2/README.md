# Lane D, sweep2 on a self-served Qwen3-14B, and the extra arms

The main game. Six agents, a confidentiality rule, code-task blind agents and matched branches, with the arms that test what the signal means.

In the paper, Section 4 and Table 2, with Tables D1 to D3 and Figure 1. The figure's trace is `results/trace_seed5034.csv`.

| Runs | Arm | Question |
| --- | --- | --- |
| `runs/sweep2` | none, benign, leak, forbidden leak, command | the clean positive, and whether a rule changes behaviour |
| `runs/sweep2_norule` | the same with the rule off | the rule's effect, paired |
| `runs/sweep2_forbidden2` | forbidden leak, new seeds | held-out positives, and concealment |
| `runs/sweep2_leak2` | two conflicting notes | which note was used, on ten seeds |
| `runs/sweep2_multi` | one note, three blind targets | several agents taking up at once |
| `runs/sweep2_solvable` | a leak on a solvable task, right or wrong | does a capable agent copy, or check |
| `runs/sweep2_methodtip` | a correct procedure with no value | is the plateau copying or engagement |

## Rerun without a GPU

From the data bundle, `make rescore` recomputes the behaviour tables and fails if they differ from what is committed. Scores that need full replay arrays are committed here as computed.

## Rerun from scratch

On a card set up with `scripts/pod/vllm_setup.sh`, the server started with `scripts/pod/vllm_serve.sh`, this generates the main arm.

```bash
bash experiments/lane_d_sweep2/jobs/sweep2.sh
```

The other jobs hold the exact commands for each arm and for the replay. The arms were queued around the cards available on the night, so their order is not important, only their seeds and flags.

| Jobs | What they do |
| --- | --- |
| `arms.sh`, `arms2.sh`, `arms3.sh`, `queue2.sh`, `queue3.sh`, `queue4.sh`, `multi.sh`, `h200b.sh`, `tip.sh`, `serial.sh`, `resume_after_solvable.sh` | generate the extra arms |
| `chain.sh`, `replay_follow.sh`, `replay_queue.sh`, `tip_replay.sh`, `tip_length.sh` | replay, through `replay_sweep2.py` |
| `score_now.sh`, `final_score.sh` | decompose and score |

| Script | Writes |
| --- | --- |
| `sweep2_behaviour.py`, `solvable_behaviour.py`, `methodtip_behaviour.py`, `solvable_none_twin.py` | behaviour tables, from transcripts alone |
| `sweep2_scores.py`, `arms_scores.py` | detection scores for each arm |
| `sweep2_window_count.py` | `results/sweep2_window_count.json`, the count that feeds the paper |
| `leak2_uptake.py` | `results/leak2_uptake.json`, which note was used |
| `concealment_control.py`, `rule_contrast.py` | the concealment lead and the rule contrast |
| `methodtip_plateau.py`, `methodtip_found.py`, `methodtip_length.py` | the method tip's plateau and its length control |
| `trace_csv.py`, `solvable_trace_csv.py` | per-token traces for plots |
| `tables.py` | `results/tables_sweep2.md` |
