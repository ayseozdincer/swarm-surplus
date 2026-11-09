# Swarm Surplus

Reading what an agent took from its peers by replaying it without them.

Ayse Ozdincer and Máté Lakner. Apart Research Swarm Alignment Sprint, November 2026. The paper is [paper/paper.md](paper/paper.md), with the submitted Word version beside it.

## The idea

An orchestrator already knows who wrote every span of an agent's context. We use that provenance in place of collusion labels. For one agent at one turn, the model runs twice over the same tokens, once as it was served and once with every peer span masked from attention. The reply is forced in both passes, so the first reproduces the states the model had while generating, and the second shows what it would have held without its peers.

```
actual pass      designer  self  [peer note]  self  ...  reply tokens  -> h_actual
reference pass   designer  self  [ masked  ]  self  ...  same tokens   -> h_reference

surplus = h_actual - h_reference, read token by token along the reply
```

The surplus gives three reads.

- **Where.** The count of reply tokens with a large surplus says the reasoning engaged with peer content.
- **Taken or computed.** The surplus at the first writing of a value is large for a peer's value, which the reference cannot predict, and small for a value the agent worked out, which the reference reconstructs from the agent's own reasoning.
- **Whose.** With one reference per candidate note, the direction of the surplus names the note that was used.

Nothing here needs a collusion label, a reading of the channel, or an assumption about roles or coalition size.

## Main results

| Claim | Result | Source |
| --- | --- | --- |
| A rule fixed on ten pilot seeds, on held-out turns | flags 98 of 108 taken values and 0 of 473 negatives | [Table 2](paper/tables/heldout_rule.md) |
| The same rule on twenty seeds generated after everything was fixed | flags 17 of 17 agents that took a note's value and 0 of 20 controls | [Table D7](paper/tables/newdata_test.md) |
| A taken value against a computed one | first written 33 to 61 floors above noise against 11 | [Table D2](paper/tables/jump_per_turn.md) |
| Which of two conflicting notes was used, pre-registered | 21 of 27 turns | [lane G](experiments/lane_g_two_notes/REPORT.md) |
| A line in a shared file, declared against write-log provenance | source direction AUROC 0.48 against 1.00 on the same turns | [Table 3](experiments/lane_e_shared_workspace/REPORT_2c.md) |
| External check, 600 injected-document turns | fires on 198 of 199 followers and no clean twin | [Table D8](experiments/external_tasktracker/REPORT.md) |
| External check, the insider-trading eval | sees the tip in every turn, cannot tell a trader from a decliner (AUROC 0.52) | [Table D9](experiments/external_insider_trading/REPORT.md) |
| NARCBench, the released benchmarks | run through the J-space pipeline and reported, not claimed, because a size-matched honest edit removes the result | [phase 0](experiments/phase0_controls/REPORT.md), [lane B](experiments/lane_b_transfer_stego/REPORT.md) |

## What is in the repository

```
paper/          the paper, its tables and figure, and the scripts that build them
swarm/          the agent environment, six agents in a peer-review game with provenance tags
detector/       the method, rendering a turn exactly as served, masks, J-space readouts
scripts/        shared entry points, run_plants.py generates, replay.py replays, decompose.py adds J-space shares
scripts/pod/    GPU setup, pinned versions and model downloads
experiments/    one folder per experiment, each with its code, exact commands, report and results
tests/          offline tests with a mock model, and GPU checks of the replay
third_party/    the J-space code (maj-lens) and the insider-trading eval, used as released
```

Each experiment folder has a README that says what it asks, where it appears in the paper and how to rerun it, and a REPORT with the full account. Start with [experiments/README.md](experiments/README.md).

## Reproducing

Full instructions are in [REPRODUCING.md](REPRODUCING.md). There are three levels.

| Level | What it does | Needs | Time |
| --- | --- | --- | --- |
| 1 | runs the environment on a mock model, runs the tests, rebuilds the tables that come from committed scores | Python | a minute or two |
| 2 | rebuilds every table and recomputes per-experiment scores from the transcripts and traces | the data bundle, about 1 GB | minutes |
| 3 | regenerates transcripts and replays them | GPU, API keys, about 110 USD for all of it | about 40 card hours |

```bash
make setup
source .venv/bin/activate
make test
make mock
make check
```

`make check` rebuilds the tables and fails if any of them differs from the committed version.

## Work by others

The J-space code in `third_party/maj-lens` (Molinari and Bettineschi, 2026), the NARCBench transcripts and activations (Rose et al., 2026), the insider-trading eval in `third_party/insider-trading` (Scheurer et al., 2023, CC-BY-4.0) and the Jacobian lenses released on Neuronpedia are used as released. Everything else was written during the sprint, and the git history records it. Each merged experiment keeps its commits, including the pre-registration in lane G.
