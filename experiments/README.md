# Experiments

One folder per experiment. The names match the paper, whose Appendix F calls the parallel workstreams lanes.

| Folder | What it asks | In the paper | Former branch |
| --- | --- | --- | --- |
| [phase0_controls](phase0_controls/) | does J-PACT reproduce from public inputs, what does the surplus measure on NARCBench Core, is the leak read at the positions prior work reads (Qwen3-32B) | Appendix B and C | `overnight-detector` |
| [lane_b_transfer_stego](lane_b_transfer_stego/) | the surplus on NARCBench Transfer and Stego (Qwen3-32B) | Appendix B | `claude/lane-b-phase1-16e264` |
| [lane_c_sweep14b](lane_c_sweep14b/) | the leak on Qwen3-14B with a converged lens, the calibration set for the rule | Section 4, Tables D1 and D2 | `claude/lane-c-14b-8c893e` |
| [lane_d_sweep2](lane_d_sweep2/) | the main game, under a confidentiality rule, with every extra arm | Section 4, Table 2 | `claude/lane-d-sweep2-affc5f` |
| [lane_e_shared_workspace](lane_e_shared_workspace/) | does coordination emerge in a shared workspace (2b), does write-log provenance recover a channel declared to nobody (2c) | Section 4, Table 3 | `claude/lane-e-2b-713994` |
| [lane_f_false_positives](lane_f_false_positives/) | false positives by task type, the compacted reference, extra positive turns | Section 4, Table D3 | `claude/lane-f-fpr-96ebe2` |
| [lane_g_two_notes](lane_g_two_notes/) | a pre-registered test of which of two notes was used, on twenty new seeds | Section 4, Table D7 | `claude/lane-g-leak2-b16938` |
| [external_tasktracker](external_tasktracker/) | the frozen rule on 600 injected-document turns | Table D8 | `claude/clever-mayer-c077dc` |
| [external_insider_trading](external_insider_trading/) | the frozen rule on the insider-trading pressure eval | Table D9 | `claude/charming-wilson-08bb71` |
| [labelling](labelling/) | thirty traces read in full, to check the labels | Appendix E | |

## What is in a folder

| Path | What it holds |
| --- | --- |
| `README.md` | what the experiment asks, where it appears in the paper, how to rerun it |
| `REPORT.md` | the full account written when it ran, with every table, what did not work and why |
| `*.py` | the experiment's replay and scoring scripts |
| `jobs/` | the exact shell commands as they ran on the GPU, with seeds and flags |
| `results/` | what those scripts wrote, per-turn scores, tables and traces |

Reports were written as the work went, so they use the working names of the time, branch names included. The numbers in them are the record. Where a report and the paper differ, the paper is later and defines one measure across all experiments.

## Words the reports use

| Word | Meaning |
| --- | --- |
| lane | one of the parallel workstreams, B to G |
| W1 to W7 | the workstreams of the first GPU run, in `phase0_controls` |
| arm | one experimental design, such as `leak2` or `method_tip` |
| branch | one variant of a seed, such as `leak` or `none`, from the same warm-up state |
| sweep1, sweep14b, sweep2 | the sets of runs on Qwen3-32B, Qwen3-14B through OpenRouter and self-served Qwen3-14B |
| blind agent | the agent whose copy of the task has the needed piece withheld |
| none twin | the same seed and agent with nothing planted |
| floor | the plain-kernel against explicit-kernel difference on turns with nothing to mask |
| gate, confirmation | count of the first 256 reply tokens above 10 floors at or above 60, surplus at the agent's own first writing at or above 20 floors |
| kept, compacted | references with position ids left as they are, or closed up after each masked span |
| 32B, 14B | Qwen3-32B and Qwen3-14B |
