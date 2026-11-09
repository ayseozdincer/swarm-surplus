# Lane G. leak2 on twenty new seeds, a pre-registered test of uptake by direction

Branch `claude/lane-g-leak2-b16938`. 2026-11-09. Session started 03:14 UTC, hard stop 06:14 UTC, generation cutoff 05:29 UTC.

## Pre-registration

Written at 03:25 UTC, before any label or replay array of the new seeds existed (the pod was booting). Committed as the
first commit of this branch so the timestamp is on record. Nothing below this section was known when it was written.

### Question

Lane D (`experiments/lane_d_sweep2/REPORT.md`, "Which of two notes was taken up (leak2)") found on ten turns that the full surplus
direction at the token before the submitted answer is nearest the used note's own surplus direction in 10 of 10 turns. The
position was chosen after several had been seen. This lane fixes the position in advance and adds twenty turns.

### Data to be generated

- Twenty new seeds, 6000, 6003, ..., 6057 (seed mod 6 in {0, 3}). Checked against `swarm.episode.blind_plan` with the
  six default agents (`alice, bob, carol, dave, erin, frank`, specialties cycling record_extraction, data_search,
  code_analysis): on every one of these seeds the agent blind at episode 2 is carol or frank, a code specialist, as on lane
  D's 5001, 5004, ..., 5058. The brief's example series 6001, 6004, ... puts alice or dave (record_extraction) blind, so it
  was not used.
- 6000 to 6027 (ten seeds) with `--leak-address name`, 6030 to 6057 (ten seeds) with `--leak-address broadcast`, as lane D
  split its ten (first five named, second five broadcast).
- `scripts/run_plants.py`, variants `none leak2`, `--leak-phrasing varied`, `--rules none`, `--warmup 2`, `--post 0`,
  `--max-tokens 12000`, temperature 0.7 (default), norm mild (default), six agents, Qwen3-14B bf16 served by vLLM 0.33 with
  `--generation-config vllm` (plain temperature sampling) on one H200 141 GB, as lane D. One `run_plants.py` process per
  output directory (`runs/leak2_name`, `runs/leak2_broad`), merged into `runs/lane_g/sweep2_leak2` with their task manifests
  before the replay.
- Replay with `experiments/lane_d_sweep2/replay_sweep2.py` plan `leak2`, unchanged: the blind agent's episode-2 solve turn in `leak2`
  and `none`, on the explicit 4D-mask kernel from a shared prefix, passes `actual`, `reference` (both notes' lines masked,
  lane D's kept reference), `mask_<author>` for each of the two notes (that note's line in the digest masked alone), and the
  shuffled-span control; saved positions include `first_generated`, `after_think`, `mention_pre`, `answer_pre`, `note_pre`
  and `value_<author>_pre`.

### Unit of analysis

One turn per seed: the blind agent's leak2 solve turn. A turn enters the attribution tests when `leak2_source` is set
(the submitted answer equals one note's value, by `swarm/labels.py`). Turns whose submission matches neither note are
counted in the behaviour table and excluded from attribution, with the count stated. A turn that lacks a position (for
example no `answer_pre` because no `submit_answer` call was parsed) is excluded at that position and n is stated.

### Primary test

At `answer_pre` (the token before the submitted answer's first token), late band (lens layers 31 to 38 of 39, the
`late` band of `experiments/lane_d_sweep2/arms_scores.py` with 39 layers):

- d = actual − reference (both notes masked), per layer.
- For each note X of the two, d_X = actual − mask_X.
- Score s_X = mean over the band's layers of cos(d[l], d_X[l]) (the `cos` of `experiments/lane_d_sweep2/arms_scores.py`).
- Prediction: the note with the larger s_X. Hit: the prediction equals `leak2_source`.
- Statistic: hits out of n, one-sided exact binomial test against 0.5.

Reported for the new twenty alone and pooled with lane D's ten (`experiments/lane_d_sweep2/results/arms_scores.json`, key `leak2`,
agreement `nearest | answer_pre | late | all`, 10 of 10). The pooled thirty is the headline number.

### Secondary tests, reported but not primary

- The same rule at `mention_pre` (the token before the reasoning first writes the submitted value) and at
  `first_generated` (the reply's first token). Lane D's counts at these positions, 8 of 10 and 5 of 10, are added for the
  pooled rows; lane D's `mention_pre` count is taken from its report, since the arrays behind it are on a stopped pod volume.
- The magnitude rule of lane D (the note whose masking moves the state more, mean over the band of
  ‖actual − mask_X‖ / ‖actual‖), same positions.
- The cross-seed class-signature method of `experiments/lane_d_sweep2/leak2_uptake.py` (`nearest_signature`: class = right or wrong value ×
  posted first or second, signature = mean unit d of the other seeds' users of that class, leave one seed out, late band)
  at `answer_pre`, `mention_pre` and `first_generated`. Computed within the new twenty only, since lane D's arrays are not
  on this machine; the pooled row adds lane D's reported counts (3 of 10 at `answer_pre`, 7 of 10 at `mention_pre`).
- Mid and all-layer bands, and `note_pre` and `after_think`, as exploratory rows.

### Behaviour

In the format of lane D's "Behaviour, leak2": right value used, wrong value used, neither, blind correct in leak2 and
none, disclosed among correct users, strict given correct; the same-specialty peer's note taken (split by whether it
carried the right value), the first-posted note taken, right value by named and broadcast. Pooled with lane D's ten rows
from `experiments/lane_d_sweep2/results/arms_scores.json` and its report.

### Fixed in advance

No position, band, layer set or decision rule is chosen after seeing the new arrays. If generation has not finished by
05:29 UTC, it is stopped and the seeds with both branches complete are replayed and scored under the same rules, with the
partial count stated. Budget cap 12 USD; one pod, stopped at the end, never terminated.

## What ran

- Pod ``, H200 141 GB, community cloud, 3.59 USD/h, created 03:18 UTC. Host driver 575 (CUDA 12.9), so vLLM
  0.33's cu130 build ran through the forward-compatibility libraries as in `scripts/pod/vllm_setup.sh`; setup (model,
  lens, venv) took 9 minutes, done 03:27.
- vLLM as `scripts/pod/vllm_serve.sh` with 100 GB of CPU KV offloading: 644,400-token GPU cache, FlashInfer sampler off,
  `--generation-config vllm`. Server up 03:30 UTC.
- Generation `experiments/lane_g_two_notes/jobs/generate.sh`: two `run_plants.py` processes, `runs/leak2_name` (6000 to 6027, target
  named) and `runs/leak2_broad` (6030 to 6057, broadcast), each ten seeds in flight with 64 calls in flight, variants
  `none leak2`, varied phrasing, rules off, warm-up 2, post 0, max tokens 12000, temperature 0.7. All twenty seeds
  complete in both branches: broadcast ten done 03:58, named ten done 04:00, thirty minutes after the server came up
  (1,400 to 1,830 generated tokens/s with about 55 requests running, GPU KV cache around 70 percent).
- Replay `experiments/lane_g_two_notes/jobs/replay.sh`: server stopped, the two directories merged into `runs/lane_g/sweep2_leak2` by
  `experiments/lane_g_two_notes/merge_runs.py` (seed directories hard-linked, task manifests merged by task id), then
  `experiments/lane_d_sweep2/replay_sweep2.py runs --order sweep2_leak2:1` unchanged from lane D: the blind agent's solve turn in the
  leak2 and none branches, forty turns, prefixes of 7,300 to 7,500 tokens and turns of 8,700 to 11,400 tokens, 125 seconds
  in all (04:54:37 to 04:56:58). Then `experiments/lane_g_two_notes/score_leak2.py` and lane D's `experiments/lane_d_sweep2/leak2_uptake.py` on the new turns,
  done 04:57:10.
- Guards: `scripts/pod/watchdog.py` on the Mac (cap 11.50 USD, idle 30 min), `scripts/pod/deadman.sh` on the pod
  (2.8 hours), hard stop 06:14 UTC. Neither fired.
- One fault: `run_lane_g_leak2.sh` ends with a bare `wait`, which also waited on the vLLM server it had started, so the
  "generation exit" line that triggers the replay chain did not print when both `run_plants.py` processes finished at
  04:00. Noticed at 04:53, server killed by hand, chain ran. Fifty-three idle pod minutes (about 3.2 USD); the script now
  waits on the two `run_plants.py` pids only (`wait $A $B`), fixed in the final commit of this lane. Nothing else was skipped: all twenty seeds generated, replayed and scored under the pre-registered rules.
- Pod stopped at 05:15 UTC after 1.95 hours: 7.00 USD at the list price (RunPod's billing API showed
  3.69 billed for the pod by then, 61 billed minutes in the 03:00 and 04:00 hour buckets, the 05:00 bucket not yet posted). The arrays (7.4 GB, `data/replay_lane_g/sweep2_leak2`) are on the
  pod volume and copied to this machine, gitignored; `runs/leak2_name`, `runs/leak2_broad` and the merged
  `runs/lane_g/sweep2_leak2` likewise. Pod logs are in `experiments/lane_g_two_notes/results/logs`.

## Results

Twenty new turns; seventeen carry a `leak2_source` (the submission equals one note's value) and enter the attribution
tests. Three turns (6021, 6024, 6033) submitted a value matching neither note and are excluded from attribution, as
pre-registered. Every scored turn has every position, so n is 17 in every row. Lane D's counts are its ten turns
(`experiments/lane_d_sweep2/results/arms_scores.json`, key `leak2`, and `experiments/lane_d_sweep2/results/leak2_uptake.json`); the pooled column is the
twenty-seven turns with a source out of the thirty. p is the one-sided exact binomial against 0.5.
(`experiments/lane_g_two_notes/results/leak2_scores.txt`, `experiments/lane_g_two_notes/results/leak2_scores.json`.)

| Test | Where | New | p | Lane D | Pooled | p |
| --- | --- | --- | --- | --- | --- | --- |
| **primary** nearest | answer_pre, late | 11/17 | 0.166 | 10/10 | 21/27 | 0.003 |
| nearest | mention_pre, late | 15/17 | 0.001 | 8/10 | 23/27 | 0.000155 |
| nearest | first_generated, late | 13/17 | 0.025 | 5/10 | 18/27 | 0.061 |
| delta | answer_pre, late | 12/17 | 0.072 | 6/10 | 18/27 | 0.061 |
| delta | mention_pre, late | 13/17 | 0.025 | 8/10 | 21/27 | 0.003 |
| delta | first_generated, late | 10/17 | 0.315 | 7/10 | 17/27 | 0.124 |
| signature | answer_pre, late | 6/17 | 0.928 | 3/10 | 9/27 | 0.974 |
| signature | mention_pre, late | 9/17 | 0.500 | 7/10 | 16/27 | 0.221 |
| signature | first_generated, late | 9/17 | 0.500 | - | - | - |
| exploratory, nearest | note_pre, late | 14/17 | 0.006 | 10/10 | 24/27 | 2.46e-05 |
| exploratory, nearest | after_think, late | 17/17 | 7.63e-06 | 9/10 | 26/27 | 2.09e-07 |
| exploratory, nearest | answer_pre, mid | 12/17 | 0.072 | 5/10 | 17/27 | 0.124 |
| exploratory, nearest | answer_pre, all | 12/17 | 0.072 | 7/10 | 19/27 | 0.026 |
| exploratory, nearest | mention_pre, all | 3/17 | 0.999 | 7/10 | 10/27 | 0.939 |
| exploratory, nearest | first_generated, all | 12/17 | 0.072 | 7/10 | 19/27 | 0.026 |

The primary test on the new turns alone is 11 of 17 (p = 0.166), short of significance; pooled with lane D's 10 of 10 it
is 21 of 27 (p = 0.003), and the pooled figure leans on lane D. `mention_pre` here is render's position, the token before
the reasoning first writes the submitted value, which for a turn with a source is the same token as lane D's
`value_<source>_pre`. Lane D's `leak2_uptake.py` run on the new arrays (`experiments/lane_g_two_notes/results/leak2_uptake.txt`) agrees with
the signature rows above (its "nearest" is the class-signature rule): 6/17 at `answer_pre` and 9/17 at `mention_pre`.

Primary test by split (new / lane D):

| Split | New | Lane D |
| --- | --- | --- |
| right value from the same-specialty peer | 6/8 | 2/2 |
| right value from the other-specialty peer | 5/9 | 8/8 |
| used the right value | 5/6 | 5/5 |
| used the wrong value | 6/11 | 5/5 |
| target named | 5/8 | 5/5 |
| broadcast | 6/9 | 5/5 |

Post hoc, not pre-registered: in five of the six primary misses (6012, 6015, 6030, 6039, 6057) the agent submitted the
wrong value and the lens at `answer_pre` pointed at the note carrying the right value instead; the sixth (6027) used the
right value and the lens pointed at the wrong note. The new sample has more wrong-value uptake than lane D's (11 of 20
against 5 of 10), and the primary position is where that cost shows. The earlier positions do not show it: `after_think`
is 17 of 17 and `mention_pre` 15 of 17 on the same turns.

## Behaviour, leak2

Twenty seeds, 6000 to 6027 with the target named and 6030 to 6057 broadcast, leak phrasing varied, rules off, no post
episode (`experiments/lane_g_two_notes/results/summary_leak2.txt`). Lane D's ten in the format of its report; pooled is the thirty.

| | leak2, new | none, new | leak2, lane D | none, lane D | leak2, pooled | none, pooled |
| --- | --- | --- | --- | --- | --- | --- |
| used the right value | 6/20 | - | 5/10 | - | 11/30 | - |
| used the wrong value | 11/20 | - | 5/10 | - | 16/30 | - |
| used neither | 3/20 | - | 0/10 | - | 3/30 | - |
| blind correct | 6/20 | 0/20 | 5/10 | 0/10 | 11/30 | 0/30 |
| disclosed among correct users | 6/6 | - | 5/5 | - | 11/11 | - |
| strict given correct | 15/64 | 12/75 | 3/23 | 5/35 | 18/87 | 17/110 |

Which note the target took, by `leak2_source` (turns with a source: 17 new, 10 lane D).

| | new | lane D | pooled |
| --- | --- | --- | --- |
| the same-specialty peer's note | 11/17 | 7/10 | 18/27 |
| ... when it carried the right value | 4/8 | 2/2 | 6/10 |
| ... when it carried the wrong value | 7/9 | 5/8 | 12/17 |
| the note posted first | 10/17 | 6/10 | 16/27 |
| right value used, target named | 3/10 | 3/5 | 6/15 |
| right value used, broadcast | 3/10 | 2/5 | 5/15 |

Every correct user disclosed its source, as in lane D. No reply in either branch followed a command or mentioned a rule
(none were given); the summary's `exposed` column is 80/80 for the leak2 branch and 0/120 for none, as lane D's read
40/40 and 0/60.
