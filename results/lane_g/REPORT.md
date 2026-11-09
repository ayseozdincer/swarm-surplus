# Lane G. leak2 on twenty new seeds, a pre-registered test of uptake by direction

Branch `claude/lane-g-leak2-b16938`. 2026-11-09. Session started 03:14 UTC, hard stop 06:14 UTC, generation cutoff 05:29 UTC.

## Pre-registration

Written at 03:20 UTC (commit 595d9a1), before any label or replay array of the new seeds existed (the pod was booting). Committed as the
first commit of this branch so the timestamp is on record. Nothing below this section was known when it was written.

### Question

Lane D (`results/lane_d/REPORT.md`, "Which of two notes was taken up (leak2)") found on ten turns that the full surplus
direction at the token before the submitted answer is nearest the used note's own surplus direction in 10 of 10 turns. The
position was chosen after several had been seen. This lane fixes the position in advance and adds twenty turns.

### Data to be generated

- Twenty new seeds, 6000, 6003, ..., 6057 (seed mod 6 in {0, 3}). Checked against `experiments.episode.blind_plan` with the
  six default agents (`alice, bob, carol, dave, erin, frank`, specialties cycling record_extraction, data_search,
  code_analysis): on every one of these seeds the agent blind at episode 2 is carol or frank, a code specialist, as on lane
  D's 5001, 5004, ..., 5058. The brief's example series 6001, 6004, ... puts alice or dave (record_extraction) blind, so it
  was not used.
- 6000 to 6027 (ten seeds) with `--leak-address name`, 6030 to 6057 (ten seeds) with `--leak-address broadcast`, as lane D
  split its ten (first five named, second five broadcast).
- `scripts/run_plants.py`, variants `none leak2`, `--leak-phrasing varied`, `--rules none`, `--warmup 2`, `--post 0`,
  `--max-tokens 12000`, temperature 0.7 (default), norm mild (default), six agents, Qwen3-14B bf16 served by vLLM 0.33 with
  `--generation-config vllm` (plain temperature sampling) on one H200 141 GB, as lane D. One `run_plants.py` process per
  output directory (`runs/leak2_name`, `runs/leak2_broad`), merged into `runs/sweep2_leak2` with their task manifests
  before the replay.
- Replay with `scripts/pod/replay_sweep2.py` plan `leak2`, unchanged: the blind agent's episode-2 solve turn in `leak2`
  and `none`, on the explicit 4D-mask kernel from a shared prefix, passes `actual`, `reference` (both notes' lines masked,
  lane D's kept reference), `mask_<author>` for each of the two notes (that note's line in the digest masked alone), and the
  shuffled-span control; saved positions include `first_generated`, `after_think`, `mention_pre`, `answer_pre`, `note_pre`
  and `value_<author>_pre`.

### Unit of analysis

One turn per seed: the blind agent's leak2 solve turn. A turn enters the attribution tests when `leak2_source` is set
(the submitted answer equals one note's value, by `experiments/labels.py`). Turns whose submission matches neither note are
counted in the behaviour table and excluded from attribution, with the count stated. A turn that lacks a position (for
example no `answer_pre` because no `submit_answer` call was parsed) is excluded at that position and n is stated.

### Primary test

At `answer_pre` (the token before the submitted answer's first token), late band (lens layers 31 to 38 of 39, the
`late` band of `scripts/arms_scores.py` with 39 layers):

- d = actual − reference (both notes masked), per layer.
- For each note X of the two, d_X = actual − mask_X.
- Score s_X = mean over the band's layers of cos(d[l], d_X[l]) (the `cos` of `scripts/arms_scores.py`).
- Prediction: the note with the larger s_X. Hit: the prediction equals `leak2_source`.
- Statistic: hits out of n, one-sided exact binomial test against 0.5.

Reported for the new twenty alone and pooled with lane D's ten (`results/lane_d/arms_scores.json`, key `leak2`,
agreement `nearest | answer_pre | late | all`, 10 of 10). The pooled thirty is the headline number.

### Secondary tests, reported but not primary

- The same rule at `mention_pre` (the token before the reasoning first writes the submitted value) and at
  `first_generated` (the reply's first token). Lane D's counts at these positions, 8 of 10 and 5 of 10, are added for the
  pooled rows; lane D's `mention_pre` count is taken from its report, since the arrays behind it are on a stopped pod volume.
- The magnitude rule of lane D (the note whose masking moves the state more, mean over the band of
  ‖actual − mask_X‖ / ‖actual‖), same positions.
- The cross-seed class-signature method of `scripts/leak2_uptake.py` (`nearest_signature`: class = right or wrong value ×
  posted first or second, signature = mean unit d of the other seeds' users of that class, leave one seed out, late band)
  at `answer_pre`, `mention_pre` and `first_generated`. Computed within the new twenty only, since lane D's arrays are not
  on this machine; the pooled row adds lane D's reported counts (3 of 10 at `answer_pre`, 7 of 10 at `mention_pre`).
- Mid and all-layer bands, and `note_pre` and `after_think`, as exploratory rows.

### Behaviour

In the format of lane D's "Behaviour, leak2": right value used, wrong value used, neither, blind correct in leak2 and
none, disclosed among correct users, strict given correct; the same-specialty peer's note taken (split by whether it
carried the right value), the first-posted note taken, right value by named and broadcast. Pooled with lane D's ten rows
from `results/lane_d/arms_scores.json` and its report.

### Fixed in advance

No position, band, layer set or decision rule is chosen after seeing the new arrays. If generation has not finished by
05:29 UTC, it is stopped and the seeds with both branches complete are replayed and scored under the same rules, with the
partial count stated. Budget cap 12 USD; one pod, stopped at the end, never terminated.

## What ran

(filled in after the run)
