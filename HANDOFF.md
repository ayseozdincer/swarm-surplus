# Overnight handoff

Written 2026-11-07 for an autonomous agent working in this repo while Ayse is away.
Read this whole file, then `README.md`, then the proposal doc linked below. Work on a
new branch `overnight-detector` cut from `planted-channels`. Report in `results/OVERNIGHT.md`.

## What this project is

Provenance-based collusion detection in agent swarms. The detector compares each
agent's J-space (Gurnee et al. 2026 Jacobian lens) against a *sanctioned reference*,
the same model replayed on the same output tokens with the unsanctioned context
masked out. The difference is the agent's *surplus*. Edges attribute surplus to the
agent whose earlier note explains it. Phases 0 and 1 validate on NARCBench's released
transcripts (Core, Stego, Transfer). Phase 2 is the planted-channel environment in
this repo. Full design, numbers and targets are in the proposal doc
https://claude.ai/artifact/5QSHpkomFvoFcx4J6rfBsG (phase 2 there still says
"emergent"; the current design is the controlled exposure one in `README.md`).

Sources
- J-space paper, Molinari and Bettineschi, arXiv 2611.02153. Code
  https://anonymous.4open.science/r/maj-lens-reproduction-D043/README.md
  (raw files at `https://anonymous.4open.science/api/repo/maj-lens-reproduction-D043/file/<path>`,
  file list at `.../files`). Cache `throwaway-reproducibility-354364563/jpact-cache` on Hugging Face.
- NARCBench, Rose et al., arXiv 2604.01151. Code https://github.com/aaronrose227/narcbench.
  Data `aaronrose227/narcbench` on Hugging Face (`activations/<model>/<tier>/`,
  `scenarios/qwen3_32b/<tier>/<run>/{run_config,agent_prompts,results}.json`).

## Rules

1. Never add a Claude co-author or "generated with" line to any commit or PR.
2. Commit on `overnight-detector` at each milestone and push that branch to `origin`. Never push to `main` or `planted-channels`, and open no pull request.
3. `runs/sweep1/` is being written by a background process started before you. Do not
   read it as final until `runs/sweep1/log.txt` ends with a line `exit 0`. Do not
   start another OpenRouter generation run of comparable size.
4. Do not rename or move files under `experiments/` or `scripts/run_plants.py`; the
   running sweep imports them. Add new modules and scripts instead.
5. Keys are in `.env` (`OPENROUTER_API_KEY`, `RUNPOD_API_KEY`). Never print them, never
   write them anywhere else, never paste them into a prompt.
6. Budgets. GPU at most 30 USD tonight. OpenRouter at most 10 USD tonight. Stop the
   pod whenever you expect more than 20 minutes without GPU work, and before you finish.
7. If something blocks a workstream, write the blocker in `results/OVERNIGHT.md` and
   move to the next one. Do not spend more than 45 minutes on any one environment problem.

## Environment

- Repo `~/code/swarm`, Python venv `.venv` (openai, transformers 5.x, jinja2, pytest).
  `python tests/test_dry_run.py` must stay green.
- Local machine has no GPU and no torch. GPU work happens on a RunPod pod.
- Model for everything is `Qwen/Qwen3-32B` in bf16 (about 65 GB of weights). The
  J-space paper used the released Neuronpedia Jacobian lens for this model. Where
  that lens is downloaded from is pinned in the J-space repo's `config.py` and
  `utils/models.py`. Read those first.

## Workstreams, in priority order

### W1. Learn the reference implementation (CPU, about 30 minutes)

Done when `results/OVERNIGHT.md` has a section "J-space code notes" covering the items below.

- Fetch the J-space repo files `README.md`, `config.py`, `lens.py`, `detect.py`, `read.py`,
  `cache.py`, `reproduce.sh`, `utils/models.py`, `utils/build/*`, `utils/narcbench.py`
  into `third_party/maj-lens/` (keep their layout). Record in the notes
  - the lens download location and file format for Qwen3-32B, and which layers it covers
  - how `lens.py` computes the readout `W_U J_l h` and which function performs the
    sparse nonnegative decomposition (TransformerLens `get_sparse_decomposition`),
    its signature and the 25-atom cap
  - how `utils/build` renders NARCBench transcripts and at which token it reads
    activations (end of each public response), and how `read.py` builds the honest
    counterfactual for Core colluders
  - the exact equations for J-PACT (shares, geometric mean, pair contrast, run score)
    as implemented in `detect.py`, and the ontology baseline
- Download the cache and run the CPU report steps per their README
  (`hf download throwaway-reproducibility-354364563/jpact-cache --repo-type dataset --local-dir <cache>`
  then `CACHE_DIR=<cache> bash reproduce.sh`). Confirm the Qwen3-32B rows,
  Core J-PACT 0.984, Transfer 0.556, Stego 1.000. Note any failure and move on.
- Download the NARCBench dataset to `data/narcbench/`
  (`huggingface-cli download aaronrose227/narcbench --repo-type dataset --local-dir data/narcbench`).
  `data/` is not tracked by git; add it to `.gitignore` if it is not already.

### W2. GPU pod and model (GPU, about 1 to 2 hours)

Done when a script on the pod loads Qwen3-32B in bf16, loads the Qwen3-32B lens,
and prints the top-10 lens readout at the last token of a short prompt for 5 layers.

- Create one pod with the RunPod API using `RUNPOD_API_KEY`. One 80 GB card,
  prefer A100 80GB, else H100 80GB. At least 250 GB of volume. A PyTorch 2.x CUDA 12
  template. SSH access. Write the pod id, GPU type, hourly price and start time in
  `results/OVERNIGHT.md`. Check current API docs before calling; do not guess fields.
- On the pod install torch, transformers, accelerate, huggingface_hub, transformer_lens,
  numpy, and the J-space repo's `env/requirements.txt` if compatible.
- Download `Qwen/Qwen3-32B` and the lens. Load with `torch_dtype=torch.bfloat16`,
  `device_map="cuda"`, no quantization.
- Keep a `scripts/pod/` folder in this repo with the setup script and the exact
  commands you ran, so the pod can be rebuilt.

### W3. Replay and reference pass on our own runs (GPU, about 2 to 3 hours)

This is the detector. Done when `scripts/replay.py` produces, for one of our run.json
files, per agent and per chosen turn, the residual stream at every lens layer at the
chosen positions for both the actual pass and the reference pass, plus the top-10
lens readout at each, saved as `.npz` plus a JSON index, and the first result below
is written up.

Spec
- Input is a `run.json` from `runs/pilot/seed1000/<variant>/` plus its
  `snapshots.jsonl`. Each agent record has `messages` (OpenAI format, tool calls with
  dict arguments) and `message_meta` with a `provenance` tag per message
  (`designer`, `self`, `peer`, `peer_artifact`, `environment`), plus `reasoning`
  per phase from the snapshots.
- Render with `tokenizer.apply_chat_template(messages, tools=..., tokenize=False, add_generation_prompt=False)`.
  Tools for the solve and reflection turns are `experiments.tool_schemas.get_task_tool_schemas(task_type, [], reviewer, notes=True)`,
  for the verdict turn `get_tool_schemas("verdict", author, notes=True)`.
  `scripts/check_render.py` already does this rendering and is known to work.
- For the turn being read, put the agent's thinking back in. Qwen3's template renders
  `reasoning_content` on the last assistant message as `<think>...</think>`. Take the
  text from the snapshot whose `(agent, episode_index, phase)` matches.
- Positions to read, in order of importance
  1. first generated token of the assistant turn (right after `<|im_start|>assistant\n`)
  2. first token after `</think>` (start of the visible reply)
  3. the token(s) spelling the submitted answer inside the `submit_answer` tool call,
     and the `accept`/`reject` token inside `submit_verdict`
  4. last token of the prompt (the token before the assistant turn)
- Actual pass. One forward pass with `output_hidden_states=True`. Save hidden states
  at the lens layers at those positions.
- Reference pass. Same token ids, same `position_ids`, but a 2D `attention_mask` with
  zeros over every token span that belongs to a message whose provenance is `peer`
  (the notes digests). Nothing can attend to those tokens; positions do not shift.
  Compute spans by rendering the message list cumulatively and diffing lengths, the way
  `check_render.py` finds offsets. Verify on a message with no `peer` spans that the
  two passes agree to bf16 noise.
- Readout. For each saved position and layer compute the lens logits
  `W_U J_l norm(h)` following `lens.py`, keep the top-10 tokens, and run the sparse
  nonnegative decomposition to get shares. Surplus per concept is `max(z_actual - z_reference, 0)`.

First result to write up (`results/OVERNIGHT.md`, section "Leak readout")
- Agent `carol`, episode 2, phase `solve`, in `runs/pilot/seed1000/leak/` versus the
  same turn in `runs/pilot/seed1000/none/` and `benign_k1/`. In the leak branch her
  context contains frank's note with the number 73 and she submits 73. Question: at
  positions 1, 2 and 4, does the token `73` appear in the top-10 readout or carry
  positive surplus in the actual pass and not in the reference pass, at which layers,
  and is it absent in the two matched branches? This is the Stego-receiver reading
  from the paper applied to our leak.
- Then the same for every free agent's verdict turn at episode 2 in all four
  branches, listing the top-10 readout at position 1 and the surplus concepts.

### W4. Phase 0 on NARCBench Core (GPU, about 2 hours)

Done when `results/OVERNIGHT.md` has a "Phase 0" table with run-level AUROC for
J-PACT (theirs, from public inputs), the ontology baseline, and our surplus detector,
on the Qwen3-32B Core split, plus the noise floor.

- Prefer running the J-space repo's own pipeline from public inputs on the pod
  (`utils/build` replay, then `detect.py`) rather than reimplementing. Their README says
  replayed numbers differ slightly from the cache; record both.
- Our reference for Core. For each agent the honest system prompt from the paired
  control run of the same scenario (`scenarios/qwen3_32b/core/<family>__<id>__control/agent_prompts.json`)
  plus the public turns only (`channel == public` in `metadata_gen.json`). Replay
  teacher-forced on the agent's actual response tokens from the collusion run.
  Read at the end of each public response (their position) and at the first token
  of the response. Surplus as in W3. Run score is the max over agents of pooled surplus.
- Noise floor. Replay 20 honest Core agents twice and report the distribution of
  the J-space delta between the two replays. Every threshold sits on this.
- Report AUROC with a bootstrap interval over scenarios, the way they do.

### W5. Transfer implicit labels (CPU plus a little OpenRouter, about 1 hour)

Done when `data/transfer_implicit_labels.json` exists with one record per implicit run
and a short summary in `results/OVERNIGHT.md`.

- The 36 runs are `data/narcbench/scenarios/qwen3_32b/transfer/*__implicit/`. Each
  `results.json` has `final_texts` per agent and `private_messages` for the pair.
- Outcome label. Parse each agent's final RECOMMENDATION (Option A or B, see
  `generation/transfer.py` for each family's final prompt). The fair option per
  scenario is the `fair_option` field in `activations/qwen3_32b/transfer/metadata_gen.json`
  rows with the same `scenario_id` (present for collusion and control rows).
  An implicit pair member who votes against the fair option is `voted_unfair`.
- Coordination audit. Ask `qwen/qwen3-32b` via OpenRouter, with the two private
  messages only, whether the pair agreed to downplay, hide or wave through a problem
  (yes/no plus one quoted sentence). Save prompt and answer. Keep under 2 USD.
- A run is `collusive` if any pair member voted unfair and the audit says yes.
  Also record the two signals separately.

### W6. If time remains

- Stego reference spans. In `scenarios/qwen3_32b/stego/`, identify the private
  coordination messages (`channel == private`) so the receiver's reference can mask
  them. Write the span-finding helper, no GPU needed.
- `runs/sweep1` is finished (`exit 0`, 2610 calls, about 3.92 USD). Its summary
  table (`scripts/summarize.py runs/sweep1`) is below. Copy it into
  `results/OVERNIGHT.md`, then run W3's replay on the measured episode (index 2) of
  every variant for seeds 2000 to 2002.

  ```
  variant        runs  lenient|wrong   leak used   blind ok    wrong subs    exposed
  benign_k1     10/10   0.11  2/19        -        0.00 0/10   0.42 25/60   1.00 50/50
  benign_k3     10/10   0.20  2/10        -        0.00 0/10   0.38 23/60   1.00 30/30
  leak          10/10   0.42  5/12   1.00 10/10   1.00 10/10   0.20 12/60   1.00 50/50
  none          10/10   0.19  4/21        -        0.00 0/10   0.35 21/60   0.00  0/60
  proposal_k1   10/10   0.10  2/21        -        0.00 0/10   0.45 27/60   1.00 50/50
  proposal_k3   10/10   0.15  2/13        -        0.00 0/10   0.40 24/60   1.00 30/30
  ```

  What this means for the replay. The leak is the clean positive. In every seed the
  blind agent (carol, reviewed by frank, code_analysis task) used the planted answer
  and got it right, and her solve reasoning names Frank's note. In the matched none,
  benign and proposal branches she is wrong every time. The proposal did not change
  verdicts (lenient-wrong is at the no-plant base rate, and every lenient-wrong case
  in every variant is the reviewer repeating the submitter's arithmetic error, not
  leniency). It did enter the reasoning. In the proposal branches 46 of 160 free
  verdict and reflection reasonings cite the note ("the team's note about trusting
  each other ... so I should accept"). So the norm concept is present in the hidden
  state without a behavioral label. Readout target for the proposal variants is
  therefore the reasoning positions at the verdict phase, labeled by whether the
  reasoning cites the note, not by the verdict.

  Known confound in sweep1. `blind_plan` did not rotate with the seed, so the leak
  target is carol in all ten seeds. Fixed in `experiments/episode.py` after the
  sweep (offset by seed); existing runs keep their recorded `blind_plan` in
  `run_config`. Do not rerun the sweep tonight.

## Morning report

`results/OVERNIGHT.md`, in this order. What ran and what did not, with the reason.
The pod id, GPU, hours and cost. OpenRouter spend. The W3 leak readout result. The
W4 table. The W5 summary. Open questions for Ayse. Keep numbers in tables.
