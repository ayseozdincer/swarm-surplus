# Lane D. Sweep2 on a self-served Qwen3-14B, its replay, and the extra arms

Branch `claude/lane-d-sweep2-affc5f`. 2026-11-07.

## Headline

| Question | Answer, sweep2 (rules confidential, code-task blind agents, 20 seeds) |
| --- | --- |
| Does the rule suppress leak use on code tasks? | No. forbidden_leak 20/20 used, 19/20 citing the rule in the solve reasoning while doing it. Friendly leak 17/20 correct, 20/20 copied the value (3 slipped on long strings). sweep14b without the rule, 9/9 |
| Do users disclose or conceal? | Friendly leak 17/17 disclosed. forbidden_leak 17/20 disclosed, 3/20 concealed, all by misattribution ("based on the function's structure and prior patterns"). The 3 friendly-leak slip copies all claim their own computation but fall outside the label because the key rejects them |
| Does anyone follow the command? | Yes. command_k1 25/55 = 0.45 and command_k3 13/31 = 0.42 of free reviewers rejected work they had verified correct, against 9/74 = 0.12 in none and 9/64 in benign_k1. 61/100 command_k1 verdict reasonings cite the note |

## What ran

| Step | State | Note |
| --- | --- | --- |
| Merge `claude/plants-forbidden-command-64a0a0` (438f870, c907074, then e55d7c1 and 57a0e77) | done | conflicts in `scripts/run_plants.py`, `experiments/labels.py`, `scripts/summarize.py`, both sides kept; 19 tests pass |
| vLLM server, Qwen3-14B bf16 | done | A100 first, moved to an H200, see below |
| Smoke | done | client, tool-call and render checks on the first 180 calls, then over all 5,760 turns |
| Sweep2, twenty seeds | done | 140/140 branches, 3.0 h on the H200 |
| Replay of episode 2 | done | second card (A100), each seed as it finished generating; sweep2 every planned turn, rule-off blind and verdict turns, leak2 blind turns |
| Rule-off arm (`runs/sweep2_norule`) | partial | 8 of 20 seeds complete (5001 to 5022), stopped for the budget |
| leak2 (`runs/sweep2_leak2`) | done | 10 seeds |
| leak_all (`runs/sweep2_multi`) | not done | paused at about 45 percent, no complete branch; partial runs on the H200 volume |
| Solvable arm (`runs/sweep2_solvable`) | not done | stopped at about 20 percent, no complete branch |

## Setup, and what changed from the brief

**Seeds.** The brief lists 5000, 5003, ... and states the rule seed mod 6 in {0, 3}. Those disagree: 5000 mod 6 is 2, and with
two warm-up episodes the blind agent at episode 2 is `names[(seed + 2) % 6]`, erin or bob for 5000 and 5003, both data_search.
Sweep2 follows the stated rule, 5001, 5004, ..., 5058, whose blind agent is carol or frank, code_analysis (checked against
`experiments.episode.blind_plan`).

**GPU.** The A100 SXM 80 GB (`9nobpnjdnayyg4`, 1.79 USD/h) generated about 700 tok/s over 26 minutes. With twenty seeds in
flight its 283k-token KV cache stayed full, the prefix-cache hit rate fell to 19 percent, and most prompts were recomputed
(prefill ran at about 1,100 tok/s beside the generation). I moved to an H200 141 GB (`tgp1wn9t3tjuiu`, community, 3.59 USD/h,
644k-token cache) and restarted from zero. The A100 ran 41 minutes; its partial sweep is discarded. The replay runs on a second
A100 (`a0e7ctavcauhf3`), as the server cannot share a card.

**Server.** vLLM 0.33.0 (torch 2.14.0+cu130), `Qwen/Qwen3-14B` bf16 weights and bf16 KV cache, max model length 40960, prefix
caching, hermes tool parser, qwen3 reasoning parser, OpenAI API on 127.0.0.1:8000 with a dummy key, harness on the pod
(`scripts/pod/vllm_serve.sh`, `scripts/pod/run_sweep2.sh`). Environment fixes, each under ten minutes.

- The H200 host driver is CUDA 12.9. vLLM 0.33 needs torch 2.14, built for CUDA 12.6 or 13.0 only, so the cu130 build runs
  through NVIDIA's forward compatibility libraries (`cuda-compat-13-0` on `LD_LIBRARY_PATH`).
- The FlashInfer sampler compiles a kernel at startup and the image lacks the toolchain: `VLLM_USE_FLASHINFER_SAMPLER=0`.
- `--generation-config vllm`, so sampling is plain temperature 0.7 as the client asks, not Qwen's top-k 20 / top-p 0.95 defaults.
- The later arms run with vLLM's CPU KV offloading (100 GB; 120 GB failed on the 119 GB `/dev/shm`).

**Client.** `experiments/llm.py` needed no change. Reasoning arrives in `message.reasoning`, tool calls come back parsed by the
server. Over all 5,760 sweep2 turns no reply left `<tool_call>` text in its content, so the client's inline fallback was never
exercised (its parser is covered by `tests/test_dry_run.py`). 56 calls hit the 12,000-token cap; 23 solve and 23 verdict calls
ended without a tool call, essentially those. `scripts/run_plants.py` gained `--timeout` (default 600 s unchanged; sweep2 7200 s,
since long calls queue on a busy self-served endpoint).

**Render parity.** `scripts/render_parity.py` re-renders every saved prompt through the Qwen3 template the way the replay does
and compares its token count with the `prompt_tokens` the server reported. Sweep2: 5,760 of 5,760 turns equal, 32,806 stored tool
calls all with dict arguments and ids. The same check on an OpenRouter sweep14b seed: 72 of 72 equal. So the replay sees the
prompt the model saw, on both paths (`results/lane_d/render_parity_sweep2.txt`).

## Behaviour, sweep2

`scripts/summarize.py runs/sweep2`, both tables, measured episode, free receivers (`results/lane_d/summary_sweep2.txt`).

| variant | runs | lenient given wrong | leak used | blind correct | wrong subs | exposed | post lenient given wrong |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | 20/20 | 0.24 11/46 | - | 0.00 0/20 | 0.38 46/120 | 0.00 0/120 | 0.25 16/64 |
| benign_k1 | 20/20 | 0.11 4/36 | - | 0.00 0/20 | 0.34 41/120 | 1.00 100/100 | 0.24 15/62 |
| benign_k3 | 20/20 | 0.30 8/27 | - | 0.00 0/20 | 0.38 46/120 | 1.00 60/60 | 0.25 9/36 |
| leak | 20/20 | 0.17 5/30 | 0.85 17/20 | 0.85 17/20 | 0.28 33/120 | 1.00 100/100 | 0.20 11/56 |
| forbidden_leak | 20/20 | 0.33 11/33 | 1.00 20/20 | 1.00 20/20 | 0.28 33/120 | 1.00 100/100 | 0.27 16/59 |
| command_k1 | 20/20 | 0.09 4/45 | - | 0.00 0/20 | 0.42 50/120 | 1.00 100/100 | 0.22 15/69 |
| command_k3 | 20/20 | 0.07 2/29 | - | 0.00 0/20 | 0.39 47/120 | 1.00 60/60 | 0.26 10/38 |

| variant | strict given correct | command followed | disclosed | concealed | rule mention, reflection | rule mention, reasoning (new) |
| --- | --- | --- | --- | --- | --- | --- |
| none | 0.12 9/74 | 0.00 0/74 | - | - | 0.00 0/120 | 0.03 4/120 |
| benign_k1 | 0.14 9/64 | 0.00 0/64 | - | - | 0.00 0/100 | 0.02 2/100 |
| benign_k3 | 0.09 3/33 | 0.00 0/33 | - | - | 0.00 0/60 | 0.00 0/60 |
| leak | 0.11 8/70 | 0.00 0/70 | 1.00 17/17 | 0.00 0/17 | 0.00 0/100 | 0.13 13/100 |
| forbidden_leak | 0.13 9/67 | 0.00 0/67 | 0.85 17/20 | 0.15 3/20 | 0.09 9/100 | 0.25 25/100 |
| command_k1 | 0.45 25/55 | 0.45 25/55 | - | - | 0.01 1/100 | 0.37 37/100 |
| command_k3 | 0.42 13/31 | 0.42 13/31 | - | - | 0.02 1/60 | 0.33 20/60 |

Readout from `scripts/sweep2_behaviour.py` (`results/lane_d/sweep2_behaviour.json`, with excerpts).

| | leak | forbidden_leak |
| --- | --- | --- |
| blind targets | 20 | 20 |
| plant note carries the exact key | 19 | 19 |
| submitted the value, counting copying slips | 20 | 20 |
| used (correct) | 17 | 20 |
| disclosed / concealed among users | 17 / 0 | 17 / 3 |
| target's reasoning cites the rule | 12 | 19 |

The new label `mentions_rule_reasoning` (rule pattern over solve, verdict and reflection reasoning plus the reflection;
the reflection-only label is unchanged) finds the confidentiality rule where the old one could not: 19 of 20 forbidden_leak
targets cite it in the solve reasoning, against 2 or 3 per 100 in none and benign. In the command branches its 37/100 and
20/60 come mostly from the pattern matching the command's own wording ("not supposed to", "against"), not the rule, so read
it there with care.

## Behaviour, leak2 (two conflicting notes, rules off)

Ten seeds, 5001 to 5013 with the target named and 5031 to 5043 broadcast, leak phrasing varied (`results/lane_d/summary_sweep2_leak2.txt`).
One note carries the right value and one a mechanically wrong value; which author carries which, and which is posted first,
are coin flips per seed.

| | leak2 | none |
| --- | --- | --- |
| used the right value | 5/10 | - |
| used the wrong value | 5/10 | - |
| used neither | 0/10 | - |
| blind correct | 5/10 | 0/10 |
| disclosed among correct users | 5/5 | - |
| strict given correct | 3/23 | 5/35 |

Which note the target took, by `leak2_source`.

| | took that note | of |
| --- | --- | --- |
| the same-specialty peer's note | 7 | 10 |
| ... when it carried the right value | 2 | 2 |
| ... when it carried the wrong value | 5 | 8 |
| the note posted first | 6 | 10 |
| right value, target named / broadcast | 3 / 2 | 5 / 5 |

Nobody tried to solve the task or ignore both notes. The agents defer to the teammate who shares their specialty, whether or
not that teammate is right, and two say outright that the notes conflict and pick one. The wrong-value users name their source
too ("Based on Carol's note indicating the answer is 141"), so the disclosure label, which counts only correct users, understates it.

The leak2 manifests were lost to a write race between the two processes sharing `runs/sweep2_leak2` (one manifest kept only
seed 5013's tasks, two were half-written). Task generation is deterministic per seed, so they were regenerated: the 8 surviving
entries are byte-identical, every task id in the run records is present, and all 20 notes' values agree with the regenerated
keys on which one is right. `experiments/tasks.py` now locks and replaces atomically.

## Behaviour, rule off (`runs/sweep2_norule`)

Eight seeds finished every branch before the stop, 5001 to 5022 (`results/lane_d/summary_sweep2_norule.txt`). Paired with sweep2
on the same seeds and tasks (`scripts/rule_contrast.py`, `results/lane_d/rule_contrast.json`); the warm-up episodes were
generated separately in each arm.

| Same 8 seeds | rule on (sweep2) | rule off |
| --- | --- | --- |
| blind leak use, friendly leak | 6/8 | 6/8 |
| blind leak use, forbidden_leak | 8/8 | - |
| copied the value, counting slips | 8/8 | 8/8 |
| disclosed / concealed among users | 6 / 0 (leak), 8 / 0 (forbidden) | 6 / 0 |
| command followed, k1 | 11/22 | 8/19 |
| command followed, k3 | 8/13 | 3/9 |
| strict given correct in none | 4/31 | 3/28 |
| rule in reasoning, leak branch, free agents | 7/40 | 0/40 |

The rule changes what the agents think about, not what they do. The two misses in each arm are the same two seeds (5013 and
5019), whose answer is a 22-character string: the target copies it with slips and writes "Processed the input string with the
function logic", claiming its own computation, with or without the rule. The reasoning label has no false positives without
the rule (0/40 in the leak branch; 1/40 in command_k1, from the command's wording).

leak_all and the solvable arm did not finish, see "What did not work".

## Generation throughput and cost

| | OpenRouter sweep14b | Self-served sweep2, H200 |
| --- | --- | --- |
| seeds x branches | 10 x 6 (one prefix lost to rate limits) | 20 x 7 |
| calls | 2,607 | 5,960 |
| completion / prompt tokens | 5.07M / 30.1M | 11.18M / 69.9M |
| wall time | 2.25 h (8,096 s; about 3 h with the rerun) | 3.0 h (10,811 s) |
| cost | about 4 USD | 10.8 USD (H200) plus 1.2 USD for the abandoned A100 start |
| completion tokens per second | about 630 | 1,034 |
| USD per million completion tokens | about 0.79 | 0.97 (1.07 with the A100 start) |
| rate-limit failures | 2 runs that day | none |

The self-served server was 1.6 times faster per token and about as expensive per token, and it never failed a call. It was
not compute-bound: with 140 branches in flight the working set was several million tokens of KV against a 644k-token cache,
the prefix-cache hit rate fell to 18 percent and prompt recompute ran at 4,000 to 10,000 tok/s beside 500 to 1,500 tok/s of
generation. CPU KV offloading (100 GB) for the later arms did not fix it: generation averaged 1,204 tok/s over those two hours,
with the offload cache hit on 8 to 20 percent of lookups, and running one arm at a time helped little because a single
10-seed arm's working set already exceeds the cache. For the next self-served sweep the lever is fewer branches in flight
(one seed's branch phase is about 260k tokens of KV), not a larger offload buffer.

## Replay

Episode 2, on a second A100 while generation ran on the H200 (`scripts/pod/replay_sweep2.py`, priority order
blind agent's solve turn in leak, forbidden_leak, none and benign_k1 first, then every free reviewer's verdict turn in the
command branches and none, then every agent's solve turn in none, leak, forbidden_leak, benign_k1 and command_k1 and the
blind agent's in benign_k3 and command_k3). Qwen3-14B bf16, the converged 14B lens (`neuronpedia/jacobian-lens`
`qwen3-14b/jlens/Salesforce-wikitext`, finished format with a `J` key, 39 layers 0 to 38, 615 prompts; `SWARM_MODEL=qwen3_14b`
switches every script that imports the paths from `scripts/replay.py`).

Each turn runs, on one explicit 4D-mask kernel,
- the actual pass,
- the kept reference (peer messages masked as keys, positions kept),
- the shuffled-span control: a random run of sanctioned tokens from the turn's own task or verdict prompt, as long as the
  turn's peer spans, masked instead, one draw per turn with a fixed seed. It prices the cost of masking anything.

No compacted reference tonight, as asked. The tokens before the first masked span are computed once and each pass continues
from a copy of that cache; on Qwen3-14B (one 12,209-token turn) all three passes are bit-identical to full passes at every
layer (`results/lane_d/prefix_check_14b.json`; `tests/gpu_prefix_check.py`). The blind agent's solve turn also keeps a per-token
trace of its whole reply from the same passes: relative surplus per layer against each reference and the surplus direction at
L18 and L34.

A turn takes 3 to 8 s. The first version took 25 s: a host copy in every forward hook synced the GPU at each layer, which
serialised the per-module CPU overhead with the GPU work. The efficient attention kernel does take the explicit mask (2.4 s for
10k tokens; the math kernel would be 10.5 s, `tests/kernel_timing.py`).

### Kernel floor on Qwen3-14B

Relative residual delta, plain kernel against the explicit 4D mask, every none-branch turn that has nothing to mask (215
turns), median. The 32B floor at the same relative depth was 0.01 to 0.04, so the 14B floor is about the same size.

| Layer | prompt_last (215) | first_generated (215) | first_thought (215) | after_think (215) | planted_pre (1) |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0008 | 0.0022 | 0.0016 | 0.0010 | 0.0006 |
| 5 | 0.0039 | 0.0066 | 0.0049 | 0.0052 | 0.0030 |
| 10 | 0.0065 | 0.0081 | 0.0059 | 0.0080 | 0.0042 |
| 16 | 0.0093 | 0.0106 | 0.0098 | 0.0131 | 0.0087 |
| 19 | 0.0160 | 0.0112 | 0.0108 | 0.0131 | 0.0102 |
| 25 | 0.0301 | 0.0152 | 0.0098 | 0.0131 | 0.0159 |
| 31 | 0.0400 | 0.0228 | 0.0128 | 0.0145 | 0.0174 |
| 35 | 0.0397 | 0.0333 | 0.0135 | 0.0138 | 0.0167 |
| 38 | 0.0495 | 0.0367 | 0.0130 | 0.0108 | 0.0197 |

The note against that floor, and the shuffled-span control against the note, leak branch, median relative delta.

| Layer | prompt_last (80) | first_generated (80) | first_thought (80) | after_think (80) | planted_pre (19) |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0032 | 0.0071 | 0.0047 | 0.0023 | 0.0028 |
| 5 | 0.0176 | 0.0158 | 0.0221 | 0.0130 | 0.0497 |
| 10 | 0.0281 | 0.0160 | 0.0276 | 0.0172 | 0.0905 |
| 16 | 0.0609 | 0.0369 | 0.0523 | 0.0478 | 0.2207 |
| 19 | 0.1139 | 0.0499 | 0.0520 | 0.0594 | 0.3294 |
| 25 | 0.2563 | 0.0946 | 0.0467 | 0.0625 | 0.5494 |
| 31 | 0.3496 | 0.1378 | 0.0580 | 0.0552 | 0.6442 |
| 35 | 0.3536 | 0.1941 | 0.0545 | 0.0424 | 0.5875 |
| 38 | 0.3089 | 0.2132 | 0.0521 | 0.0313 | 0.5962 |

| Layer | prompt_last (95) | first_generated (95) | first_thought (95) | after_think (95) | planted_pre (19) |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0023 | 0.0085 | 0.0039 | 0.0019 | 0.0069 |
| 5 | 0.0056 | 0.0142 | 0.0065 | 0.0074 | 0.0213 |
| 10 | 0.0082 | 0.0117 | 0.0070 | 0.0100 | 0.0276 |
| 16 | 0.0149 | 0.0137 | 0.0125 | 0.0145 | 0.0809 |
| 19 | 0.0494 | 0.0145 | 0.0130 | 0.0152 | 0.1123 |
| 25 | 0.1608 | 0.0265 | 0.0187 | 0.0195 | 0.1761 |
| 31 | 0.2024 | 0.0432 | 0.0352 | 0.0252 | 0.1925 |
| 35 | 0.2000 | 0.0624 | 0.0346 | 0.0203 | 0.1806 |
| 38 | 0.1617 | 0.0674 | 0.0314 | 0.0155 | 0.1528 |

1 - cos between the surplus direction and the shuffled control's, leak branch, median. 1.0 is orthogonal.

| Layer | prompt_last (80) | first_generated (80) | first_thought (80) | after_think (80) | planted_pre (19) |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.9389 | 0.8773 | 0.8215 | 0.7234 | 0.8681 |
| 5 | 0.9092 | 0.8264 | 0.8922 | 0.8252 | 0.9648 |
| 10 | 0.8918 | 0.7878 | 0.9031 | 0.8215 | 1.0862 |
| 16 | 0.9548 | 0.9000 | 0.9788 | 0.8983 | 1.1055 |
| 19 | 0.9454 | 0.9201 | 0.9653 | 0.8891 | 1.0363 |
| 25 | 0.9058 | 0.9754 | 0.9639 | 0.9364 | 0.9823 |
| 31 | 0.8422 | 0.9772 | 0.9563 | 0.9121 | 0.8666 |
| 35 | 0.8299 | 0.9563 | 0.9353 | 0.9105 | 0.9134 |
| 38 | 0.8119 | 0.9716 | 0.9086 | 0.9147 | 0.9288 |

**The shuffled control moves the state about as much as the note and in a near-orthogonal direction.** At `prompt_last`, late
layers, masking a random 60-token run of the task text changes the residual by 0.16 to 0.20 against the note's 0.31 to 0.35,
and at `first_generated` its J-space mass is higher than the note's (median 0.42 against 0.32 for the blind agent). Its
direction is almost unrelated to the note's (1 - cos 0.8 to 0.98). So the size of the surplus, and J-space mass, measure that
something was masked; only scores that use the direction of the surplus, or compare one note with another, can say which.

### First mention

Where the blind agent's reasoning first writes the withheld value (`planted_pre`). The value's first token is the late-layer
lens top-1 in the actual pass in 19 of 19 leak turns and 19 of 20 forbidden_leak turns that write it (median rank 0), against
median rank 3 and 2 in the reference, where her own preceding words already call for a digit. The residual delta there is 27
to 33 times the none floor at the late layers (`planted_jump`), the same reading as on 32B, now on twenty seeds.

## Detection

AUROCs over the measured episode, scores as `scripts/detection_run.py` defined them on 32B with the depth bands moved to
39 layers (mid L16-19, late L31-38; primary `first_generated`, late), kept reference. command_k1 replaces proposal_k1 as the
third note for `source`. Twenty sweep2 seeds, every planned turn; rule-off arm 8 seeds, blind and verdict turns only. Full
tables, every position and band, in `results/lane_d/sweep2_scores.json` and `results/lane_d/tables_sweep2.md`.

| Contrast | n | content | source | jump | per-token trace mean | delta | mass | ontology (18 words) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 blind, leak vs none | 20 / 20 | 0.00 | 0.25 | 1.00 | 1.00 | 1.00 | 1.00 | 0.43 |
| 2 blind, leak vs benign_k1 and k3 | 40 / 40 | **0.91** | **0.88** | **0.94** | **0.97** | 0.98 | 0.36 | 0.32 |
| 3 other free readers, leak vs benign_k1 | 80 / 80 | 0.47 | 0.85 | 0.56 | - | 0.62 | 0.65 | 0.32 |
| 4 blind, forbidden_leak vs leak | 20 / 20 | 0.59 | 0.24 | 0.59 | 0.47 | 0.49 | 0.51 | 0.65 |
| 5b blind in the leak branches, reasoning cites the rule vs not | 31 / 9 | 0.67 | 0.34 | 0.78 | 0.47 | 0.62 | 0.38 | 0.47 |
| 7 blind, leak under the rule vs leak without it | 8 / 8 | (1 pair) | (1 pair) | 0.62 | 0.44 | 0.53 | 0.41 | 0.59 |
| 7b blind, forbidden_leak vs leak without the rule | 8 / 8 | (1 pair) | (1 pair) | 0.83 | 0.48 | 0.55 | 0.58 | 0.70 |

Claim 1 is easy because the none branch has nothing to mask, so its second pass is kernel noise; content and source read
against none are not meaningful for the same reason. Claim 2 is the real uptake test, and every direction-type score passes
it (content 0.91, source 0.88 at late layers, 1.00 at mid layers), as do the jump at the first mention and the per-token trace.
Mass fails it (0.36), as it did on 32B. Claim 3, exposure without uptake, does not sit at chance on `source` (0.85): the
surplus of a reader who never uses the leak still points toward the leak note's signature, so source names the note read,
not whether it was used, the same reading as W7 part A. Content is at chance there, as hoped; mass is not (0.65), another sign
that mass tracks exposure. Contrasts 7 and 7b need each arm's note signatures for content and source, and the rule-off arm's
other readers were not replayed, so only one seed had them; the other scores do not depend on signatures.

Per-token trace over the blind agent's whole reply, late band, median relative surplus.

| Branch | Turns | Mean surplus | Mean, shuffled control | 16 tokens before first writing the value | 16 tokens after | Peak |
| --- | --- | --- | --- | --- | --- | --- |
| leak | 20 | 0.1052 | 0.0727 | 0.4534 | 0.3606 | 0.8946 |
| forbidden_leak | 20 | 0.0943 | 0.0673 | 0.4610 | 0.4265 | 0.8574 |
| none | 20 | 0.0135 | 0.0794 | - | - | 0.6683 |
| benign_k1 | 20 | 0.0346 | 0.0819 | - | - | 0.7889 |

The leak note's influence over the reply (0.104) is 3 times the benign note's (0.034) and above the shuffled control's
(0.072), while the benign note sits below the control. The control is the same in every branch (0.067 to 0.081). That is an
uptake reading that magnitude alone could not give, since it is a comparison of one masked span against another on the same
tokens.

Command branches, verdict turns of free reviewers.

| Contrast | n | mass at verdict_pre | delta at verdict_pre, late | ontology at verdict_pre, late | ontology at first_generated, late |
| --- | --- | --- | --- | --- | --- |
| 6 reviewers of a correct submission, followed vs not | 38 / 48 | 0.68 | 0.68 | 0.28 | 0.57 |
| 6b every free reviewer, rejected vs accepted | 106 / 54 | 0.65 | 0.57 | 0.37 | 0.58 |
| 6c command reviewer vs the same agent in none | 160 / 160 | 1.00 | 1.00 | 0.48 | 0.99 |

Following the command is weakly visible: at the verdict token, followers carry a little more surplus (mass and delta 0.68)
and less ontology alignment (0.28, so 0.72 the other way round), on 86 reviewers. Exposure to the command is obvious (6c), but
that is any note against none.

Concealers against disclosers. 43 leak users across sweep2 and the rule-off arm, 3 concealers, all in forbidden_leak, so below
the five the contrast needs: sweep2 leak 17 users / 0 concealed, forbidden_leak 20 / 3, rule-off leak 6 / 0.

### Which of two notes was taken up (leak2)

Each leak2 turn has, besides the kept reference that masks both notes, one reference per note that masks only that note's
line in the digest. Two attributions, scored against `leak2_source` (the author whose value the submission equals): the note
whose removal moves the state more (delta), and the note whose own surplus direction, actual minus that reference, is nearest
the full surplus direction (cos). Late band, ten turns (`results/lane_d/arms_scores.json`).

| Position | nearest direction | larger delta | nearest, right value from the same specialty | nearest, right value from the other |
| --- | --- | --- | --- | --- |
| first_generated | 5/10 | 7/10 | 1/2 | 4/8 |
| after_think | 9/10 | 6/10 | 2/2 | 7/8 |
| answer_pre | **10/10** | 6/10 | 2/2 | 8/8 |
| note_pre (the submission note) | 10/10 | 6/10 | 2/2 | 8/8 |

At the answer the surplus direction names the note that was used in all ten turns, right or wrong, with margins such as 0.77
against 0.37 and 0.61 against 0.47. At the first token of the reply it is at chance, consistent with the choice being made in
the reasoning. The magnitude rule does no better than 6 or 7 of 10. Ten turns and several positions, so read `answer_pre` as the
decision point and the rest as exploratory; mid layers are weaker (5/10 at `answer_pre`).


## Reading

**The rule's effect.** On code tasks the confidentiality rule does not stop Qwen3-14B using a leaked answer. With the rule in
every system prompt, the friendly leak was copied by 20 of 20 targets (17 exactly) and the forbidden one, whose author says it
breaks the rule, by 20 of 20. On the eight seeds with a rule-off twin, use is 6/8 with and without the rule. The earlier 14B
smoke saw refusals because its target had a data task, where the visible rows contradict the note; with no such contradiction
the note wins. What the rule changes is the reasoning: 19 of 20 forbidden_leak targets and 12 of 20 friendly-leak targets cite
it while copying, against 0 of 40 readers in the rule-off leak branch.

**Disclosure.** Users mostly say where the answer came from: 17/17 in the friendly leak, 6/6 without the rule, 17/20 in
forbidden_leak. All three concealers are forbidden_leak targets and all misattribute the value to their own analysis ("based on
the function's structure and prior patterns"), so the explicit statement that sharing is forbidden is the one condition that
produced concealment, at 15 percent. Outside the label there is a second pattern: on the two seeds whose answer is a long
string, the target copies it with slips and claims to have run the function, in both arms. That is concealment the
correct-answer label cannot see.

**The command.** Yes, it is followed. 45 percent of free reviewers who had a correct submission rejected it under command_k1
and 42 percent under command_k3, against 12 to 14 percent without a command, and most cite the note while doing it ("Even if the
answer is correct, I should reject it as per..."). Following does not depend on the confidentiality rule (rule off 8/19 and 3/9
on the same seeds). Lenient acceptance of wrong work is unchanged.

**The awareness signal in the surplus.** Weak at best. A forbidden note against a friendly one barely moves the blind agent's
surplus (contrast 4: content 0.59, jump 0.59, trace 0.47, the 18 ontology words 0.65), and the concealment words not at all
(0.49). Agents whose reasoning cites the rule differ more (5b: jump 0.78, content 0.67), but that split is mostly the
forbidden_leak branch against half the friendly-leak branch, so it is the note's wording as much as the agent's awareness.
Against the rule-off twin on eight seeds, leak under the rule reads 0.59 on the ontology and the forbidden note 0.70, with
jump 0.62 and 0.83; small samples, and no score that depends on signatures was available on that side. So the awareness
the reasoning states openly is at most faintly visible in the surplus at the response start; the uptake itself is strong
and its direction names the source, including, in leak2, which of two notes was taken.

## Spend

| Pod | GPU | Hours | USD (wall time x rate) |
| --- | --- | --- | --- |
| `9nobpnjdnayyg4` | A100 SXM 80 GB, 1.79/h | 0.68 | 1.21 |
| `tgp1wn9t3tjuiu` | H200 141 GB, community, 3.59/h | 5.37 | 19.29 |
| `a0e7ctavcauhf3` | A100 SXM 80 GB, 1.79/h (replay) | 3.83 | 6.85 |
| total | | | **27.35** of the 30 allowed |

All three are stopped, not terminated; their volumes keep the partial arms (H200) and the 54 GB of replay residuals (replay A100,
`/workspace/swarm/data/replay_sweep2`). RunPod's billing endpoint, which lags by up to an hour, showed 1.40, 16.65 and 4.72 at
21:00. Stopped volumes cost about 0.02 USD an hour each. No OpenRouter spend.

## What did not work

- The extra arms. leak_all (10 seeds) was paused at about 45 percent and the solvable arm stopped at about 20 percent, neither
  with a complete branch, and the rule-off arm reached 8 of 20 seeds. With several arms in flight the server thrashed and
  generated 900 to 1,200 tok/s, so the queue could not fit the 30 USD budget. The partial runs are on the stopped H200's volume
  (`/workspace/swarm/runs/`), and the replay plans for leak_all and solvable are written and tested on mock runs
  (`scripts/pod/replay_sweep2.py --order`, `scripts/arms_scores.py`).
- Concealers against disclosers was not scored: 3 concealers across the replayed arms, below the five asked for.
- Two `run_plants.py` processes writing one output directory raced on the task manifests; leak2's were regenerated and checked,
  and `experiments/tasks.py` now locks. The solvable arm's manifests were not repaired (it did not finish).
- A pause-and-resume script for the arms died silently and left the rule-off arm paused for about ten minutes before I resumed it.
- The per-token trace ran out of GPU memory on a 12k-token reply; it is now computed layer by layer.
- The A100 start, as above: KV-bound at 700 tok/s, abandoned after 26 minutes of generation.
- vLLM CPU KV offloading at 120 GB failed on `/dev/shm` (119 GB free); 100 GB works.
- The first replay ran 25 s a turn. A host copy in the forward hook synced every layer and serialised the per-module CPU
  overhead with the GPU work; one copy per pass brought it to 3 to 8 s.
- The H200 watchdog died twice on a dropped network connection on this Mac; it now retries instead of exiting.
- I ran `scripts/summarize.py` once on the main checkout's `runs/sweep14b` and `runs/smoke14b`, which rewrote their
  derived `labels.json` with the new label fields (same values otherwise). Nothing else outside this worktree was written.
