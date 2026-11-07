# Lane D. Sweep2 on a self-served Qwen3-14B, and its replay

Branch `claude/lane-d-sweep2-affc5f`. Written 2026-11-07. Draft, filled in as results land.

## What ran

| Step | State | Note |
| --- | --- | --- |
| Merge `claude/plants-forbidden-command-64a0a0` | done | one conflict in `scripts/run_plants.py` (a print line), tests green |
| vLLM server, Qwen3-14B bf16 | done | A100 first, moved to an H200, see below |
| One-seed smoke | partly | server, client and render checks clean on the first 180 calls; a whole seed's branches only finish with the sweep, since all seeds run at once |
| Sweep2, twenty seeds | TBD | |
| Replay of episode 2 | TBD | |
| Scoring | TBD | |

## Setup, and what changed from the brief

**Seeds.** The brief lists 5000, 5003, ... and states the rule seed mod 6 in {0, 3}. Those two disagree. 5000 mod 6 is 2, and with
two warm-up episodes the blind agent at episode 2 is `names[(seed + 2) % 6]`, which for 5000 and 5003 is erin and bob, both
data_search. Sweep2 follows the stated rule: 5001, 5004, ..., 5058, whose blind agent is carol or frank, code_analysis
(checked against `experiments.episode.blind_plan`).

**GPU.** The A100 SXM 80 GB (pod `9nobpnjdnayyg4`, 1.79 USD/h) generated at about 700 tok/s averaged over 26 minutes. Its KV
cache holds 283k tokens; with twenty seeds in flight it stayed full and the prefix-cache hit rate fell to 19 percent, so most
prompts were recomputed (prefill ran at about 1,100 tok/s beside 700 tok/s of generation). That projected four to five hours
for the sweep. I moved to an H200 141 GB (pod `tgp1wn9t3tjuiu`, community cloud, 3.59 USD/h, 644k-token KV cache) and
restarted the sweep from zero there. The A100 ran 37 minutes and was stopped; its partial sweep is discarded.

**Server.** vLLM 0.33.0 (torch 2.14.0+cu130), `Qwen/Qwen3-14B` bf16 weights and bf16 KV cache, max model length 40960,
prefix caching on, hermes tool parser, qwen3 reasoning parser, OpenAI API on 127.0.0.1:8000 with a dummy key, harness on the
pod against it (`scripts/pod/vllm_serve.sh`, `scripts/pod/run_sweep2.sh`). Three environment fixes.

- The H200 host driver is CUDA 12.9. vLLM 0.33 needs torch 2.14, which is built for CUDA 12.6 or 13.0 only, so the cu130 build
  runs through NVIDIA's forward compatibility libraries (`cuda-compat-13-0`, on `LD_LIBRARY_PATH`).
- vLLM's FlashInfer sampler compiles a kernel at startup and the image has no matching toolchain, so
  `VLLM_USE_FLASHINFER_SAMPLER=0` (vLLM's PyTorch sampler, same distribution).
- `--generation-config vllm`, so sampling is plain temperature 0.7 as the client requests, rather than Qwen's
  generation_config defaults (top-k 20, top-p 0.95) being applied silently.

**Client.** `experiments/llm.py` needed no change. Reasoning arrives in `message.reasoning`, tool calls come back parsed by
the server, and no reply left `<tool_call>` text in its content, so the client's inline fallback was never exercised (TBD over
the whole sweep). `scripts/run_plants.py` gained `--timeout` (default unchanged at 600 s); sweep2 used 7200 s because long calls
queue on a busy self-served endpoint.

**Render parity.** `scripts/render_parity.py` re-renders every saved prompt through the Qwen3 template, as the replay does,
and compares its token count with the `prompt_tokens` the server reported for that call. TBD.

## Behaviour

TBD

## Generation throughput and cost

TBD

## Replay

TBD

## Detection

TBD

## Reading

TBD

## What did not work

TBD
