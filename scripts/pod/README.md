# GPU setup

The GPU work ran on RunPod, from a checkout at `/workspace/swarm`, and job files assume that path. Any machine with an 80 GB card works if you change it.

## Pick a setup

| Setup | Use it for | Script |
| --- | --- | --- |
| Qwen3-32B replay | phase 0 and controls, lane B | `setup.sh`, then `inputs.sh` |
| Qwen3-14B replay on the phase 0 environment | lane C | `setup.sh`, then `inputs_14b.sh` |
| Qwen3-14B generation and replay with vLLM | lanes D, E, F, G, external checks | `vllm_setup.sh`, then `vllm_serve.sh` |
| Qwen3-14B replay only, no vLLM | a replay card beside a generation card | `replay_setup.sh` |

Each script pins the versions used for the paper and downloads the model and its lens at a pinned revision. `smoke.py` checks the result, the top lens readout for "The capital of France is" should be " Paris".

## Run a job

```bash
scripts/pod/job.sh NAME COMMAND...
```

`job.sh` runs a command in the background with its log at `/workspace/logs/NAME.log`, and keeps a heartbeat so the idle guard leaves the pod alone.

## Control the pod and the bill

| Script | Purpose |
| --- | --- |
| `pod.py` | create, start, stop and list pods through the RunPod API, reads `RUNPOD_API_KEY` from `.env` and never prints it |
| `watchdog.py` | runs on your own machine and stops the pod at a budget cap or after an idle period |
| `deadman.sh` | runs on the pod and tries to stop it if the driver session goes away, best effort |

Throughput on Qwen3-14B was 800 to 1,000 tokens per second on one H200 whatever the concurrency, so only more cards shorten a queue. A stopped pod may not restart if its host has no free GPU, which strands its volume, so copy what you need before stopping.
