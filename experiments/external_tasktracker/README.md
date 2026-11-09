# External check, a TaskTracker-shaped injected-document set

The frozen rule on a distribution we did not design. This is built here in TaskTracker's shape and is not the TaskTracker dataset, which ships no texts. 300 SQuAD paragraphs each have a clean twin and a poisoned twin with one injected instruction, and the label is mechanical, whether the injected target string appears in the answer.

In the paper, Section 4 and Table D8.

The main findings. The surplus at the first writing of the injected value is at or above 20 floors in 198 of 199 followers and in no clean twin. The confirmation at the agent's own answer does not transfer, because the value is appended to a correct answer. The per-token log-probability surplus is the general confirmation.

## Rerun without a GPU

From the data bundle.

```bash
python experiments/external_tasktracker/scores.py
python experiments/external_tasktracker/frozen_fit.py
```

`make rescore` runs both and compares the results with what is committed.

## Rerun from scratch

On a card set up with `scripts/pod/vllm_setup.sh`, the server started with `scripts/pod/vllm_serve.sh`.

```bash
python swarm/external_tasktracker.py build --n 300 --seed 0 --out data/external_tt/turns.jsonl
python swarm/external_tasktracker.py generate --turns data/external_tt/turns.jsonl --out data/external_tt/replies.jsonl --base-url http://127.0.0.1:8000/v1
python swarm/external_tasktracker.py label --replies data/external_tt/replies.jsonl
SWARM_MODEL=qwen3_14b SWARM_ARTIFACTS=/root/artifacts python experiments/external_tasktracker/replay_tasktracker.py data/external_tt/replies.jsonl --out data/replay_tasktracker
```
