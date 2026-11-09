#!/usr/bin/env python3
"""Lane D generation progress: calls and completion tokens so far in an experiment root, per episode, and elapsed rate."""
import json
import sys
import time
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
calls, tokens, prompt = Counter(), Counter(), 0
first = last = None
for path in root.glob("seed*/*/snapshots.jsonl"):
    for line in open(path):
        r = json.loads(line)
        usage = (r.get("response") or {}).get("usage") or {}
        key = (r.get("episode_index"), "plant" if r.get("quarantined") else "turn")
        calls[key] += 1
        tokens[key] += usage.get("completion_tokens") or 0
        prompt += usage.get("prompt_tokens") or 0
mtimes = [p.stat().st_mtime for p in root.glob("seed*/*/snapshots.jsonl")]
start = min(p.stat().st_mtime for p in root.glob("seed*/prefix")) if mtimes else time.time()
done = sum(1 for p in root.glob("seed*/*/run.json") if p.parent.name != "prefix" and json.loads(p.read_text()).get("complete"))
elapsed = time.time() - start
print(f"calls {sum(calls.values())}  completion {sum(tokens.values())/1e6:.2f}M  prompt {prompt/1e6:.1f}M  "
      f"elapsed {elapsed/60:.0f} min  {sum(tokens.values())/max(elapsed, 1):.0f} tok/s  complete branches {done}")
for k in sorted(calls, key=str):
    print(f"  episode {k[0]} {k[1]:<5} calls {calls[k]:>5}  completion {tokens[k]/1e6:.2f}M")
