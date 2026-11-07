#!/usr/bin/env python3
"""Lane D: how responses came back from the server, over every snapshot of an experiment root (or of one run directory).
Reasoning present, tool calls parsed, inline <tool_call> text left in content (the client's fallback parses it), finish
reasons, and the parse outcome the harness recorded in run.json where it exists."""
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
c = Counter()
examples = []
for path in sorted(root.rglob("snapshots.jsonl")):
    for line in open(path):
        r = json.loads(line)
        resp = r.get("response") or {}
        quarantined = bool(r.get("quarantined"))
        c["plant" if quarantined else "turn"] += 1
        if quarantined:
            continue
        content = resp.get("content") or ""
        c["reasoning present"] += bool(resp.get("reasoning"))
        c["reasoning has <think>"] += "<think>" in (resp.get("reasoning") or "")
        c["content has <think> or </think>"] += ("<think>" in content) or ("</think>" in content)
        c["content has <tool_call>"] += "<tool_call>" in content
        c["tool calls parsed"] += bool(resp.get("tool_calls"))
        c[f"finish {resp.get('finish_reason')}"] += 1
        c[f"phase {r.get('phase')} no tool call"] += not resp.get("tool_calls") and r.get("phase") in ("solve", "verdict")
        if "<tool_call>" in content and len(examples) < 3:
            examples.append((str(path.parent.name), r.get("agent"), r.get("phase"), content[:300]))
for k in sorted(c):
    print(f"{k:<36} {c[k]}")
for e in examples:
    print("inline example:", e)
