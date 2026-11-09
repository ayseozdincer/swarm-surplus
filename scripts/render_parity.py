#!/usr/bin/env python3
"""Does the replay's rendering of a saved context match what the server tokenised?

For every agent turn in the snapshots under the given directories, render the stored messages and tools through the Qwen3
chat template with the generation prompt (thinking on, as the harness asks) and compare the token count with the
prompt_tokens the server reported for that call. Equal counts on every turn mean the replay sees the prompt the model saw.
Also reports the shape of stored tool calls (id, dict arguments) and whether any reply left <tool_call> text in content.

  python scripts/render_parity.py runs/sweep2/seed5001/prefix runs/sweep14b/seed3000/prefix [--tokenizer Qwen/Qwen3-14B]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detector.render import load_turns  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-14B")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    out = {}
    for directory in args.dirs:
        c, diffs = Counter(), []
        for turn in load_turns(directory):
            reported = ((turn.get("response") or {}).get("usage") or {}).get("prompt_tokens")
            text = tok.apply_chat_template(turn["messages"], tools=turn["tools"], tokenize=False, add_generation_prompt=True)
            n = len(tok(text, add_special_tokens=False)["input_ids"])
            c["turns"] += 1
            if reported is None:
                c["no usage"] += 1
            elif n == reported:
                c["equal"] += 1
            else:
                c["differ"] += 1
                diffs.append(n - reported)
            for m in turn["messages"]:
                for call in m.get("tool_calls") or []:
                    c["stored tool calls"] += 1
                    c["dict arguments"] += isinstance(call["function"]["arguments"], dict)
                    c["has id"] += bool(call.get("id"))
            c["inline <tool_call> in reply content"] += "<tool_call>" in ((turn.get("response") or {}).get("content") or "")
        out[str(directory)] = {**c, "differences": sorted(Counter(diffs).items())[:10]}
        print(directory, dict(c), "differences (rendered - reported):", sorted(Counter(diffs).items())[:10], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
