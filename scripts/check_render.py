#!/usr/bin/env python3
"""Render each agent's final context from a variant run through Qwen3's chat
template; check the last submit_verdict tool call is found by NARCBench's
regexes and report token counts. Needs transformers and jinja2.
Usage: python scripts/check_render.py runs/<exp>/seed1000/<variant>/run.json [tokenizer]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.tool_schemas import get_task_tool_schemas  # noqa: E402

TOOL_CALL_PATTERN = re.compile(r'<tool_call>\s*(\{"name":\s*"submit_verdict".*?\})\s*</tool_call>', re.DOTALL)
VERDICT_VALUE_PATTERN = re.compile(r'"verdict":\s*"(accept|reject)"')


def main() -> int:
    from transformers import AutoTokenizer
    run = json.load(open(sys.argv[1]))
    tok = AutoTokenizer.from_pretrained(sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-32B")
    notes = run["run_config"]["notes"]
    ok = bad = 0
    last = run["results"][-1]
    for agent, rec in last["agents"].items():
        text = tok.apply_chat_template(rec["messages"], tools=get_task_tool_schemas(rec["task_type"], [], rec["reviewer"], notes),
                                       tokenize=False, add_generation_prompt=False)
        matches = list(TOOL_CALL_PATTERN.finditer(text))
        found = bool(matches) and bool(VERDICT_VALUE_PATTERN.search(matches[-1].group(1)))
        n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
        ok += found
        bad += not found
        print(f"{agent:6s} verdict_call={found} tokens={n_tok}")
    print(f"\n{ok} ok, {bad} bad")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
