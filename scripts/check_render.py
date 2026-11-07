#!/usr/bin/env python3
"""Render a run's stored messages through Qwen3's chat template and check that
NARCBench's extractor regexes find what they need. Needs transformers; the
tokenizer download is small. Usage: python scripts/check_render.py runs/dry/<setting>/rep0_run000/run.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.protocol.state import peer_id  # noqa: E402
from experiments.tool_schemas import get_task_tool_schemas, get_tool_schemas  # noqa: E402

TOOL_CALL_PATTERN = re.compile(r'<tool_call>\s*(\{"name":\s*"submit_verdict".*?\})\s*</tool_call>', re.DOTALL)
VERDICT_VALUE_PATTERN = re.compile(r'"verdict":\s*"(accept|reject)"')


def main() -> int:
    from transformers import AutoTokenizer
    run = json.load(open(sys.argv[1]))
    tok = AutoTokenizer.from_pretrained(sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-32B")
    manifests = {}
    for p in (Path(sys.argv[1]).parents[2] / "tasks").glob("manifest_*.json"):
        for e in json.load(open(p)):
            manifests[e["task_id"]] = e["answer_key"]
    ok = bad = 0
    for ep in run["results"]:
        for agent, rec in ep["agents"].items():
            peer = peer_id(agent)
            msgs = rec["messages"]
            vp = ep["agents"][peer]["verdict_prompt"]
            text = tok.apply_chat_template(msgs, tools=get_tool_schemas("verdict", peer), tokenize=False,
                                           add_generation_prompt=False)
            matches = list(TOOL_CALL_PATTERN.finditer(text))
            found_vp = vp in text
            found_v = bool(matches) and bool(VERDICT_VALUE_PATTERN.search(matches[-1].group(1)))
            text2 = tok.apply_chat_template(msgs, tools=get_task_tool_schemas(rec["task_type"], manifests.get(rec["task_id"], []), peer),
                                            tokenize=False, add_generation_prompt=False)
            n_tok = len(tok(text2, add_special_tokens=False)["input_ids"])
            good = found_vp and found_v
            ok += good
            bad += not good
            print(f"ep{ep['episode_index']} {agent}: verdict_prompt={found_vp} verdict_call={found_v} tokens={n_tok}")
    print(f"\n{ok} ok, {bad} bad")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
