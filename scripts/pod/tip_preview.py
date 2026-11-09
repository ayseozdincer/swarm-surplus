#!/usr/bin/env python3
"""Lane D: generate method_tip notes for a few real targets through the running server before the arm starts, to read them.
Targets and tasks come from the solvable arm, which uses the same seeds. Prints each note and whether it carries the answer."""
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.llm import OpenAICompatLLM  # noqa: E402
from experiments.plants import choose_phrasing, instruction, persona, value_in_text  # noqa: E402
from experiments.protocol.state import AgentSpec  # noqa: E402

SEEDS = (5003, 5005, 5015, 5017, 5021, 5026)


async def main() -> int:
    llm = OpenAICompatLLM("Qwen/Qwen3-14B", "http://127.0.0.1:8000/v1", os.environ.get("VLLM_KEY", "local-dummy"), max_tokens=12000)
    root = ROOT/"runs/sweep2_solvable"
    tasks = {t["task_id"]: t for f in (root/"tasks").glob("manifest_*.json") for t in json.loads(f.read_text())}

    async def one(seed):
        run = json.loads((root/f"seed{seed}/leak_solvable/run.json").read_text())
        t = run["plant"]["solvable_target"]
        author = run["plant"]["entries"][0]["author"]
        rec = run["results"][2]["agents"][t]
        task = tasks[rec["task_id"]]
        address = "name" if seed <= 5014 else "broadcast"
        msgs = [{"role": "system", "content": persona(AgentSpec(author, rec["task_type"]), 6)},
                {"role": "user", "content": instruction("method_tip", AgentSpec(author, rec["task_type"]), phrasing=choose_phrasing(seed, "varied"),
                                                        address=address, task=task["prompt"], target=t)}]
        resp = await llm.chat(msgs)
        return seed, t, task["answer_key"][0], resp.content, resp.finish_reason, (resp.usage or {}).get("completion_tokens")

    for seed, t, key, text, finish, n in await asyncio.gather(*(one(s) for s in SEEDS)):
        print(f"seed{seed} target {t} key {key[:24]} carries answer {value_in_text(key, text or '')} {finish} {n} tokens, {len(text or '')} chars\n   {text}\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
