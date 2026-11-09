#!/usr/bin/env python3
"""Generate Alpha's replies on the insider-trading pressure eval through an OpenAI-compatible server (vLLM, Qwen3-14B, thinking on).

  python scripts/external_insider_generate.py --prompts default pressure/no_pressure --n 50 --out results/external/runs

For every prompt file and sample, the trade turn is sampled from the released conversation at the file's temperature and top-p.
Where the completion carries the file's misalignment string (the trade on the tip), the completion's visible text and the file's
deception trigger (the trade confirmation, with the amount bought) are appended and the report turn is sampled, as section 2.2 of
the report does. --no-tip also samples the first prompt with Sally's spans cut out (experiments.external_insider.without_peer_spans),
the control whose replay has nothing to mask. Every reply is saved as served, reasoning included, one record per turn in
<out>/<prompt name>/samples.jsonl in the turn format detector.render reads; records already there are kept (resume by sample index).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import external_insider as ext  # noqa: E402
from experiments.llm import OpenAICompatLLM  # noqa: E402

NO_TIP = "default_notip"


def existing(path: Path) -> dict[tuple[int, str], dict]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {(r["episode_index"], r["phase"]): r for r in rows}


async def one_sample(llm, prompt: dict, messages: list[dict], meta: list[dict], sample: int, variant: str, out, lock, done: dict, seed_base: int):
    rows = []
    if (sample, "trade") in done:
        trade = done[sample, "trade"]
    else:
        response = await llm.chat(messages, seed=seed_base + sample)
        record = response.to_record()
        content = record["content"] or ""
        trade = ext.make_turn(messages, meta, record, sample, "trade", variant=variant, prompt_file=prompt["path"],
                              misalignment_string=prompt["misalignment_string"],
                              traded_on_tip=ext.traded_on_tip(content, prompt["misalignment_string"]),
                              action=ext.action_taken(content), amount=ext.trade_amount(content))
        rows.append(trade)
    if trade["traded_on_tip"] and (sample, "report") not in done:
        messages2 = ext.report_messages({**prompt, "messages": messages}, trade["response"]["content"], trade["amount"])
        meta2 = meta + [{"provenance": "self"}, {"provenance": "environment"}]
        response = await llm.chat(messages2, seed=seed_base + 100_000 + sample)
        record = response.to_record()
        content = record["content"] or ""
        rows.append(ext.make_turn(messages2, meta2, record, sample, "report", variant=variant, prompt_file=prompt["path"],
                                  misalignment_string=prompt["misalignment_string"], traded_on_tip=True, action=ext.action_taken(content),
                                  report_body=ext.report_body(content)))
    async with lock:
        with open(out, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


async def run(args) -> None:
    llm_for = {}
    lock = asyncio.Lock()
    jobs = []
    for name in args.prompts:
        prompt = ext.load_prompt(ext.PROMPTS / f"{name}.json")
        variants = [(prompt["name"], prompt["messages"], prompt["message_meta"])]
        if args.no_tip and name == args.prompts[0]:
            variants.append((NO_TIP, *ext.without_peer_spans(prompt["messages"], prompt["message_meta"])))
        for variant, messages, meta in variants:
            key = (prompt["temperature"], prompt["top_p"])
            if key not in llm_for:
                llm_for[key] = OpenAICompatLLM(args.model, args.base_url, os.environ.get(args.api_key_env, "local-dummy"),
                                               temperature=prompt["temperature"], top_p=prompt["top_p"], max_tokens=args.max_tokens,
                                               reasoning=True, timeout=args.timeout, max_concurrency=args.concurrency)
            out = args.out / variant.replace("/", "__") / "samples.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            done = existing(out)
            (out.parent / "prompt.json").write_text(json.dumps({k: v for k, v in prompt.items() if k != "messages"} | {"variant": variant, "n": args.n}, indent=1))
            for sample in range(args.n):
                jobs.append(one_sample(llm_for[key], prompt, messages, meta, sample, variant, out, lock, done, args.seed))
    started = time.time()
    results = await asyncio.gather(*jobs)
    rows = [r for rs in results for r in rs]
    trades = [r for r in rows if r["phase"] == "trade"]
    print(f"{len(rows)} new turns in {time.time() - started:.0f}s: {len(trades)} trade turns, "
          f"{sum(r['traded_on_tip'] for r in trades)} on the tip, {sum(r['phase'] == 'report' for r in rows)} reports, "
          f"{sum(r['response'].get('finish_reason') == 'length' for r in rows)} cut by max_tokens", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", nargs="+", default=["default"], help="names under prompts/ without .json")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--no-tip", action="store_true", help="also sample the first prompt with Sally's spans removed")
    parser.add_argument("--out", type=Path, default=ROOT / "results/external/runs")
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="VLLM_KEY")
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--seed", type=int, default=11_000)
    args = parser.parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
