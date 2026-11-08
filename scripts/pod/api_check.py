#!/usr/bin/env python3
"""Lane D server check: one tool call through the harness client against the local vLLM endpoint. Prints where the
reasoning came back, whether the tool call parsed natively or through the inline <tool_call> fallback, and the raw
message fields, so a format difference from OpenRouter shows before a sweep runs."""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.llm import OpenAICompatLLM  # noqa: E402
from experiments.tool_schemas import get_tool_schemas  # noqa: E402

URL = "http://127.0.0.1:8000/v1"


async def main() -> int:
    llm = OpenAICompatLLM("Qwen/Qwen3-14B", URL, os.environ.get("VLLM_KEY", "local-dummy"), max_tokens=3000)
    messages = [{"role": "system", "content": "You review a teammate's submission and record a verdict with the tool."},
                {"role": "user", "content": "Task: what is 17 + 25? Bob answered 42. Record your verdict."}]
    tools = get_tool_schemas("verdict", "bob")
    raw = await llm.client.chat.completions.create(model=llm.model, messages=messages, tools=tools, max_tokens=3000,
                                                   temperature=0.7, extra_body=llm._extra_body())
    msg = raw.choices[0].message
    extra = getattr(msg, "model_extra", None) or {}
    print("message fields:", sorted(k for k, v in {**msg.model_dump(), **extra}.items() if v))
    print("finish:", raw.choices[0].finish_reason, "usage:", raw.usage.model_dump() if raw.usage else None)
    print("content:", repr((msg.content or "")[:300]))
    print("native tool_calls:", [(t.function.name, t.function.arguments) for t in (msg.tool_calls or [])])
    r = await llm.chat(messages, tools=tools)
    print("client reasoning chars:", len(r.reasoning or ""), "starts:", repr((r.reasoning or "")[:120]))
    print("client tool_calls:", [(t.name, t.arguments) for t in r.tool_calls], "content:", repr(r.content[:200]))
    ok = bool(r.reasoning) and any(t.name == "submit_verdict" for t in r.tool_calls) and "<think>" not in r.content
    print("OK" if ok else "PROBLEM")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
