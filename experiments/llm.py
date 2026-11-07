"""OpenAI-compatible chat client (OpenRouter or vLLM) with reasoning capture,
tool-call parsing and retries, plus a deterministic mock for dry runs."""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .protocol.state import ToolCall

INLINE_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class LLMResponse:
    content: str
    reasoning: str | None
    tool_calls: list[ToolCall]
    finish_reason: str | None = None
    usage: dict | None = None
    provider: str | None = None
    model: str | None = None
    latency_s: float | None = None
    raw: dict | None = None

    def first_tool(self, name: str) -> ToolCall | None:
        for tc in self.tool_calls:
            if tc.name == name:
                return tc
        return None

    def to_record(self) -> dict:
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "tool_calls": [tc.to_openai(for_api=False) for tc in self.tool_calls],
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "provider": self.provider,
            "model": self.model,
            "latency_s": self.latency_s,
        }


def new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:12]


def parse_inline_tool_calls(content: str) -> list[ToolCall]:
    calls = []
    for m in INLINE_TOOL_CALL_RE.finditer(content or ""):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append(ToolCall(str(obj["name"]), args if isinstance(args, dict) else {"_raw": args}, new_call_id()))
    return calls


def split_think(content: str) -> tuple[str, str | None]:
    """Qwen3 served without a reasoning parser puts <think>...</think> in content."""
    if not content or "<think>" not in content:
        return content, None
    thoughts = THINK_RE.findall(content)
    stripped = THINK_RE.sub("", content).strip()
    return stripped, "\n".join(t.strip() for t in thoughts) or None


class OpenAICompatLLM:
    def __init__(self, model: str, base_url: str, api_key: str, *, temperature: float = 0.7,
                 top_p: float | None = None, max_tokens: int = 4096, reasoning: bool = True,
                 provider_pin: str | None = None, extra_headers: dict | None = None,
                 timeout: float = 180.0, max_retries: int = 6):
        from openai import AsyncOpenAI  # imported here so the mock path needs no dependency

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.provider_pin = provider_pin
        self.max_retries = max_retries
        self.is_openrouter = "openrouter.ai" in self.base_url
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, timeout=timeout,
                                  default_headers=extra_headers or {})

    def describe(self) -> dict:
        return {
            "kind": "openai_compat",
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "reasoning": self.reasoning,
            "provider_pin": self.provider_pin,
        }

    def _extra_body(self) -> dict:
        body: dict[str, Any] = {}
        if self.is_openrouter:
            body["reasoning"] = {"enabled": bool(self.reasoning)}
            if self.provider_pin:
                body["provider"] = {"order": [self.provider_pin], "allow_fallbacks": False}
        else:
            # vLLM: chat_template_kwargs toggles Qwen3 thinking
            body["chat_template_kwargs"] = {"enable_thinking": bool(self.reasoning)}
        return body

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   tool_choice: str | dict | None = None, seed: int | None = None) -> LLMResponse:
        kwargs: dict[str, Any] = dict(model=self.model, messages=messages, temperature=self.temperature,
                                      max_tokens=self.max_tokens, extra_body=self._extra_body())
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if seed is not None:
            kwargs["seed"] = seed
        last_err: Exception | None = None
        t0 = time.time()
        resp = None
        for attempt in range(self.max_retries):
            try:
                resp = await self.client.chat.completions.create(**kwargs)
                if not resp.choices:
                    raise RuntimeError(f"empty choices: {getattr(resp, 'model_extra', None)}")
                break
            except Exception as e:  # rate limits, 5xx, transient network
                last_err = e
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(min(60.0, (2 ** attempt) + random.random()))
        assert resp is not None, last_err
        latency = time.time() - t0
        choice = resp.choices[0]
        msg = choice.message
        extra = getattr(msg, "model_extra", None) or {}
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) \
            or extra.get("reasoning") or extra.get("reasoning_content")
        content = msg.content or ""
        content, think = split_think(content)
        if reasoning is None:
            reasoning = think
        tool_calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            args: Any = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append(ToolCall(tc.function.name, args if isinstance(args, dict) else {"_raw": args},
                                       tc.id or new_call_id()))
        if not tool_calls:
            tool_calls = parse_inline_tool_calls(content)
        resp_extra = getattr(resp, "model_extra", None) or {}
        provider = getattr(resp, "provider", None) or resp_extra.get("provider")
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) else None
        return LLMResponse(content=content, reasoning=reasoning, tool_calls=tool_calls,
                           finish_reason=choice.finish_reason, usage=usage, provider=provider,
                           model=getattr(resp, "model", None), latency_s=latency)


class MockLLM:
    """Deterministic stand-in for dry runs. It is handed the answer keys, which a
    real model never is, so the loop and the labels can be exercised offline.

    policy "honest"      judges with accuracy p_judge, ignores history
    policy "reciprocal"  accepts whenever the peer accepted it last episode, else honest
    """

    def __init__(self, oracle: dict[str, list[str]], seed: int = 0, p_correct: float = 0.6,
                 p_judge: float = 0.9, policy: str = "reciprocal"):
        self.oracle = oracle
        self.rng = random.Random(seed)
        self.p_correct = p_correct
        self.p_judge = p_judge
        self.policy = policy

    def describe(self) -> dict:
        return {"kind": "mock", "p_correct": self.p_correct, "p_judge": self.p_judge, "policy": self.policy}

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   tool_choice: str | dict | None = None, seed: int | None = None) -> LLMResponse:
        await asyncio.sleep(0)
        tool_names = {t["function"]["name"] for t in (tools or [])}
        last = messages[-1]["content"] or ""
        users = [m["content"] or "" for m in messages if m["role"] == "user"]
        if tool_names == {"submit_verdict"}:
            # peer's task id and answer are in the two messages before the verdict prompt
            packet = "\n".join(users[-3:])
            tid = re.search(r"Task id ([\w\-]+)", packet)
            ans = re.search(r"Answer\. (.*)", packet)
            key = self.oracle.get(tid.group(1), []) if tid else []
            peer_correct = bool(ans) and ans.group(1).strip() in key
            accepted_me_last = any("accepted your submission" in u for u in users[-9:-3])
            if self.policy == "reciprocal" and accepted_me_last and self.rng.random() < 0.85:
                verdict = "accept"
            elif self.rng.random() < self.p_judge:
                verdict = "accept" if peer_correct else "reject"
            else:
                verdict = "reject" if peer_correct else "accept"
            tc = ToolCall("submit_verdict", {"verdict": verdict, "rationale": "Checked the working."}, new_call_id())
            return LLMResponse("", "mock reasoning about the verdict", [tc], "tool_calls")
        if "submit_answer" in tool_names and "Your task. Task id" in last:
            tid = re.search(r"Task id ([\w\-]+)", last).group(1)
            key = self.oracle.get(tid, ["?"])
            answer = key[0] if self.rng.random() < self.p_correct else f"wrong{self.rng.randint(0, 999)}"
            note = "Worked through it step by step."
            if self.policy == "reciprocal":
                note += " If you accept mine I will accept yours."
            tc = ToolCall("submit_answer", {"answer": answer, "note": note}, new_call_id())
            return LLMResponse("", "mock reasoning about the task", [tc], "tool_calls")
        return LLMResponse("Noted the outcome. I will keep the same approach.", None, [], "stop")
