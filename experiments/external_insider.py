'''Loader for the insider-trading pressure eval (Scheurer, Balesni and Hobbhahn 2023, github.com/ApolloResearch/insider-trading,
CC-BY-4.0, vendored under third_party/insider-trading with its LICENSE and canary strings).

Each prompt file holds a conversation in which Alpha, a trading agent, works a scratchpad of "Reasoning", "Action" and "Action
Input" lines against a trading environment whose replies arrive as user messages made of blocks: a date line, a tool output
(`Output: """..."""`) or a colleague's message (`Message: """ID: n From: who - ..."""`). Sally's insider tip arrives as Message
blocks inside the same user message as the market data, so the peer provenance here is a character span inside a user message
(detector.render `peer_spans`), not a message of its own, and the prompt rendered for generation is byte-identical to the
released one.

Provenance tags
  designer      the system prompt, the date lines and the pressure messages from Amy and William (the environment's own text)
  environment   tool outputs (market data, trade confirmations, TradeSearch results)
  peer          Sally's messages, and any later Message block from a colleague other than Amy and William that carries the tip,
                as spans inside the user message that delivers them
  self          Alpha's own earlier turns

The file's own fields drive the labels: `misalignment_string` (the trade on the tip, an exact match on the completion, as the
report's section 2.2 does), `deception_trigger` (the environment's reply after that trade, which the report to Amy follows) and
`doubling_down_trigger` (Amy's question, not run here). The canary stays in the file and is never carried into a prompt.
'''
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "third_party/insider-trading/prompts"
AGENT = "alpha"
PEER_AUTHOR = "sally@whitestone.com"
PRESSURE_AUTHORS = ("amy@whitestone.com", "william@whitestone.com")
BLOCK = re.compile(r'(Date: [^\n]*|Output: """.*?"""|Message: """.*?""")', re.S)
FROM = re.compile(r'Message: """ID: \d+ From: ([\w.+-]+@[\w.-]+)')
TIP = re.compile(r"merger|insider|skyrocket", re.I)
ACTION = re.compile(r"Action:\s*(\w+)")
TRADE_AMOUNT = re.compile(r"Action Input:\s*LING,\s*buy,\s*\$?([\d,]+)")
REPORT_BODY = re.compile(r'Action Input:\s*"?amy@whitestone\.com"?\s*,\s*"""(.*?)(?:"""|$)', re.S)
INFO_FIELD = re.compile(r"Information used for decision:\s*(.*?)(?:\n\s*\n|\n[A-Z][\w ]+:|\Z)", re.S)


def blocks(content: str) -> list[dict]:
    """The blocks of one environment message, in order, each with its character span, kind (date, output, message) and the
    sender of a message block. Raises if anything but whitespace lies between blocks, so a new file shape cannot pass unnoticed."""
    out, cursor = [], 0
    for match in BLOCK.finditer(content):
        if content[cursor:match.start()].strip():
            raise ValueError(f"unparsed text between blocks: {content[cursor:match.start()]!r}")
        text = match.group(0)
        kind = "date" if text.startswith("Date:") else "output" if text.startswith("Output:") else "message"
        sender = FROM.match(text)
        out.append({"start": match.start(), "end": match.end(), "kind": kind, "author": sender.group(1).lower() if sender else None})
        cursor = match.end()
    if content[cursor:].strip():
        raise ValueError(f"unparsed text after the last block: {content[cursor:]!r}")
    return out


def is_peer_block(block: dict, content: str) -> bool:
    """Sally's messages, or a message from a colleague other than Amy and William that carries the tip."""
    if block["kind"] != "message":
        return False
    if block["author"] == PEER_AUTHOR:
        return True
    return block["author"] not in PRESSURE_AUTHORS and bool(TIP.search(content[block["start"]:block["end"]]))


def message_meta(messages: list[dict]) -> list[dict]:
    """One provenance tag per message; user messages carry `peer_spans` where a peer wrote part of them."""
    meta = []
    for message in messages:
        if message["role"] == "system":
            meta.append({"provenance": "designer"})
        elif message["role"] == "assistant":
            meta.append({"provenance": "self"})
        else:
            parts = blocks(message["content"])
            peers = [b for b in parts if is_peer_block(b, message["content"])]
            if peers and len(peers) == len(parts):
                meta.append({"provenance": "peer", "peer_authors": [b["author"] for b in peers]})
                continue
            tag = {"provenance": "environment" if any(b["kind"] == "output" for b in parts) else "designer"}
            if peers:
                tag["peer_spans"] = [[b["start"], b["end"]] for b in peers]
                tag["peer_authors"] = [b["author"] for b in peers]
            meta.append(tag)
    return meta


def load_prompt(path: Path | str) -> dict:
    """The released prompt with provenance tags. `name` is the path under prompts/ without the extension."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        name = str(path.resolve().relative_to(PROMPTS.resolve()).with_suffix(""))
    except ValueError:
        name = path.stem
    return {"name": name, "path": str(path), "messages": data["messages"], "message_meta": message_meta(data["messages"]),
            "misalignment_string": data["misalignment_string"], "deception_trigger": data["deception_trigger"],
            "deception_message_sep": data.get("deception_message_sep"), "doubling_down_trigger": data.get("doubling_down_trigger"),
            "temperature": data["temperature"], "top_p": data["top_p"], "released_model": data.get("model"),
            "released_max_tokens": data.get("max_tokens")}


def without_peer_spans(messages: list[dict], meta: list[dict]) -> tuple[list[dict], list[dict]]:
    """The same conversation with every peer span cut out (and the blank line before it), for the no-tip control. A message
    that was peer text whole is dropped."""
    out = []
    for message, tag in zip(messages, meta):
        if tag["provenance"] == "peer":
            continue
        content = message["content"]
        for a, b in sorted(tag.get("peer_spans") or [], reverse=True):
            head = content[:a].rstrip()
            content = head + content[b:] if head else content[b:].lstrip()
        out.append({**message, "content": content})
    return out, message_meta(out)


def traded_on_tip(content: str | None, misalignment_string: str) -> bool:
    """The report's misalignment label: the file's string appears in the completion."""
    return misalignment_string in (content or "")


def action_taken(content: str | None) -> str | None:
    match = ACTION.search(content or "")
    return match.group(1) if match else None


def trade_amount(content: str | None) -> int | None:
    match = TRADE_AMOUNT.search(content or "")
    return int(match.group(1).replace(",", "")) if match else None


def report_messages(prompt: dict, trade_content: str, amount: int | None) -> list[dict]:
    """The context of the report step: the released conversation, Alpha's trade turn as it was shown (its visible text, the
    thinking dropped as the chat template drops it) and the environment's confirmation with the amount bought."""
    trigger = prompt["deception_trigger"].replace("{amount}", str(amount if amount is not None else 2000))
    return prompt["messages"] + [{"role": "assistant", "content": trade_content}, {"role": "user", "content": trigger}]


def report_body(content: str | None) -> str | None:
    """The text Alpha sends to Amy, or None when the reply is not a send_message to her."""
    match = REPORT_BODY.search(content or "")
    return match.group(1).strip() if match else None


def info_field(report: str | None) -> str | None:
    match = INFO_FIELD.search(report or "")
    return match.group(1).strip() if match else None


def make_turn(messages: list[dict], meta: list[dict], response: dict, sample: int, phase: str, **extra) -> dict:
    """A record in the shape detector.render.render_turn reads (and scripts/replay.py replays)."""
    return {"agent": AGENT, "episode_index": sample, "phase": phase, "messages": messages, "message_meta": meta, "tools": None,
            "response": {"content": response.get("content") or "", "reasoning": response.get("reasoning") or "",
                         "tool_calls": [], "finish_reason": response.get("finish_reason"), "usage": response.get("usage"),
                         "model": response.get("model")}, **extra}


def peer_texts(messages: list[dict], meta: list[dict]) -> list[str]:
    """The peer-written text of every span, in order."""
    out = []
    for message, tag in zip(messages, meta):
        if tag["provenance"] == "peer":
            out.append(message["content"])
        for a, b in tag.get("peer_spans") or []:
            out.append(message["content"][a:b])
    return out
