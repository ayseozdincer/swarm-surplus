"""Render one agent turn for teacher-forced replay: token ids, the peer spans to mask, the read positions.

A turn is one record of snapshots.jsonl, the context as served plus the model's response. The thinking goes back
in through `reasoning_content`, which Qwen3's template writes as <think>...</think> on the last assistant message.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ANSWER_TOKENS_KEPT = 8                  # a long answer is read at its first tokens only
PEER = "peer"
# A reasoning that talks about the shared notes file, as opposed to the `note` field of a submission.
NOTE_CITATION = re.compile(r"team(?:'s|’s)?\s+(?:shared\s+)?notes?\b|notes?\s+file|shared notes?|trust(?:ing)? each other"
                           r"|default(?:ing)? to (?:approv|accept)|mutual support|collective buy-in", re.I)


def load_turns(run_dir: Path) -> list[dict]:
    """Every model call of a variant run that an agent answered in its own context."""
    rows = [json.loads(line) for line in open(Path(run_dir) / "snapshots.jsonl", encoding="utf-8")]
    return [r for r in rows if r.get("message_meta") and not r.get("quarantined")]


def find_turn(turns: list[dict], agent: str, episode: int, phase: str) -> dict:
    hits = [t for t in turns if (t["agent"], t["episode_index"], t["phase"]) == (agent, episode, phase)]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} snapshots for {(agent, episode, phase)}")
    return hits[0]


def assistant_message(response: dict) -> dict:
    message = {"role": "assistant", "content": response.get("content") or "",
               "reasoning_content": response.get("reasoning") or ""}
    if response.get("tool_calls"):
        message["tool_calls"] = response["tool_calls"]
    return message


def peer_char_spans(tokenizer, messages: list[dict], meta: list[dict], tools, full_text: str) -> list[tuple[int, int]]:
    """Character span of every message tagged `peer`, wrapper tokens included. Each comes from rendering the list up
    to and including that message and taking the last block. The list then ends on a user message, so no earlier
    assistant message picks up an empty think block and the render is a true prefix of the full text."""
    spans = []
    for i, tag in enumerate(meta):
        if tag["provenance"] != PEER:
            continue
        if messages[i]["role"] != "user":
            raise ValueError(f"peer message {i} has role {messages[i]['role']}")
        upto = tokenizer.apply_chat_template(messages[:i + 1], tools=tools, tokenize=False, add_generation_prompt=False)
        start = upto.rfind("<|im_start|>user\n")
        if start < 0 or not full_text.startswith(upto):
            raise ValueError(f"peer message {i} does not render as a prefix of the full text")
        spans.append((start, len(upto)))
    return spans


def token_span(offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    """[first, last) token indices covering characters [start, end)."""
    first = next(i for i, (a, b) in enumerate(offsets) if b > start)
    last = next((i for i, (a, b) in enumerate(offsets) if a >= end), len(offsets))
    return first, last


def token_at(offsets: list[tuple[int, int]], char: int) -> int:
    return next(i for i, (a, b) in enumerate(offsets) if a <= char < b)


def tool_value_span(full_text: str, reply_start: int, key: str) -> tuple[int, int] | None:
    """Character span of the string value of `key` in the last tool call of the reply."""
    call = full_text.rfind("<tool_call>", reply_start)
    if call < 0:
        return None
    match = re.compile(rf'"{key}":\s*"').search(full_text, call)
    if match is None:
        return None
    end = match.end()
    while full_text[end] != '"' or full_text[end - 1] == "\\":
        end += 1
    return match.end(), end


def render_turn(tokenizer, turn: dict) -> dict:
    """ids, mask (1 keeps a token, 0 is a peer token) and the named read positions of one turn."""
    messages, meta, tools = turn["messages"], turn["message_meta"], turn["tools"]
    prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [assistant_message(turn["response"])], tools=tools, tokenize=False,
                                         add_generation_prompt=False)
    if not full.startswith(prompt):
        raise ValueError("generation prompt is not a prefix of the full render")
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], [tuple(o) for o in encoded["offset_mapping"]]

    mask = [1]*len(ids)
    char_spans = peer_char_spans(tokenizer, messages, meta, tools, full)
    token_spans = [token_span(offsets, a, b) for a, b in char_spans]
    for a, b in token_spans:
        mask[a:b] = [0]*(b - a)

    reply = len(prompt)
    positions = {"prompt_last": token_at(offsets, reply - 1), "first_generated": token_at(offsets, reply)}
    think_close = full.find("</think>", reply)
    if full.startswith("<think>\n", reply):
        positions["first_thought"] = token_at(offsets, reply + len("<think>\n"))
    if think_close >= 0:
        positions["after_think"] = token_at(offsets, think_close + len("</think>"))
    for key in ("answer", "verdict"):
        span = tool_value_span(full, reply, key)
        if span and span[1] > span[0]:
            first, last = token_span(offsets, *span)
            positions[f"{key}_pre"] = first - 1
            for n, index in enumerate(range(first, min(last, first + ANSWER_TOKENS_KEPT))):
                positions[f"{key}_{n}"] = index
    # Where the reasoning first writes the submitted answer. Before this token no self-written text carries it, so if it
    # came from a peer message the reference has no source for it. Later positions cannot tell the two apart.
    answer = tool_value_span(full, reply, "answer")
    mention = full.find(full[slice(*answer)], reply, think_close) if answer and think_close >= 0 and answer[1] > answer[0] else -1
    if mention >= 0:
        positions["mention_0"] = token_at(offsets, mention)
        positions["mention_pre"] = positions["mention_0"] - 1
    citation = NOTE_CITATION.search(full, reply, think_close) if think_close >= 0 else None
    if citation:
        positions["citation"] = token_at(offsets, citation.start())
    end = full.rfind("<|im_end|>")
    positions["end_of_response"] = token_at(offsets, end - 1)
    return {"text": full, "ids": ids, "mask": mask, "positions": positions, "prompt_chars": reply,
            "peer_token_spans": token_spans, "prompt_tokens": positions["prompt_last"] + 1}
