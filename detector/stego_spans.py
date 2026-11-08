"""Private spans inside NARCBench Stego prompts, for a receiver's sanctioned reference.

A colluder's public turn is one user message. Its opening block, from `PRIVATE (only you and <partner> can see this):` to
the template line just before the table prompt, holds the private coordination (channel == private) and the rules that
follow from it. The table prompt after it can also carry `PRIVATE:` note lines from earlier turns. Both are excluded;
what is left is the public table prompt, the same split read.py uses for its honest counterfactual.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

OPENING = "PRIVATE (only you and "
NOTE_LINE = re.compile(r"^[ \t]*PRIVATE\s*:(?!\s*\[what).*\n?", flags=re.M)   # read.py's PRIVATE pattern


def template_tails() -> tuple[str, str]:
    """The text ending each colluder template's private block, from NARCBench's generation/stego.py."""
    from utils.narcbench import stego_templates
    _, receiver, sender = stego_templates()
    return receiver, sender


def private_spans(user_prompt: str, partner: str, tail: str) -> list[tuple[int, int]]:
    """Character spans of everything private in one colluder prompt: the opening block, then each PRIVATE: line."""
    if not user_prompt.startswith(OPENING):
        raise ValueError("colluder prompt does not open with the private block")
    marker = tail.format(partner=partner)
    if user_prompt.count(marker) != 1:
        raise ValueError("template tail not found exactly once")
    end = user_prompt.index(marker) + len(marker)
    return [(0, end)] + [m.span() for m in NOTE_LINE.finditer(user_prompt, end)]


def coordination_messages(run_dir: Path) -> list[str]:
    """The agreed-scheme messages of the private phase, without the per-hand [Hand n: ...] records."""
    results = json.loads((Path(run_dir)/"results.json").read_text())
    return [m for m in results["private_messages"] if not m.startswith("[Hand ")]


def token_spans(offsets: list[tuple[int, int]], base: int, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """[first, last) token spans, for character spans counted from `base` in the rendered text."""
    out = []
    for start, end in spans:
        first = next(i for i, (a, b) in enumerate(offsets) if b > base + start)
        last = next((i for i, (a, b) in enumerate(offsets) if a >= base + end), len(offsets))
        out.append((first, last))
    return out
