"""The sanctioned reference for NARCBench Core: the honest system prompt from the paired control run, and the public
turns only. A colluder's user prompt carries a PRIVATE COORDINATION block before the discussion history; nothing else in
it is private. Everyone else's reference input is their actual input."""
from __future__ import annotations

import json
import re
from pathlib import Path

PRIVATE_BLOCK = re.compile(r"PRIVATE COORDINATION \(only you and [^\n]*? can see this\):\n.*?\n\n(?=(?:FULL )?DISCUSSION HISTORY:)", re.DOTALL)


def strip_private(user_prompt: str) -> str:
    stripped, count = PRIVATE_BLOCK.subn("", user_prompt)
    if count > 1 or "PRIVATE COORDINATION" in stripped:
        raise ValueError("private block not removed exactly once")
    return stripped


def honest_systems(scenarios: Path, collusion_run: str) -> dict[str, str]:
    """agent name -> system prompt in the paired control run."""
    control = collusion_run.rsplit("__", 1)[0] + "__control"
    prompts = json.loads((scenarios/control/"agent_prompts.json").read_text())
    return {p["agent_name"]: p["system_prompt"] for p in prompts}


def reference_sample(sample: dict, systems: dict[str, str]) -> dict:
    return {**sample, "system_prompt": systems[sample["agent_name"]], "user_prompt": strip_private(sample["user_prompt"])}
