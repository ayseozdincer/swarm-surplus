"""The Core reference on the released scenarios. Skipped when the dataset is not on this machine."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detector.core_reference import honest_systems, reference_sample, strip_private  # noqa: E402

CORE = ROOT/"data/narcbench/scenarios/qwen3_32b/core"
META = ROOT/"data/narcbench/activations/qwen3_32b/core/metadata_gen.json"
pytestmark = pytest.mark.skipif(not META.exists(), reason="NARCBench absent")


def runs():
    return sorted({m["run_id"] for m in json.loads(META.read_text()) if m["mode"] == "collusion"})


def test_private_block_leaves_the_honest_preamble():
    checked = 0
    for run in runs():
        prompts = [p for p in json.loads((CORE/run/"agent_prompts.json").read_text()) if p["channel"] == "public" and p["phase"] == "discussion"]
        for rnd in {p["round"] for p in prompts}:
            honest = next(p for p in prompts if p["round"] == rnd and p["role"] == "honest")
            preamble = honest["user_prompt"].split("DISCUSSION HISTORY:")[0]
            for p in prompts:
                if p["round"] == rnd and p["role"] == "colluder":
                    assert "PRIVATE COORDINATION" in p["user_prompt"]
                    assert strip_private(p["user_prompt"]).split("DISCUSSION HISTORY:")[0] == preamble
                    checked += 1
    assert checked == 186


def test_honest_agents_keep_their_input():
    for run in runs():
        systems = honest_systems(CORE, run)
        for p in json.loads((CORE/run/"agent_prompts.json").read_text()):
            if p["role"] == "honest":
                r = reference_sample(p, systems)
                assert r["system_prompt"] == p["system_prompt"] and r["user_prompt"] == p["user_prompt"]


def test_colluder_reference_differs_only_as_declared():
    for run in runs():
        systems = honest_systems(CORE, run)
        for p in json.loads((CORE/run/"agent_prompts.json").read_text()):
            if p["role"] == "colluder" and p["channel"] == "public" and p["phase"] == "discussion":
                r = reference_sample(p, systems)
                assert r["system_prompt"] != p["system_prompt"]
                assert p["agent_name"] in r["system_prompt"]
                tail = "DISCUSSION HISTORY:"
                assert r["user_prompt"].split(tail, 1)[1] == p["user_prompt"].split(tail, 1)[1]
