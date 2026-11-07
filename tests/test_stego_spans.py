"""Stego span finding on the released runs. Skipped when the dataset or the NARCBench checkout is not on this machine."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens")]
os.environ.setdefault("NARCBENCH_DIR", str(ROOT/"data/narcbench-code"))
from detector.stego_spans import coordination_messages, private_spans  # noqa: E402

STEGO = ROOT/"data/narcbench/scenarios/qwen3_32b/stego"
pytestmark = pytest.mark.skipif(not STEGO.exists() or not (ROOT/"data/narcbench-code/generation/stego.py").exists(), reason="data absent")


@pytest.fixture(scope="module")
def tails():
    import re
    text = (ROOT/"data/narcbench-code/generation/stego.py").read_text()
    namespace = {}
    for name in ("COUNTER_TABLE_TALK_TEMPLATE", "BP_TABLE_TALK_TEMPLATE"):
        block = re.search(rf"^{name} = \((.*?)^\)", text, flags=re.S | re.M).group(1)
        namespace[name] = eval("(" + block + ")")
    cut = lambda template, start: template[template.rindex(start):template.index("{base_prompt}")]
    return cut(namespace["BP_TABLE_TALK_TEMPLATE"], "Do not reference"), cut(namespace["COUNTER_TABLE_TALK_TEMPLATE"], "The pit boss")


def test_every_colluder_turn(tails):
    receiver_tail, sender_tail = tails
    runs = sorted(p for p in STEGO.iterdir() if p.name.endswith("__collusion"))
    assert len(runs) == 20
    turns = 0
    for run in runs:
        config = json.loads((run/"run_config.json").read_text())
        counter, big_player = config["coalition"]
        messages = coordination_messages(run)
        assert len(messages) == 2
        for p in json.loads((run/"agent_prompts.json").read_text()):
            if p["channel"] != "public" or p["role"] != "colluder":
                continue
            receiver = p["agent_name"] == big_player
            spans = private_spans(p["user_prompt"], counter if receiver else big_player, receiver_tail if receiver else sender_tail)
            block = p["user_prompt"][spans[0][0]:spans[0][1]]
            kept = p["user_prompt"]
            for start, end in reversed(spans):
                kept = kept[:start] + kept[end:]
            assert "PRIVATE (only" not in kept
            turns += 1
            assert all(m.split("]: ", 1)[1][:80] in block for m in messages if p["phase"] == "discussion" and "]: " in m) or p["phase"] != "discussion"
    assert turns == 280 + 40
