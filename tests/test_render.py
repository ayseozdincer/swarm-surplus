"""Rendering checks on the pilot run. Skipped when the run or the Qwen3 tokenizer is not on this machine."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detector.render import compacted_positions, find_turn, load_turns, render_turn  # noqa: E402

RUN = ROOT / "runs/pilot/seed1000"


def test_compacted_positions():
    assert compacted_positions([1, 1, 0, 0, 1, 1]) == [0, 1, 2, 3, 2, 3]
    assert compacted_positions([1, 0, 1, 0, 1]) == [0, 1, 1, 3, 2]
    assert compacted_positions([1, 1, 1]) == [0, 1, 2]


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    except OSError:
        pytest.skip("tokenizer unavailable")


@pytest.mark.skipif(not RUN.exists(), reason="pilot run absent")
def test_leak_turn(tokenizer):
    turn = find_turn(load_turns(RUN / "leak"), "carol", 2, "solve")
    r = render_turn(tokenizer, turn)
    decode = lambda i: tokenizer.decode(r["ids"][i])
    assert decode(r["positions"]["first_generated"]) == "<think>"
    assert decode(r["positions"]["prompt_last"]) == "\n"
    assert decode(r["positions"]["prompt_last"] - 1) == "assistant"
    assert decode(r["positions"]["after_think"] - 1) == "</think>"
    assert "".join(decode(r["positions"][k]) for k in ("answer_0", "answer_1")) == "73"
    assert decode(r["positions"]["mention_0"]) == "7"
    assert r["positions"]["first_generated"] < r["positions"]["mention_pre"] < r["positions"]["after_think"]
    assert len(r["peer_token_spans"]) == 1
    a, b = r["peer_token_spans"][0]
    masked = tokenizer.decode(r["ids"][a:b])
    assert masked.startswith("<|im_start|>user\nNew entries in the team notes file.") and "73" in masked
    assert masked.endswith("<|im_end|>\n")
    assert sum(1 for m in r["mask"] if m == 0) == b - a


@pytest.mark.skipif(not RUN.exists(), reason="pilot run absent")
@pytest.mark.parametrize("variant", ["leak", "benign_k1", "proposal_k1"])
def test_compacted_prompt_lines_up_with_none(tokenizer, variant):
    plain = render_turn(tokenizer, find_turn(load_turns(RUN/"none"), "carol", 2, "solve"))
    r = render_turn(tokenizer, find_turn(load_turns(RUN/variant), "carol", 2, "solve"))
    position = compacted_positions(r["mask"])
    kept = [(t, p) for t, p, m in zip(r["ids"][:r["prompt_tokens"]], position, r["mask"]) if m]
    assert kept == list(zip(plain["ids"][:plain["prompt_tokens"]], range(plain["prompt_tokens"])))


@pytest.mark.skipif(not RUN.exists(), reason="pilot run absent")
def test_matched_turn_has_no_peer_span(tokenizer):
    r = render_turn(tokenizer, find_turn(load_turns(RUN / "none"), "carol", 2, "solve"))
    assert r["peer_token_spans"] == [] and all(r["mask"])


@pytest.mark.skipif(not RUN.exists(), reason="pilot run absent")
@pytest.mark.parametrize("variant", ["none", "leak", "benign_k1", "proposal_k1"])
def test_token_counts_match_what_was_served(tokenizer, variant):
    for turn in load_turns(RUN / variant):
        r = render_turn(tokenizer, turn)
        served = turn["response"]["usage"]["prompt_tokens"]
        assert r["prompt_tokens"] == served, (turn["agent"], turn["episode_index"], turn["phase"], r["prompt_tokens"], served)
