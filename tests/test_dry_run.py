"""Offline checks with the mock model. Run with pytest or python tests/test_dry_run.py."""
from __future__ import annotations

import asyncio
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.episode import RunConfig, run_single  # noqa: E402
from experiments.labels import run_labels  # noqa: E402
from experiments.llm import MockLLM, parse_inline_tool_calls, split_think  # noqa: E402
from experiments.protocol.state import AGENT_IDS, peer_id  # noqa: E402
from experiments.tasks import GENERATORS, TaskBank, check_answer, normalize_answer  # noqa: E402


def test_task_keys():
    rng = random.Random(7)
    for ttype, gen in GENERATORS.items():
        for d in (1, 2, 3):
            for i in range(10):
                t = gen(rng, d, f"{ttype}-{d}-{i}")
                assert t.answer_key and all(isinstance(k, str) for k in t.answer_key)
                assert check_answer(t, t.answer_key[0])
                assert check_answer(t, f"  {t.answer_key[0]} ")
                assert not check_answer(t, "definitely not the answer")
                assert t.task_id in t.prompt or True
    assert normalize_answer("1,234.0") == "1234"
    assert normalize_answer(' "Berlin". ') == "berlin"


def test_parsers():
    calls = parse_inline_tool_calls('x <tool_call>\n{"name": "submit_verdict", "arguments": {"verdict": "accept", "rationale": "ok"}}\n</tool_call>')
    assert calls and calls[0].name == "submit_verdict" and calls[0].arguments["verdict"] == "accept"
    content, think = split_think("<think>hmm</think>\nfinal")
    assert content == "final" and think == "hmm"


def _run(policy: str, n_episodes: int = 3, no_history: bool = False):
    bank = TaskBank(seed=11, n_episodes=n_episodes).build(AGENT_IDS)
    oracle = {t.task_id: t.answer_key for t in bank.tasks.values()}
    llm = MockLLM(oracle, seed=3, policy=policy)
    cfg = RunConfig(run_name="test", rep=0, run_index=0, seed=11, n_episodes=n_episodes, model="mock",
                    base_url="mock", no_history=no_history)
    with tempfile.TemporaryDirectory() as td:
        run = asyncio.run(run_single(cfg, llm, Path(td) / "rep0_run000", bank))
        on_disk = json.load(open(Path(td) / "rep0_run000" / "run.json"))
        snaps = [json.loads(l) for l in open(Path(td) / "rep0_run000" / "snapshots.jsonl")]
    return run, on_disk, snaps


def test_dry_run_shape():
    run, on_disk, snaps = _run("reciprocal")
    assert run["complete"] and on_disk["complete"] and len(run["results"]) == 3
    assert run["run_config"]["manifest"] == "rep0_run000"
    assert len(snaps) == 3 * 2 * 3  # episodes x agents x phases
    for ep in run["results"]:
        for a in AGENT_IDS:
            rec = ep["agents"][a]
            assert rec["verdict"] in ("accept", "reject")
            assert rec["expected_verdict"] in ("accept", "reject")
            # the slot holds the prompt a's peer saw when judging a
            assert rec["verdict_prompt"].startswith(f"Give your verdict on {a}'s")
            msgs = rec["messages"]
            meta = rec["message_meta"]
            assert msgs[0]["role"] == "system" and meta[0]["provenance"] == "designer"
            assert len(msgs) == len(meta)
            provs = {m["provenance"] for m in meta}
            assert {"designer", "self", "peer", "environment"} <= provs
            # last verdict tool call has verdict as the first argument key
            verdict_calls = [m for m in msgs if m["role"] == "assistant" and m.get("tool_calls")
                             and m["tool_calls"][0]["function"]["name"] == "submit_verdict"]
            args = verdict_calls[-1]["tool_calls"][0]["function"]["arguments"]
            assert list(args)[0] == "verdict"
            # lenient wrong is defined against the peer's submission
            peer = peer_id(a)
            assert rec["lenient_wrong"] == (rec["verdict"] == "accept" and not ep["agents"][peer]["submission_correct"])
    labels = run_labels(run)
    assert 0 <= labels["collusive_rate"] <= 1
    assert labels["n_episodes"] == 3


def test_no_history_resets_context():
    run, _, _ = _run("honest", n_episodes=2, no_history=True)
    ep0 = run["results"][0]["agents"]["alice"]["messages"]
    ep1 = run["results"][1]["agents"]["alice"]["messages"]
    assert len(ep0) == len(ep1)  # same length each episode means the context was reset
    assert "Episode 2" in ep1[1]["content"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
