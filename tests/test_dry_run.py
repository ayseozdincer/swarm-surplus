"""Offline checks with the mock model. Run with pytest or python tests/test_dry_run.py."""
from __future__ import annotations

import asyncio
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.episode import SwarmConfig, assignment, blind_plan, run_seed  # noqa: E402
from experiments.labels import collect, run_labels  # noqa: E402
from experiments.llm import MockLLM, parse_inline_tool_calls, split_think  # noqa: E402
from experiments.plants import PlantSpec  # noqa: E402
from experiments.protocol.state import AgentSpec  # noqa: E402
from experiments.tasks import GENERATORS, TaskBank, check_answer, normalize_answer  # noqa: E402

AGENTS = [AgentSpec("alice", "record_extraction"), AgentSpec("bob", "data_search"),
          AgentSpec("carol", "record_extraction"), AgentSpec("dave", "data_search"),
          AgentSpec("erin", "code_analysis"), AgentSpec("frank", "code_analysis")]


def test_task_keys_and_blind():
    rng = random.Random(7)
    for ttype, gen in GENERATORS.items():
        for d in (1, 3, 5):
            for i in range(6):
                t = gen(rng, d, f"{ttype}-{d}-{i}")
                assert check_answer(t, t.answer_key[0]) and not check_answer(t, "definitely not")
                b = gen(rng, d, f"{ttype}-{d}-{i}-blind", blind=True)
                assert b.blind and b.review_prompt and b.blind_fact
                assert b.answer_key[0] in b.blind_fact
                if ttype == "record_extraction":
                    tid = b.meta["target_id"]
                    assert b.prompt.count(tid) == 1 and b.review_prompt.count(tid) == 2
                elif ttype == "data_search":
                    assert b.meta["hidden_rows"] >= 2 and b.prompt.count("\n") == b.review_prompt.count("\n") - b.meta["hidden_rows"]
                else:
                    assert "withheld" in b.prompt and "withheld" not in b.review_prompt
    assert normalize_answer("1,234.0") == "1234"
    assert normalize_answer(' "Berlin". ') == "berlin"


def test_assignment_is_derangement():
    for ep in range(8):
        rev = assignment(AGENTS, ep)
        assert set(rev.values()) == set(rev.keys())
        assert all(a != r for a, r in rev.items())


def test_parsers():
    calls = parse_inline_tool_calls('x <tool_call>\n{"name": "submit_verdict", "arguments": {"verdict": "accept", "rationale": "ok"}}\n</tool_call>')
    assert calls and calls[0].arguments["verdict"] == "accept"
    assert split_think("<think>hmm</think>\nfinal") == ("final", "hmm")


def _cfg(seed: int, n_warmup: int = 1, n_post: int = 0) -> SwarmConfig:
    return SwarmConfig(run_name="test", seed=seed, agents=AGENTS, n_warmup=n_warmup, n_post=n_post,
                       model="mock", base_url="mock")


def _mock(cfg: SwarmConfig, **kw) -> MockLLM:
    bank = TaskBank(cfg.seed, cfg.n_episodes).build(cfg.agents, blind_plan(cfg))
    return MockLLM({t.task_id: {"key": t.answer_key, "blind": t.blind} for t in bank.tasks.values()}, seed=cfg.seed, **kw)


def test_branching_and_plants():
    cfg = _cfg(11, n_warmup=1, n_post=1)
    variants = [PlantSpec("none"), PlantSpec("benign", 3), PlantSpec("leak"), PlantSpec("proposal", 3)]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "exp"
        llm = _mock(cfg, p_correct=0.3, p_use_leak=1.0, p_comply=1.0, p_judge=1.0)
        out = asyncio.run(run_seed(cfg, llm, root / "seed11", variants, tasks_dir=root / "tasks"))
        assert all(v.get("complete") for v in out.values()), out
        pe = cfg.plant_episode
        runs = {v.name: json.load(open(root / "seed11" / v.name / "run.json")) for v in variants}
        prefix = json.load(open(root / "seed11" / "prefix" / "run.json"))
        assert len(prefix["results"]) == 1 and all(len(r["results"]) == 3 for r in runs.values())
        # warm-up episodes are identical across variants (branched from one prefix)
        first = {n: r["results"][0]["agents"]["alice"]["submission"] for n, r in runs.items()}
        assert len({json.dumps(v) for v in first.values()}) == 1
        # plants
        assert runs["none"]["plant"]["entries"] == []
        assert len(runs["benign_k3"]["plant"]["entries"]) == 3
        assert [e["kind"] for e in runs["proposal_k3"]["plant"]["entries"]] == ["proposal", "agree", "agree"]
        assert {e["author"] for e in runs["benign_k3"]["plant"]["entries"]} == {e["author"] for e in runs["proposal_k3"]["plant"]["entries"]}
        leak = runs["leak"]
        targets = leak["plant"]["targets"]
        assert len(targets) == 1 and len(leak["plant"]["entries"]) == 1
        t = targets[0]
        rec = leak["results"][pe]["agents"][t]
        assert rec["blind"] and rec["leak_available"] and rec["leak_used"] and rec["submission_correct"]
        author = leak["plant"]["entries"][0]["author"]
        assert leak["results"][pe]["agents"][author]["instructed"]
        assert author != t and not rec["instructed"]
        # in the no-plant branch the same blind agent cannot answer
        assert not runs["none"]["results"][pe]["agents"][t]["submission_correct"]
        assert not runs["none"]["results"][pe]["agents"][t]["leak_available"]
        # proposal: every free reviewer of a wrong submission who saw the plant complied (p_comply=1)
        ep = runs["proposal_k3"]["results"][pe]["agents"]
        for a, r in ep.items():
            if not r["instructed"] and not r["reviewed_submission_correct"]:
                assert r["exposed_planted"] and r["lenient_wrong"], (a, r["verdict"], r["exposed_planted"])
        # benign: honest judging (p_judge=1) gives no lenient verdicts
        ep = runs["benign_k3"]["results"][pe]["agents"]
        assert not any(r["lenient_wrong"] for r in ep.values())
        # provenance and quarantine
        for name, r in runs.items():
            for rec in r["results"][-1]["agents"].values():
                assert len(rec["messages"]) == len(rec["message_meta"])
                assert rec["message_meta"][0]["provenance"] == "designer"
                assert not any("Output only the note text" in (m.get("content") or "") for m in rec["messages"])
                if r["plant"]["entries"]:
                    peers = [m for m in rec["message_meta"] if m["provenance"] == "peer"]
                    assert peers, "notes digest should have been delivered"
        for line in open(root / "seed11" / "leak" / "snapshots.jsonl"):
            d = json.loads(line)
            if d["phase"].startswith("plant_"):
                assert d["quarantined"] and d["message_meta"] is None
        labels = run_labels(leak)
        assert labels["measured"]["leak_used_free"] == 1
        agg = collect(root)
        assert agg["variants"]["leak"]["leak_used_free"]["rate"] == 1.0
        assert agg["variants"]["none"]["blind_correct_free"]["rate"] == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
