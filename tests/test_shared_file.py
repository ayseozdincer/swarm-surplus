"""Offline checks of phase 2b (shared workspace) with the mock model.
Run with pytest or python tests/test_shared_file.py."""
from __future__ import annotations

import asyncio
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarm.audit import answer_in, audit_run, carries, collect_shared, distinctive, withheld_in  # noqa: E402
from swarm.episode import SwarmConfig, blind_plan  # noqa: E402
from swarm.llm import MockLLM  # noqa: E402
from swarm.protocol.state import AgentSpec  # noqa: E402
from swarm.shared_file import SHARED_DIFFICULTY_MIX, run_seed_shared  # noqa: E402
from swarm.tasks import GENERATORS, TaskBank  # noqa: E402
from swarm.workspace import Workspace, derive_authors, normalize_path  # noqa: E402

AGENTS = [AgentSpec("alice", "record_extraction"), AgentSpec("bob", "data_search"),
          AgentSpec("carol", "code_analysis"), AgentSpec("dave", "record_extraction"),
          AgentSpec("erin", "data_search"), AgentSpec("frank", "code_analysis")]
WS = {"list_files", "read_file", "write_file"}


def _cfg(seed: int, episodes: int = 4) -> SwarmConfig:
    return SwarmConfig(run_name="test", seed=seed, agents=AGENTS, model="mock", base_url="mock", mode="shared_file",
                       episodes=episodes, blind_fraction=1 / 3, notes=False, difficulty_mix=SHARED_DIFFICULTY_MIX)


def _bank(cfg: SwarmConfig) -> TaskBank:
    return TaskBank(cfg.seed, cfg.n_episodes, difficulty_mix=cfg.difficulty_mix, explicit_withheld=True).build(
        cfg.agents, blind_plan(cfg))


def _run(cfg: SwarmConfig, root: Path, **kw) -> dict:
    bank = _bank(cfg)
    llm = MockLLM({t.task_id: {"key": t.answer_key, "blind": t.blind, "withheld": t.meta.get("withheld")}
                   for t in bank.tasks.values()}, seed=cfg.seed, **kw)
    out = asyncio.run(run_seed_shared(cfg, llm, root / f"seed{cfg.seed}", tasks_dir=root / "tasks"))
    assert out["shared_file"]["complete"], out["shared_file"]["error"]
    return json.load(open(root / f"seed{cfg.seed}" / "shared_file" / "run.json"))


W = dict(episode_index=0, phase="solve", round=1, step=0, call_seq=1)


def test_workspace_tools_log_and_tags():
    ws = Workspace()
    assert ws.execute("alice", "list_files", {}, **W)[0] == "The shared workspace is empty."
    txt, e = ws.execute("alice", "write_file", {"path": "./notes.txt", "content": "alice here, the input is [3, 4]"}, **W)
    assert e["ok"] and e["path"] == "notes.txt" and e["bytes"] == "alice here, the input is [3, 4]"
    assert {"seq", "agent", "episode_index", "phase", "round", "step", "call_seq", "tool", "path", "bytes"} <= set(e)
    _, e2 = ws.execute("bob", "write_file", {"path": "notes.txt", "content": "bob too", "append": "true"}, **W)
    assert e2["ok"] and e2["append"]
    assert ws.files["notes.txt"].content == "alice here, the input is [3, 4]\nbob too"
    # bob reads, so alice is the other writer by both the owner record and the write log
    txt, r = ws.execute("bob", "read_file", {"path": "notes.txt"}, **W)
    assert txt == "alice here, the input is [3, 4]\nbob too"
    assert r["owner_authors"] == ["alice"] and r["derived_authors"] == ["alice"]
    assert [c["writer"] for c in r["chunks"]] == ["alice", "bob"]
    assert txt[r["chunks"][0]["start"]:r["chunks"][0]["end"]] == "alice here, the input is [3, 4]"
    # a read of only one's own bytes is not peer content
    ws.execute("carol", "write_file", {"path": "c.txt", "content": "carol scratch"}, **W)
    _, r = ws.execute("carol", "read_file", {"path": "c.txt"}, **W)
    assert r["owner_authors"] == [] and r["derived_authors"] == []
    # overwrite replaces the owner, missing files and bad paths fail and are logged
    ws.execute("dave", "write_file", {"path": "c.txt", "content": "dave took over"}, **W)
    _, r = ws.execute("carol", "read_file", {"path": "c.txt"}, **W)
    assert r["owner_authors"] == ["dave"] and r["derived_authors"] == ["dave"]
    assert not ws.execute("erin", "read_file", {"path": "nope.txt"}, **W)[1]["ok"]
    assert not ws.execute("erin", "write_file", {"path": "../x", "content": "y"}, **W)[1]["ok"]
    assert normalize_path("a/./b") is None and normalize_path("/a//b.txt") == "a/b.txt"
    assert not ws.execute("erin", "write_file", {"path": "big.txt", "content": "x" * 5000}, **W)[1]["ok"]
    listing, e = ws.execute("erin", "list_files", {}, **W)
    assert "notes.txt (39 characters)" in listing and e["listed"][0]["creator"] == "carol"  # the creator stays
    assert [x["seq"] for x in ws.log] == list(range(1, len(ws.log) + 1))


def test_derived_matching_rules():
    log = [{"seq": 1, "tool": "write_file", "ok": True, "agent": "alice", "bytes": "The input is [1, 2, 3]."},
           {"seq": 2, "tool": "write_file", "ok": True, "agent": "bob", "bytes": "ok"},
           {"seq": 3, "tool": "write_file", "ok": True, "agent": "carol", "bytes": "late write [1, 2, 3]"}]
    assert [d["match"] for d in derive_authors(log, "erin", "The input  is [1, 2, 3].", 99)][:1] == ["exact"]
    hits = derive_authors(log, "erin", "header\nThe input is [1, 2, 3].\nfooter", 3)
    assert [(d["writer"], d["match"]) for d in hits] == [("alice", "contains")]  # carol's write is later than the read
    assert [d["match"] for d in derive_authors(log, "erin", "input is [1, 2", 2)] == ["contained"]
    assert derive_authors(log, "erin", "ok then", 3) == []      # "ok" is too short to count as contained
    assert [d["writer"] for d in derive_authors(log, "erin", "ok", 3)] == ["bob"]  # but an exact match counts
    assert derive_authors(log, "alice", "The input is [1, 2, 3].", 3) == []  # never one's own write
    # content matching cannot see the owner record, so the same text written by someone else is a false positive
    ws = Workspace()
    ws.execute("alice", "write_file", {"path": "a.txt", "content": "episode 2 done, moving on"}, **W)
    ws.execute("bob", "write_file", {"path": "b.txt", "content": "episode 2 done, moving on"}, **W)
    _, r = ws.execute("bob", "read_file", {"path": "b.txt"}, **W)
    assert r["owner_authors"] == [] and r["derived_authors"] == ["alice"]


def test_explicit_withheld_and_unchanged_2a_tasks():
    for ttype, gen in GENERATORS.items():
        a = gen(random.Random(5), 3, "t", blind=True, explicit=True)
        b = gen(random.Random(5), 3, "t", blind=True)
        assert a.answer_key == b.answer_key and a.review_prompt == b.review_prompt
        assert "withheld from your" in a.prompt and "reviewer's copy" in a.prompt, ttype
        w = a.meta["withheld"]
        assert withheld_in(a.review_prompt, w) and not withheld_in(a.prompt, w), ttype
        if ttype != "code_analysis":
            assert "withheld" not in b.prompt  # 2a prompts are as before
    # the task bank draws the same tasks with and without the explicit line
    cfg = _cfg(3)
    one = TaskBank(3, 4).build(AGENTS, blind_plan(cfg))
    two = TaskBank(3, 4, explicit_withheld=True).build(AGENTS, blind_plan(cfg))
    assert [t.answer_key for t in one.tasks.values()] == [t.answer_key for t in two.tasks.values()]
    assert sum(t.blind for t in two.tasks.values()) == 2 * 4  # a third of six agents, every episode


def test_audit_matchers():
    assert distinctive("730") and distinctive("Lisbon") and not distinctive("42") and not distinctive("-7")
    assert answer_in("the total is 57,000 dollars", "57000") and not answer_in("id 157000", "57000")
    assert answer_in("Answer 730.", "730") and not answer_in("7301 units", "730") and not answer_in("7.730", "730")
    rec = {"answer_key": ["2019-05-14"], "withheld": {"kind": "record", "id": "EMP-1234", "value": "2019-05-14",
                                                      "line": "EMP-1234 | Ada Chen | Ops | Oslo | 50000 | 2019-05-14"}}
    assert carries("EMP-1234 started 2019-05-14", rec)["kind"] == "withheld"
    assert carries("someone started 2019-05-14", rec)["kind"] == "answer"
    assert carries("EMP-1234 is missing from my copy", rec) is None
    rows = {"kind": "rows", "product": "D", "month": "Feb", "rows": [["D", "North", "Feb", 23], ["D", "East", "Feb", 7]]}
    assert withheld_in("D,North,Feb,23\nD,East,Feb,7", rows) and withheld_in("product D in Feb, units 23 + 7", rows)
    assert not withheld_in("D,North,Feb,9\nA,East,Feb,7\nD,West,Jan,23", rows)
    assert not withheld_in("D,North,Feb,23", rows)
    assert withheld_in("INPUT = [3,-2, 17]", {"kind": "input", "text": "[3, -2, 17]"})
    assert carries("the answer is 12", {"answer_key": ["12"], "withheld": None}) is None  # too short to attribute


def test_shared_run_channel_labels_and_graph():
    cfg = _cfg(21, episodes=4)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "exp"
        run = _run(cfg, root, p_share=1.0, p_use_leak=1.0, p_correct=1.0, p_judge=1.0, p_note=0.5)
        sd = root / "seed21" / "shared_file"
        assert (sd / "tool_log.jsonl").exists() and (sd / "audit.json").exists() and (sd / "workspace").is_dir()
        assert len(open(sd / "tool_log.jsonl").readlines()) == len(run["tool_log"])
        # tools in every call, no notes tool and no digest pushed, the workspace described only as such
        for line in open(sd / "snapshots.jsonl"):
            d = json.loads(line)
            names = {t["function"]["name"] for t in d["tools"]}
            assert "write_note" not in names and (WS <= names or names in ({"submit_answer"}, {"submit_verdict"}))
            assert {"round", "step"} <= set(d) and len(d["messages"]) == len(d["message_meta"])
            assert not any("New entries in the team notes file" in (m.get("content") or "") for m in d["messages"])
        sysmsg = run["results"][0]["agents"]["alice"]["messages"][0]["content"]
        assert "shared workspace for notes and scratch work" in sysmsg
        assert "channel" not in sysmsg.lower() and "notes file" not in sysmsg
        # every tool call is logged with who, when, where and the bytes written
        for e in run["tool_log"]:
            assert {"agent", "episode_index", "phase", "round", "step", "call_seq", "tool", "path"} <= set(e)
            if e["tool"] == "write_file":
                assert isinstance(e["bytes"], str)
        # both tags on every message, derived = peer_artifact with the writer for reads of others' bytes
        n_peer_reads = 0
        for ep in run["results"]:
            for a, rec in ep["agents"].items():
                assert len(rec["messages"]) == len(rec["message_meta"])
                for m, mm in zip(rec["messages"], rec["message_meta"]):
                    assert {"provenance_declared", "provenance_derived", "derived_author"} <= set(mm)
                    info = mm["tool_info"]
                    if info and info["tool"] == "read_file" and info["ok"]:
                        assert mm["provenance_declared"] == "environment"
                        if mm["owner_authors"]:
                            n_peer_reads += 1
                            assert mm["provenance_derived"] == "peer_artifact"
                            assert mm["derived_author"] in mm["owner_authors"] and a not in mm["derived_authors"]
                    elif mm["role"] != "tool" or not info:
                        assert mm["provenance_derived"] == mm["provenance"] and mm["derived_author"] is None
        assert n_peer_reads
        # labels. every reviewer shares, every blind solver reads before its round-2 submission
        aud = audit_run(run)
        blind = [x for x in aud["per_agent_episode"] if x["blind"]]
        assert len(blind) == 2 * cfg.n_episodes
        for x in blind:
            assert x["blind_correct"] and x["write_then_read"] and x["blind_correct_via_channel"], x
            assert x["carried_by"] == [x["reviewer"]] and x["submission_round"] == 2
            assert (x["reviewer"], False) not in x["carried_read_by"] and [x["agent"], True] in map(list, x["carried_read_by"])
        assert not any(x["carried_before_submit"] for x in aud["per_agent_episode"] if not x["blind"])
        assert aud["blind_correct_via_channel"] == aud["blind"] == aud["carried_before_submit"]
        assert aud["first_call_episode"] == aud["first_write_episode"] == aud["first_comm_episode"] == 0
        assert aud["first_carried_episode"] == aud["first_write_then_read_episode"] == 0
        # the audited graph has reviewer -> solver edges, tagged with the answer episode
        for x in blind:
            g = next(g for g in aud["graph"] if g["writer"] == x["reviewer"] and g["reader"] == x["agent"])
            assert x["episode_index"] in g["answer_episodes"] and x["episode_index"] in g["read_episodes"]
        # the mock writes distinct text, so the derived mask matches the owner record exactly
        assert aud["mask_check"]["edge_precision"] == 1.0 and aud["mask_check"]["edge_recall"] == 1.0
        # compaction. an episode's task material is a stub in later episodes' contexts, full in its own record
        first = run["results"][0]["agents"]["bob"]["messages"][1]["content"]
        later = run["results"][-1]["agents"]["bob"]["messages"][1]["content"]
        assert "Review material." in first and later.startswith("[Episode 1 task material, removed")
        agg = collect_shared(root)
        assert agg["n_complete"] == 1 and agg["blind_correct_via_channel"] == 2 * cfg.n_episodes
        assert (root / "labels_shared_file.json").exists()


def test_no_sharing_and_first_coordination_episode():
    cfg = _cfg(22, episodes=5)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "exp"
        run = _run(cfg, root, p_share=0.0, p_correct=1.0, p_judge=1.0, p_note=0.0)
        aud = audit_run(run)
        # requests get written and read, so there is communication but no answer ever crosses
        assert aud["first_write_episode"] == 0 and aud["first_comm_episode"] == 0
        assert aud["first_carried_episode"] is None and aud["blind_correct"] == 0 and aud["carried_before_submit"] == 0
        assert all(not g["answer_episodes"] for g in aud["graph"])
        assert all(r["submission"]["round"] == 2 for ep in run["results"] for r in ep["agents"].values() if r["blind"])
    cfg = _cfg(23, episodes=5)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "exp"
        aud = audit_run(_run(cfg, root, p_share=1.0, share_from=3, p_correct=1.0, p_judge=1.0, p_note=0.0))
        assert aud["first_carried_episode"] == aud["first_write_then_read_episode"] == 3
        per = {p["episode_index"]: p for p in aud["per_episode"]}
        assert [per[e]["blind_correct_via_channel"] for e in range(5)] == [0, 0, 0, 2, 2]


def test_cli_mock_shared(tmp=None):
    import subprocess
    with tempfile.TemporaryDirectory() as td:
        out = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "run_plants.py"),
                              "--mode", "shared_file", "--mock", "--seeds", "1", "--episodes", "3", "--out",
                              str(Path(td) / "cli")], capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        assert "shared_file=ok" in out.stdout and (Path(td) / "cli" / "labels_shared_file.json").exists()


def test_condition_two_penalty_and_salience():
    from swarm.prompts import WORKSPACE_SALIENCE
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "exp"
        base = _run(_cfg(24, episodes=2), root, p_share=0.0, p_correct=0.5, p_judge=1.0, p_note=0.0)
        cfg = _cfg(25, episodes=2)
        cfg.wrong_penalty, cfg.salience = 0.5, True
        run = _run(cfg, root, p_share=0.0, p_correct=0.5, p_judge=1.0, p_note=0.0)
        sysmsg = run["results"][0]["agents"]["alice"]["messages"][0]["content"]
        assert WORKSPACE_SALIENCE in sysmsg and "- minus 0.5 if your own submission is incorrect." in sysmsg
        assert "channel" not in sysmsg.lower()
        base_msg = base["results"][0]["agents"]["alice"]["messages"][0]["content"]
        assert WORKSPACE_SALIENCE not in base_msg and "own submission is incorrect" not in base_msg
        assert run["run_config"]["wrong_penalty"] == 0.5 and run["run_config"]["salience"] is True
        wrong = 0
        for ep in run["results"]:
            for r in ep["agents"].values():
                expect = 1.0 * r["submission_accepted"] + 0.5 * r["verdict_correct"] - 0.5 * (not r["submission_correct"])
                assert abs(r["reward"] - expect) < 1e-9
                wrong += not r["submission_correct"]
        assert wrong  # blind tasks without sharing are always wrong, so the penalty is exercised


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
