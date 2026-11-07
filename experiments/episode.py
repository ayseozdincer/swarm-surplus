"""The two-agent repeated peer-review loop.

One run is n_episodes episodes between alice and bob. Each episode:
  solve       each agent gets its own task and calls submit_answer
  verdict     each agent gets a review packet for the peer's submission and calls submit_verdict
  outcome     rewards are revealed and each agent writes a private reflection
Every model call is written to snapshots.jsonl exactly as served, and every
message in an agent's context carries a provenance tag.
"""
from __future__ import annotations

import asyncio
import json
import re
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .llm import LLMResponse, new_call_id
from .prompts import (outcome_message, review_packet_designer, review_packet_peer, system_prompt,
                      task_message, verdict_prompt_text)
from .protocol.state import AGENT_IDS, AgentContext, Message, ToolCall, peer_id
from .provenance import SnapshotWriter
from .tasks import Task, TaskBank, check_answer
from .tool_schemas import get_task_tool_schemas, get_tool_schemas


@dataclass
class RunConfig:
    run_name: str
    rep: int
    run_index: int
    seed: int
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.0
    n_episodes: int = 10
    no_history: bool = False
    norm: str = "mild"
    model: str = ""
    base_url: str = ""
    temperature: float = 0.7
    reasoning: bool = True

    @property
    def manifest(self) -> str:
        # NARCBench's extractor parses the repeat out of this string with split("rep")[-1].split("_")[0]
        return f"rep{self.rep}_run{self.run_index:03d}"


def _assistant_message(resp: LLMResponse, tool_calls: list[ToolCall] | None, ep: int, phase: str,
                       synthesized: bool = False) -> Message:
    return Message("assistant", resp.content if resp.content else None, tool_calls=tool_calls or None,
                   provenance="self", episode_index=ep, phase=phase, reasoning=resp.reasoning,
                   synthesized=synthesized)


def _extract_final_answer(content: str) -> str:
    m = re.search(r"(?:final answer|answer)\s*[:.]\s*(.+)", content or "", re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    lines = [ln.strip() for ln in (content or "").strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


class Runner:
    def __init__(self, cfg: RunConfig, llm, out_dir: Path, bank: TaskBank):
        self.cfg = cfg
        self.llm = llm
        self.out_dir = Path(out_dir)
        self.bank = bank
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.snaps = SnapshotWriter(self.out_dir / "snapshots.jsonl")
        self.ctx = {a: AgentContext(a) for a in AGENT_IDS}
        self.call_seq = 0

    # ------------------------------------------------------------------ calls

    async def _call(self, agent: str, ep: int, phase: str, tools: list[dict]) -> LLMResponse:
        c = self.ctx[agent]
        self.call_seq += 1
        seq = self.call_seq
        messages = c.openai_messages(for_api=True)
        resp = await self.llm.chat(messages, tools=tools, seed=self.cfg.seed * 1000 + seq)
        self.snaps.write({
            "run": self.cfg.manifest, "rep": self.cfg.rep, "run_index": self.cfg.run_index, "seq": seq,
            "episode_index": ep, "agent": agent, "phase": phase,
            "llm": self.llm.describe(),
            "messages": c.openai_messages(for_api=False),
            "message_meta": c.meta(),
            "tools": tools,
            "response": resp.to_record(),
        })
        return resp

    # ------------------------------------------------------------------ phases

    async def _solve(self, agent: str, task: Task, ep: int) -> dict:
        peer = peer_id(agent)
        c = self.ctx[agent]
        c.add(Message("user", task_message(ep, self.cfg.n_episodes, task.task_id, task.prompt, peer),
                      provenance="designer", episode_index=ep, phase="solve"))
        tools = get_task_tool_schemas(task.task_type, task.answer_key, peer)
        resp = await self._call(agent, ep, "solve", tools)
        tc = resp.first_tool("submit_answer")
        parsed = tc is not None
        if tc is not None:
            answer = str(tc.arguments.get("answer", "")).strip()
            note = str(tc.arguments.get("note", "")).strip()
            tc.arguments = {"answer": answer, "note": note}
        else:
            answer = _extract_final_answer(resp.content)
            note = (resp.content or "").strip()[:600]
            tc = ToolCall("submit_answer", {"answer": answer, "note": note}, new_call_id())
        c.add(_assistant_message(resp, [tc], ep, "solve", synthesized=not parsed))
        c.add(Message("tool", "Submission recorded.", tool_call_id=tc.id, name="submit_answer",
                      provenance="environment", episode_index=ep, phase="solve"))
        return {"answer": answer, "note": note, "reasoning": resp.reasoning, "content": resp.content,
                "tool_call_parsed": parsed}

    async def _judge(self, agent: str, peer_task: Task, peer_sub: dict, ep: int) -> dict:
        peer = peer_id(agent)
        c = self.ctx[agent]
        c.add(Message("user", review_packet_designer(ep, peer, peer_task.task_id, peer_task.prompt),
                      provenance="designer", episode_index=ep, phase="verdict"))
        c.add(Message("user", review_packet_peer(peer, peer_sub["answer"], peer_sub["note"]),
                      provenance="peer", episode_index=ep, phase="verdict"))
        vp = verdict_prompt_text(peer)
        c.add(Message("user", vp, provenance="designer", episode_index=ep, phase="verdict"))
        tools = get_tool_schemas("verdict", peer)
        resp = await self._call(agent, ep, "verdict", tools)
        tc = resp.first_tool("submit_verdict")
        parsed = tc is not None
        verdict = None
        rationale = ""
        if tc is not None:
            v = str(tc.arguments.get("verdict", "")).strip().lower()
            verdict = v if v in ("accept", "reject") else None
            rationale = str(tc.arguments.get("rationale", "")).strip()
            tc.arguments = {"verdict": v, "rationale": rationale}  # verdict first, as the extractor's regex expects
        else:
            m = re.search(r"\b(accept|reject)\b", (resp.content or "").lower())
            if m:
                verdict = m.group(1)
                rationale = (resp.content or "").strip()[:600]
                tc = ToolCall("submit_verdict", {"verdict": verdict, "rationale": rationale}, new_call_id())
        c.add(_assistant_message(resp, [tc] if tc else None, ep, "verdict", synthesized=(tc is not None and not parsed)))
        if tc is not None:
            c.add(Message("tool", "Verdict recorded.", tool_call_id=tc.id, name="submit_verdict",
                          provenance="environment", episode_index=ep, phase="verdict"))
        return {"verdict": verdict, "rationale": rationale, "reasoning": resp.reasoning, "content": resp.content,
                "verdict_prompt": vp, "tool_call_parsed": parsed}

    async def _reflect(self, agent: str, task: Task, ep: int, text: str) -> dict:
        peer = peer_id(agent)
        c = self.ctx[agent]
        c.add(Message("user", text, provenance="environment", episode_index=ep, phase="reflection"))
        tools = get_task_tool_schemas(task.task_type, task.answer_key, peer)
        resp = await self._call(agent, ep, "reflection", tools)
        reflection = (resp.content or "").strip()
        if not reflection and resp.tool_calls:
            reflection = json.dumps(resp.tool_calls[0].arguments, ensure_ascii=False)
        c.add(_assistant_message(resp, None, ep, "reflection"))
        return {"reflection": reflection, "reasoning": resp.reasoning}

    # ------------------------------------------------------------------ run

    def _run_json(self, results: list[dict], complete: bool, error: str | None) -> dict:
        return {
            "run_config": {
                "manifest": self.cfg.manifest,
                **asdict(self.cfg),
                "llm": self.llm.describe(),
                "agent_ids": list(AGENT_IDS),
                "task_seed": self.bank.seed,
                "harness_version": __version__,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "results": results,
            "complete": complete,
            "error": error,
        }

    def _save(self, results: list[dict], complete: bool, error: str | None) -> dict:
        data = self._run_json(results, complete, error)
        tmp = self.out_dir / "run.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.replace(self.out_dir / "run.json")
        return data

    async def run(self) -> dict:
        cfg = self.cfg
        for a in AGENT_IDS:
            self.ctx[a].add(Message("system", system_prompt(a, peer_id(a), cfg.n_episodes, cfg.alpha, cfg.beta,
                                                            cfg.gamma, cfg.norm),
                                    provenance="designer", phase="system"))
        results: list[dict] = []
        error = None
        try:
            for ep in range(cfg.n_episodes):
                spec = self.bank.episodes[ep]
                tasks: dict[str, Task] = spec["tasks"]
                if cfg.no_history:
                    for a in AGENT_IDS:
                        self.ctx[a].messages = self.ctx[a].messages[:1]
                        self.ctx[a].cumulative_reward = 0.0

                subs = dict(zip(AGENT_IDS, await asyncio.gather(*(self._solve(a, tasks[a], ep) for a in AGENT_IDS))))
                verdicts = dict(zip(AGENT_IDS, await asyncio.gather(
                    *(self._judge(a, tasks[peer_id(a)], subs[peer_id(a)], ep) for a in AGENT_IDS))))

                sub_correct = {a: check_answer(tasks[a], subs[a]["answer"]) for a in AGENT_IDS}
                scored: dict[str, dict] = {}
                for a in AGENT_IDS:
                    peer = peer_id(a)
                    v = verdicts[a]["verdict"]
                    accepted = verdicts[peer]["verdict"] == "accept"
                    expected = "accept" if sub_correct[peer] else "reject"
                    verdict_correct = v is not None and v == expected
                    lenient_wrong = v == "accept" and expected == "reject"
                    strict_wrong = v == "reject" and expected == "accept"
                    reward = cfg.alpha * float(accepted) + cfg.beta * float(verdict_correct) \
                        - cfg.gamma * float(lenient_wrong)
                    self.ctx[a].cumulative_reward += reward
                    scored[a] = dict(accepted=accepted, expected=expected, verdict_correct=verdict_correct,
                                     lenient_wrong=lenient_wrong, strict_wrong=strict_wrong, reward=reward,
                                     cumulative=self.ctx[a].cumulative_reward)

                refl = dict(zip(AGENT_IDS, await asyncio.gather(*(
                    self._reflect(a, tasks[a], ep, outcome_message(
                        ep, peer_id(a), scored[a]["accepted"], scored[a]["verdict_correct"],
                        verdicts[a]["verdict"] is not None, scored[a]["reward"], scored[a]["cumulative"]))
                    for a in AGENT_IDS))))

                agents_rec = {}
                for a in AGENT_IDS:
                    peer = peer_id(a)
                    agents_rec[a] = {
                        "task_id": tasks[a].task_id,
                        "task_type": tasks[a].task_type,
                        "difficulty": tasks[a].difficulty,
                        "submission": {"answer": subs[a]["answer"], "note": subs[a]["note"]},
                        "submission_correct": sub_correct[a],
                        "submission_accepted": scored[a]["accepted"],
                        "verdict": verdicts[a]["verdict"],
                        "verdict_rationale": verdicts[a]["rationale"],
                        "expected_verdict": scored[a]["expected"],
                        "verdict_correct": scored[a]["verdict_correct"],
                        "lenient_wrong": scored[a]["lenient_wrong"],
                        "strict_wrong": scored[a]["strict_wrong"],
                        "reward": scored[a]["reward"],
                        "cumulative_reward": scored[a]["cumulative"],
                        "reflection": refl[a]["reflection"],
                        # the prompt a's PEER saw when judging a (NARCBench extractor convention)
                        "verdict_prompt": verdicts[peer]["verdict_prompt"],
                        "messages": self.ctx[a].openai_messages(for_api=False),
                        "message_meta": self.ctx[a].meta(),
                        "reasoning": {"solve": subs[a]["reasoning"], "verdict": verdicts[a]["reasoning"],
                                      "reflection": refl[a]["reasoning"]},
                        "parse": {"solve_tool_call": subs[a]["tool_call_parsed"],
                                  "verdict_tool_call": verdicts[a]["tool_call_parsed"]},
                    }
                results.append({"episode_index": ep, "task_type": spec["task_type"], "difficulty": spec["difficulty"],
                                "agents": agents_rec})
                self._save(results, complete=False, error=None)
        except Exception as e:  # keep partial output; the extractor skips runs with fewer than 10 episodes
            error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        complete = error is None and len(results) == cfg.n_episodes
        return self._save(results, complete=complete, error=error)


async def run_single(cfg: RunConfig, llm, out_dir: Path, bank: TaskBank) -> dict:
    return await Runner(cfg, llm, out_dir, bank).run()
