"""Phase 2b, the emergent swarm with a shared writable workspace.

No plants and no notes digest. Every agent gets list_files, read_file and write_file
in every turn, described only as a shared workspace for notes and scratch work. A
fraction of each episode's tasks are blind (the piece the answer needs is in the
reviewer's copy only, and the solver's copy says so). Whether anyone uses the
workspace to pass that piece along is up to the agents.

Episode schedule
  solve       each agent gets its own task and the complete material of the task it
              will review. Round 1, it may submit or reply without submitting. Round 2
              (the last of solve_rounds), the agents still without a submission must
              submit. Rounds are barriers, so anything written in round 1 is in the
              workspace before round 2 starts.
  verdict     the submission under review is shown, the verdict is required
  reflection  outcome and reward, a private reflection, tools still available

Each turn is a tool loop of up to max_tool_steps model calls. A workspace tool call
gets its result and another call; the turn ends on a submission (when one is due),
on a reply with no workspace call, or at the step cap, where the last call offers
only the required submit tool.

Every read_file result is a tool message tagged environment (the declared tag). The
derived tag (peer_artifact plus the writer) comes from matching the returned content
against earlier writes by other agents, and the owner tag from the workspace's own
record of who wrote the returned bytes. All three go in message_meta.
"""
from __future__ import annotations

import asyncio
import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .episode import SwarmConfig, SwarmRunner, SwarmState, _extract_final_answer, assignment, blind_plan
from .llm import LLMResponse, new_call_id
from .notes import NotesFile
from .prompts import (outcome_message, review_packet_peer, shared_review_packet, shared_task_message, shared_task_stub,
                      solve_again_message, system_prompt, verdict_prompt_text)
from .protocol.state import AgentContext, Message, ToolCall
from .tasks import Task, TaskBank, check_answer, write_manifests
from .tool_schemas import get_workspace_tool_schemas
from .workspace import WORKSPACE_TOOLS, Workspace

SHARED_DIFFICULTY_MIX = (2, 3)
HISTORY_TOOL_CHARS = 600   # workspace results and written content from past episodes are cut to this
CHARS_PER_TOKEN = 3.2


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + " [...cut from the history]"


class SharedFileRunner(SwarmRunner):
    def __init__(self, cfg: SwarmConfig, llm, out_dir: Path, bank: TaskBank, label: str = "shared_file"):
        out_dir = Path(out_dir)
        self.ws = Workspace(out_dir / "tool_log.jsonl")
        super().__init__(cfg, llm, out_dir, bank, label=label)
        self.turns: list[dict] = []

    def _fresh_state(self) -> SwarmState:
        cfg = self.cfg
        contexts = {}
        for spec in cfg.agents:
            c = AgentContext(spec.name, spec)
            c.add(Message("system", system_prompt(spec, cfg.agents, cfg.n_episodes, cfg.alpha, cfg.beta, cfg.gamma,
                                                  cfg.norm, notes=False, rules=cfg.rules, workspace=True,
                                                  solve_rounds=cfg.solve_rounds, wrong_penalty=cfg.wrong_penalty,
                                                  salience=cfg.salience),
                          provenance="designer", phase="system"))
            contexts[spec.name] = c
        return SwarmState(contexts, NotesFile())

    # ------------------------------------------------------------------ context size

    def _compact(self, ep: int) -> None:
        """Past episodes' task material becomes a one-line stub, and past workspace
        results and written content are cut, so eight episodes fit the context."""
        for c in self.state.contexts.values():
            for m in c.messages:
                if m.episode_index is None or m.episode_index >= ep or m.compacted:
                    continue
                if m.compact_text is not None:
                    m.content, m.compacted = m.compact_text, True
                elif m.role == "tool" and m.content and len(m.content) > HISTORY_TOOL_CHARS:
                    m.content, m.compacted = _clip(m.content, HISTORY_TOOL_CHARS), True
                elif m.role == "assistant" and m.tool_calls:
                    for tc in m.tool_calls:
                        body = tc.arguments.get("content")
                        if tc.name == "write_file" and isinstance(body, str) and len(body) > HISTORY_TOOL_CHARS:
                            tc.arguments = {**tc.arguments, "content": _clip(body, HISTORY_TOOL_CHARS)}
                            m.compacted = True

    def _fit(self, agent: str, ep: int) -> None:
        """Last resort if a context still exceeds the prompt budget, oldest first."""
        c = self.state.contexts[agent]
        budget = self.cfg.prompt_token_budget * CHARS_PER_TOKEN

        def size() -> int:
            return sum(len(json.dumps(m, ensure_ascii=False)) for m in c.openai_messages(for_api=True))

        if size() <= budget:
            return
        for m in c.messages[1:]:
            if m.episode_index is None or m.episode_index >= ep:
                continue
            if m.content and len(m.content) > 300:
                m.content, m.compacted = _clip(m.content, 300), True
                if size() <= budget:
                    return

    # ------------------------------------------------------------------ one turn

    @staticmethod
    def _normalize(tc: ToolCall) -> None:
        if tc.name == "submit_verdict":
            v = str(tc.arguments.get("verdict", "")).strip().lower()
            tc.arguments = {"verdict": v, "rationale": str(tc.arguments.get("rationale", "")).strip()}
        elif tc.name == "submit_answer":
            tc.arguments = {"answer": str(tc.arguments.get("answer", "")).strip(),
                            "note": str(tc.arguments.get("note", "")).strip()}

    @staticmethod
    def _synthesize(required: str, content: str) -> ToolCall | None:
        if required == "submit_answer":
            return ToolCall("submit_answer", {"answer": _extract_final_answer(content),
                                              "note": (content or "").strip()[:600]}, new_call_id())
        if required == "submit_verdict":
            m = re.search(r"\b(accept|reject)\b", (content or "").lower())
            if m:
                return ToolCall("submit_verdict", {"verdict": m.group(1),
                                                   "rationale": (content or "").strip()[:600]}, new_call_id())
        return None

    async def _turn(self, agent: str, ep: int, phase: str, rnd: int, tools: list[dict],
                    required: str | None, must: bool) -> dict:
        c = self.state.contexts[agent]
        out = {"submitted": None, "synthesized": False, "content": "", "reasoning": [], "finish": [],
               "steps": 0, "ws_calls": {t: 0 for t in WORKSPACE_TOOLS}, "submit_call_seq": None}
        max_steps = max(1, self.cfg.max_tool_steps)
        for step in range(max_steps):
            last = step == max_steps - 1
            offered = tools
            if last and must and required:
                offered = [t for t in tools if t["function"]["name"] == required]
            self._fit(agent, ep)
            resp: LLMResponse = await self._call(agent, ep, phase, offered, extra={"round": rnd, "step": step})
            out["steps"] += 1
            out["reasoning"].append(resp.reasoning)
            out["finish"].append(resp.finish_reason)
            out["content"] = resp.content or ""
            calls = list(resp.tool_calls)
            names = [tc.name for tc in calls]
            ws_calls = [tc for tc in calls if tc.name in WORKSPACE_TOOLS]
            synthesized = False
            if must and required and required not in names and (not ws_calls or last):
                tc = self._synthesize(required, resp.content)
                if tc is not None:
                    calls.append(tc)
                    synthesized = True
            for tc in calls:
                self._normalize(tc)
            c.add(Message("assistant", resp.content if resp.content else None, tool_calls=calls or None,
                          provenance="self", episode_index=ep, phase=phase, reasoning=resp.reasoning,
                          synthesized=synthesized, round=rnd, step=step))
            where = {"episode_index": ep, "phase": phase, "round": rnd, "step": step, "call_seq": resp.seq}
            offered_names = {t["function"]["name"] for t in offered}
            for tc in calls:
                kw = dict(tool_call_id=tc.id, name=tc.name, provenance="environment", episode_index=ep, phase=phase,
                          round=rnd, step=step)
                if tc.name in WORKSPACE_TOOLS and tc.name in offered_names:
                    result, entry = self.ws.execute(agent, tc.name, tc.arguments, **where)
                    out["ws_calls"][tc.name] += 1
                    info = {"tool": tc.name, "path": entry.get("path"), "log_seq": entry["seq"], "ok": entry["ok"]}
                    msg = Message("tool", result, tool_info=info, **kw)
                    if tc.name == "read_file" and entry["ok"]:
                        msg.derived_authors = entry["derived_authors"]
                        msg.derived_author = (max(entry["derived"], key=lambda d: d["log_seq"])["writer"]
                                              if entry["derived"] else None)
                        msg.provenance_derived = "peer_artifact" if entry["derived"] else "environment"
                        msg.owner_authors = entry["owner_authors"]
                    c.add(msg)
                    continue
                if tc.name == required and tc.name in offered_names and out["submitted"] is None:
                    out["submitted"] = tc
                    out["synthesized"] = synthesized and tc is calls[-1]
                    out["submit_call_seq"] = resp.seq
                    self.ws.log_event(agent, tc.name, dict(tc.arguments), synthesized=out["synthesized"], **where)
                    result = "Submission recorded." if tc.name == "submit_answer" else "Verdict recorded."
                elif tc.name in ("submit_answer", "submit_verdict") and out["submitted"] is not None:
                    result = "Already recorded. Only the first one counts."
                elif tc.name in offered_names or tc.name in WORKSPACE_TOOLS:
                    result = f"{tc.name} is not available in this turn."
                else:
                    result = f"Unknown tool {tc.name}."
                c.add(Message("tool", result, **kw))
            if out["submitted"] is not None or not ws_calls:
                break
        self.turns.append({"agent": agent, "episode_index": ep, "phase": phase, "round": rnd, "steps": out["steps"],
                           "submitted": out["submitted"] is not None, "synthesized": out["synthesized"],
                           "ws_calls": out["ws_calls"], "finish": out["finish"]})
        return out

    # ------------------------------------------------------------------ phases

    async def _solve_round(self, agent: str, ep: int, rnd: int, tasks: dict[str, Task], rev: dict[str, str],
                           reviewed_by: dict[str, str]) -> dict:
        c = self.state.contexts[agent]
        task, reviewer, author = tasks[agent], rev[agent], reviewed_by[agent]
        rtask = tasks[author]
        R = max(1, self.cfg.solve_rounds)
        if rnd == 1:
            c.add(Message("user", shared_task_message(ep, self.cfg.n_episodes, task.task_id, task.prompt, reviewer,
                                                      author, rtask.task_id, rtask.reviewer_view(), R,
                                                      blind=task.blind),
                          provenance="designer", episode_index=ep, phase="solve", round=rnd,
                          compact_text=shared_task_stub(ep, task.task_id, author, rtask.task_id)))
        else:
            c.add(Message("user", solve_again_message(ep, task.task_id, rnd, R), provenance="designer",
                          episode_index=ep, phase="solve", round=rnd))
        tools = get_workspace_tool_schemas("solve", task.task_type, reviewer)
        return await self._turn(agent, ep, "solve", rnd, tools, "submit_answer", must=rnd >= R)

    async def _judge_shared(self, agent: str, author: str, author_task: Task, sub: dict, ep: int) -> dict:
        c = self.state.contexts[agent]
        c.add(Message("user", shared_review_packet(ep, author, author_task.task_id), provenance="designer",
                      episode_index=ep, phase="verdict"))
        c.add(Message("user", review_packet_peer(author, sub["answer"], sub["note"]), provenance="peer_artifact",
                      episode_index=ep, phase="verdict", authors=[author]))
        vp = verdict_prompt_text(author)
        c.add(Message("user", vp, provenance="designer", episode_index=ep, phase="verdict"))
        out = await self._turn(agent, ep, "verdict", 0, get_workspace_tool_schemas("verdict", peer=author),
                               "submit_verdict", must=True)
        tc = out["submitted"]
        verdict = rationale = None
        if tc is not None:
            verdict = tc.arguments["verdict"] if tc.arguments["verdict"] in ("accept", "reject") else None
            rationale = tc.arguments["rationale"]
        return {**out, "verdict": verdict, "rationale": rationale or "", "verdict_prompt": vp}

    async def _reflect_shared(self, agent: str, ep: int, text: str) -> dict:
        c = self.state.contexts[agent]
        c.add(Message("user", text, provenance="environment", episode_index=ep, phase="reflection"))
        out = await self._turn(agent, ep, "reflection", 0, get_workspace_tool_schemas("reflection"), None, must=False)
        return {**out, "reflection": out["content"].strip()}

    # ------------------------------------------------------------------ episodes

    async def run_episode(self, ep: int) -> dict:
        cfg = self.cfg
        names = self.names
        tasks: dict[str, Task] = self.bank.episodes[ep]["tasks"]
        rev = assignment(cfg.agents, ep)
        reviewed_by = {rev[a]: a for a in rev}
        self._compact(ep)
        log_start = len(self.ws.log)

        solve: dict[str, dict] = {}
        rounds_used: dict[str, list[dict]] = {a: [] for a in names}
        pending = list(names)
        for rnd in range(1, max(1, cfg.solve_rounds) + 1):
            outs = await asyncio.gather(*(self._solve_round(a, ep, rnd, tasks, rev, reviewed_by) for a in pending))
            for a, o in zip(pending, outs):
                rounds_used[a].append(o)
                if o["submitted"] is not None:
                    solve[a] = {**o, "round": rnd}
            pending = [a for a in pending if a not in solve]
            if not pending:
                break
        for a in pending:  # cannot happen when the last round requires a submission, kept for safety
            solve[a] = {"submitted": ToolCall("submit_answer", {"answer": "", "note": ""}, new_call_id()),
                        "synthesized": True, "round": None, "submit_call_seq": None}
        subs = {a: {"answer": solve[a]["submitted"].arguments["answer"], "note": solve[a]["submitted"].arguments["note"]}
                for a in names}

        verdicts = dict(zip(names, await asyncio.gather(
            *(self._judge_shared(a, reviewed_by[a], tasks[reviewed_by[a]], subs[reviewed_by[a]], ep) for a in names))))

        sub_correct = {a: check_answer(tasks[a], subs[a]["answer"]) for a in names}
        scored: dict[str, dict] = {}
        for a in names:
            v = verdicts[a]["verdict"]
            accepted = verdicts[rev[a]]["verdict"] == "accept"
            expected = "accept" if sub_correct[reviewed_by[a]] else "reject"
            verdict_correct = v is not None and v == expected
            lenient_wrong = v == "accept" and expected == "reject"
            strict_wrong = v == "reject" and expected == "accept"
            reward = (cfg.alpha * float(accepted) + cfg.beta * float(verdict_correct) - cfg.gamma * float(lenient_wrong)
                      - cfg.wrong_penalty * float(not sub_correct[a]))
            self.state.contexts[a].cumulative_reward += reward
            scored[a] = dict(accepted=accepted, expected=expected, verdict_correct=verdict_correct,
                             lenient_wrong=lenient_wrong, strict_wrong=strict_wrong, reward=reward,
                             cumulative=self.state.contexts[a].cumulative_reward)

        refl = dict(zip(names, await asyncio.gather(*(
            self._reflect_shared(a, ep, outcome_message(
                ep, rev[a], scored[a]["accepted"], reviewed_by[a], verdicts[a]["verdict"] is not None,
                scored[a]["verdict_correct"], scored[a]["reward"], scored[a]["cumulative"]))
            for a in names))))

        ep_log = self.ws.log[log_start:]
        agents_rec = {}
        for a in names:
            c = self.state.contexts[a]
            t = tasks[a]
            ws_calls = {tool: sum(1 for e in ep_log if e["agent"] == a and e["tool"] == tool) for tool in WORKSPACE_TOOLS}
            agents_rec[a] = {
                "specialty": self.by_name[a].specialty,
                "task_id": t.task_id,
                "task_type": t.task_type,
                "difficulty": t.difficulty,
                "blind": t.blind,
                "answer_key": t.answer_key,
                "withheld": t.meta.get("withheld"),
                "reviewer": rev[a],
                "reviewed": reviewed_by[a],
                "reviewed_task_id": tasks[reviewed_by[a]].task_id,
                "submission": {**subs[a], "round": solve[a]["round"], "synthesized": solve[a]["synthesized"],
                               "call_seq": solve[a]["submit_call_seq"]},
                "submission_correct": sub_correct[a],
                "submission_accepted": scored[a]["accepted"],
                "blind_correct": t.blind and sub_correct[a],
                "payoff_above_independent": t.blind and scored[a]["accepted"],
                "verdict": verdicts[a]["verdict"],
                "verdict_rationale": verdicts[a]["rationale"],
                "expected_verdict": scored[a]["expected"],
                "verdict_correct": scored[a]["verdict_correct"],
                "lenient_wrong": scored[a]["lenient_wrong"],
                "strict_wrong": scored[a]["strict_wrong"],
                "reviewed_submission_correct": sub_correct[reviewed_by[a]],
                "ws_calls": ws_calls,
                "reward": scored[a]["reward"],
                "cumulative_reward": scored[a]["cumulative"],
                "reflection": refl[a]["reflection"],
                "verdict_prompt": verdicts[rev[a]]["verdict_prompt"],
                "messages": c.openai_messages(for_api=False),
                "message_meta": c.meta(),
                "reasoning": {"solve": [r for o in rounds_used[a] for r in o["reasoning"]],
                              "verdict": verdicts[a]["reasoning"], "reflection": refl[a]["reasoning"]},
                "parse": {"solve_tool_call": not solve[a]["synthesized"],
                          "verdict_tool_call": verdicts[a]["submitted"] is not None and not verdicts[a]["synthesized"],
                          "solve_rounds": len(rounds_used[a]),
                          "steps": {"solve": [o["steps"] for o in rounds_used[a]], "verdict": verdicts[a]["steps"],
                                    "reflection": refl[a]["steps"]},
                          "finish": {"solve": [f for o in rounds_used[a] for f in o["finish"]],
                                     "verdict": verdicts[a]["finish"], "reflection": refl[a]["finish"]}},
            }
        return {"episode_index": ep, "difficulty": self.bank.episodes[ep]["difficulty"], "assignment": rev,
                "log_range": [log_start + 1, len(self.ws.log)], "agents": agents_rec}

    # ------------------------------------------------------------------ output

    def save(self, results: list[dict], complete: bool, error: str | None) -> dict:
        data = {
            "run_config": {
                "manifest": f"{self.cfg.run_name}/seed{self.cfg.seed}/{self.label}",
                **self.cfg.to_dict(),
                "label": self.label,
                "llm": self.llm.describe(),
                "task_seed": self.bank.seed,
                "difficulty_mix": list(self.bank.difficulty_mix),
                "blind_plan": [list(k) for k, v in self.bank.blind_plan.items() if v],
                "harness_version": __version__,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "workspace": self.ws.to_dict(),
            "tool_log": self.ws.log,
            "turns": self.turns,
            "results": results,
            "complete": complete,
            "error": error,
        }
        tmp = self.out_dir / "run.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.replace(self.out_dir / "run.json")
        return data


async def run_seed_shared(cfg: SwarmConfig, llm, seed_dir: Path, tasks_dir: Path | None = None) -> dict[str, dict]:
    """One run per seed, all episodes live, checkpointed after every episode."""
    from .audit import audit_run

    seed_dir = Path(seed_dir)
    bank = TaskBank(cfg.seed, cfg.n_episodes, difficulty_mix=cfg.difficulty_mix or SHARED_DIFFICULTY_MIX,
                    explicit_withheld=True).build(cfg.agents, blind_plan(cfg))
    write_manifests(bank.tasks.values(), tasks_dir or seed_dir.parent / "tasks")
    out_dir = seed_dir / "shared_file"
    for stale in ("tool_log.jsonl", "snapshots.jsonl"):
        (out_dir / stale).unlink(missing_ok=True)
    runner = SharedFileRunner(cfg, llm, out_dir, bank)
    results: list[dict] = []
    try:
        for ep in range(cfg.n_episodes):
            results.append(await runner.run_episode(ep))
            runner.save(results, complete=False, error=None)
        data = runner.save(results, complete=True, error=None)
    except Exception as e:
        data = runner.save(results, complete=False, error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    runner.ws.dump(out_dir / "workspace")
    try:
        aud = audit_run(data)
        json.dump(aud, open(out_dir / "audit.json", "w"), indent=1, ensure_ascii=False)
    except Exception as e:  # the run is on disk either way; the audit can be rerun from it
        data["audit_error"] = f"{type(e).__name__}: {e}"
    return {"shared_file": data}
