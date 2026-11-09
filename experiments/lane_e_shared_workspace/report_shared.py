#!/usr/bin/env python3
"""Markdown tables for a phase 2b experiment root: per run use and first episodes,
blind outcomes, the audited graph, cost, and every workspace write for a human audit.
Usage: python experiments/lane_e_shared_workspace/report_shared.py runs/<exp> [runs/<exp2> ...]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from swarm.audit import collect_shared  # noqa: E402

PRICES = {"qwen/qwen3-14b": (0.12e-6, 0.24e-6), "qwen/qwen3-32b": (0.08e-6, 0.28e-6)}


def ep(x) -> str:
    return "-" if x is None else str(x + 1)


def usage(run_dir: Path, model: str) -> dict:
    calls = pt = ct = trunc = 0
    first = last = None
    for line in open(run_dir / "snapshots.jsonl"):
        d = json.loads(line)
        r = d["response"]
        u = r.get("usage") or {}
        calls += 1
        pt += u.get("prompt_tokens") or 0
        ct += u.get("completion_tokens") or 0
        trunc += r.get("finish_reason") == "length"
        first = first or d["ts"]
        last = d["ts"]
    p, c = PRICES.get(model, (0.0, 0.0))
    return {"calls": calls, "prompt": pt, "completion": ct, "truncated": trunc, "cost": pt * p + ct * c,
            "first": first, "last": last}


def main() -> int:
    for root in map(Path, sys.argv[1:]):
        lab = collect_shared(root)
        print(f"### {root}\n")
        print("| run | complete | calls | cost USD | truncated | ws calls | first call | first write | first cross-agent read "
              "| first carried | blind correct | via write-then-read | blind in round 2 |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        tot = Counter()
        for r in lab["runs"]:
            sd = root / Path(r["manifest"]).parent.name / "shared_file"
            run = json.load(open(sd / "run.json"))
            u = usage(sd, run["run_config"]["model"])
            r2 = sum(1 for e in run["results"] for a in e["agents"].values() if a["blind"] and a["submission"]["round"] == 2)
            for k in ("calls", "truncated"):
                tot[k] += u[k]
            tot["cost"] += u["cost"]
            tot["r2"] += r2
            print(f"| {sd.parent.name} | {r['complete']} | {u['calls']} | {u['cost']:.3f} | {u['truncated']} | "
                  f"{r['n_tool_calls']} | {ep(r['first_call_episode'])} | {ep(r['first_write_episode'])} | "
                  f"{ep(r['first_comm_episode'])} | {ep(r['first_carried_episode'])} | {r['blind_correct']}/{r['blind']} | "
                  f"{r['blind_correct_via_channel']}/{r['blind']} | {r2}/{r['blind']} |")
        print(f"| total | {lab['n_complete']}/{lab['n_runs']} | {tot['calls']} | {tot['cost']:.3f} | {tot['truncated']} | "
              f"{sum(r['n_tool_calls'] for r in lab['runs'])} | runs {lab['runs_with_any_call']} | runs "
              f"{lab['runs_with_any_write']} | runs {lab['runs_with_comm']} | runs {lab['runs_with_carried']} | "
              f"{lab['blind_correct']}/{lab['blind']} | {lab['blind_correct_via_channel']}/{lab['blind']} | "
              f"{tot['r2']}/{lab['blind']} |\n")
        mc = lab["mask_check"]
        print(f"Derived mask against the owner record over {mc['reads']} reads. Edges tp {mc['edge_tp']}, fp "
              f"{mc['edge_fp']}, fn {mc['edge_fn']}, precision {mc['edge_precision']}, recall {mc['edge_recall']}.\n")
        print("| run | audited graph, writer -> reader (read episodes) [answer episodes] |")
        print("| --- | --- |")
        for r in lab["runs"]:
            g = "; ".join(f"{x['writer']} -> {x['reader']} ({', '.join(str(e + 1) for e in x['read_episodes'])})"
                          + (f" [{', '.join(str(e + 1) for e in x['answer_episodes'])}]" if x["answer_episodes"] else "")
                          for x in r["graph"]) or "none"
            print(f"| {Path(r['manifest']).parent.name} | {g} |")
        print()
        print("Every workspace write, for a human audit.\n")
        print("| run | seq | episode | phase, round | writer | path | carried for | bytes |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- |")
        n = 0
        for d in sorted(root.glob("seed*/shared_file/audit.json")):
            aud = json.load(open(d))
            run = json.load(open(d.parent / "run.json"))
            cw = {}
            for c in aud["carried_writes"]:
                cw.setdefault(c["write_seq"], []).append(f"{c['target']} ({c['kind']}, before={c['before_target_submit']})")
            for e in run["tool_log"]:
                if e["tool"] != "write_file":
                    continue
                n += 1
                body = e["bytes"].replace("|", "/").replace("\n", " / ")[:160]
                print(f"| {d.parent.parent.name} | {e['seq']} | {e['episode_index'] + 1} | {e['phase']}, {e.get('round')} | "
                      f"{e['agent']} | {e['path']} | {'; '.join(cw.get(e['seq'], [])) or '-'} | {body} |")
        if not n:
            print("| - | - | - | - | - | - | - | no writes |")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
