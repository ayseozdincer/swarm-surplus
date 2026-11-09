#!/usr/bin/env python3
"""Lane D. Would the agent have reached the planted right value without the note? For each target of the solvable arm's right-note branch,
whether it submitted the value against whether the same agent on the same task was correct in the none branch of that seed, and in the
method_tip arm's none branch, a second independent run of the same agent and task. Writes results/lane_d/solvable_none_twin.json.

  python scripts/solvable_none_twin.py
"""
from __future__ import annotations

import json
import re
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBER = re.compile(r"-?\d+(\.\d+)?$")


def task_key(arm: str, task_id: str) -> str:
    for f in (ROOT/"runs"/arm/"tasks").glob("manifest_*.json"):
        for task in json.loads(f.read_text()):
            if task["task_id"] == task_id:
                return task["answer_key"][0]
    raise KeyError(task_id)


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    n, row, col = a + b + c + d, a + b, a + c
    p = lambda x: comb(col, x)*comb(n - col, row - x)/comb(n, row)
    return sum(p(x) for x in range(max(0, row + col - n), min(row, col) + 1) if p(x) <= p(a)*(1 + 1e-9))


def main() -> int:
    rows = []
    for r in json.loads((ROOT/"results/lane_d/solvable_behaviour.json").read_text())["rows"]:
        if r["variant"] != "leak_solvable":
            continue
        seed, target = r["seed"], r["target"]
        first = json.loads((ROOT/f"runs/sweep2_solvable/{seed}/none/run.json").read_text())["results"][2]["agents"][target]
        second = json.loads((ROOT/f"runs/sweep2_methodtip/{seed}/none/run.json").read_text())["results"][2]["agents"][target]
        key = task_key("sweep2_solvable", first["task_id"])
        assert key == r["value"] == task_key("sweep2_methodtip", second["task_id"]), (seed, target)
        rows.append({"seed": seed, "target": target, "answer_type": "number" if NUMBER.match(key) else "string", "value": key,
                     "submitted": r["copied_value"], "none_right": first["submission_correct"], "second_none_right": second["submission_correct"],
                     "disclosed": r["disclosed"], "reply_tokens": r["reply_tokens"], "none_reply_tokens": r["none_reply_tokens"]})
    table = {name: {"n": len(sel), "none_right": sum(x["none_right"] for x in sel), "none_wrong": sum(not x["none_right"] for x in sel),
                    "right_in_both_none": sum(x["none_right"] and x["second_none_right"] for x in sel),
                    "right_in_one_none": sum(x["none_right"] != x["second_none_right"] for x in sel),
                    "wrong_in_both_none": sum(not x["none_right"] and not x["second_none_right"] for x in sel)}
             for name, sel in (("submitted", [x for x in rows if x["submitted"]]), ("did not submit", [x for x in rows if not x["submitted"]]))}
    types = {t: {"n": sum(x["answer_type"] == t for x in rows), "submitted": sum(x["answer_type"] == t and x["submitted"] for x in rows),
                 "none_right": sum(x["answer_type"] == t and x["none_right"] for x in rows),
                 "second_none_right": sum(x["answer_type"] == t and x["second_none_right"] for x in rows)} for t in ("number", "string")}
    a, b = table["submitted"]["none_right"], table["submitted"]["none_wrong"]
    c, d = table["did not submit"]["none_right"], table["did not submit"]["none_wrong"]
    out = {"table": table, "fisher_two_sided": fisher_two_sided(a, b, c, d), "answer_type": types,
           "none_runs_disagree": sum(x["none_right"] != x["second_none_right"] for x in rows), "rows": rows}
    (ROOT/"results/lane_d/solvable_none_twin.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for x in rows:
        print(x["seed"], x["answer_type"], "submitted", int(x["submitted"]), "none", int(x["none_right"]), "second none", int(x["second_none_right"]), x["reply_tokens"], x["none_reply_tokens"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
