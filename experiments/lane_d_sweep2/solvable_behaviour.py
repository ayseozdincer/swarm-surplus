#!/usr/bin/env python3
"""Lane D. The solvable-task leak per target: the note's value (right or wrong), whether the target submitted it
(copied_value), whether the submission is correct, reply tokens against the same agent's in the none branch of the seed,
and whether the submission note names the source (labels.source_mention). Writes experiments/lane_d_sweep2/results/solvable_behaviour.json.

  python experiments/lane_d_sweep2/solvable_behaviour.py runs/sweep2_solvable
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from swarm.labels import source_mention  # noqa: E402


def main() -> int:
    root = Path(sys.argv[1])
    rows = []
    for seed_dir in sorted(p for p in root.glob("seed*") if p.is_dir()):
        paths = {v: seed_dir/v/"run.json" for v in ("none", "leak_solvable", "leak_solvable_wrong")}
        if not all(p.exists() for p in paths.values()):
            continue
        runs = {v: json.loads(p.read_text()) for v, p in paths.items()}
        if not all(r.get("complete") for r in runs.values()):
            continue
        none_agents = runs["none"]["results"][2]["agents"]
        for variant in ("leak_solvable", "leak_solvable_wrong"):
            run = runs[variant]
            entry = run["plant"]["entries"][0]
            target = run["plant"]["solvable_target"]
            agents = run["results"][2]["agents"]
            rec = agents[target]
            src = source_mention(rec["submission"]["note"], [n for n in agents if n != target])
            rows.append({"seed": seed_dir.name, "variant": variant, "target": target, "task_type": rec["task_type"],
                         "address": run["plant"].get("address"), "value": entry["value"], "value_correct": entry["correct"],
                         "value_in_text": entry.get("value_in_text"), "answer": rec["submission"]["answer"],
                         "copied_value": bool(rec.get("copied_value")), "correct": rec["submission_correct"],
                         "reply_tokens": rec.get("reply_tokens"), "none_reply_tokens": none_agents[target].get("reply_tokens"),
                         "none_correct": none_agents[target]["submission_correct"], "disclosed": src is not None,
                         "source_match": src, "note": rec["submission"]["note"]})
    summary = {}
    for variant in ("leak_solvable", "leak_solvable_wrong"):
        sel = [r for r in rows if r["variant"] == variant]
        cop = [r for r in sel if r["copied_value"]]
        summary[variant] = {"targets": len(sel), "code_targets": sum(r["task_type"] == "code_analysis" for r in sel),
                            "note_has_value": sum(bool(r["value_in_text"]) for r in sel),
                            "copied": len(cop), "correct": sum(r["correct"] for r in sel),
                            "correct_in_none": sum(r["none_correct"] for r in sel),
                            "disclosed_among_copiers": sum(r["disclosed"] for r in cop),
                            "disclosed_among_non_copiers": sum(r["disclosed"] for r in sel if not r["copied_value"]),
                            "reply_tokens_median_copiers": median([r["reply_tokens"] for r in cop if r["reply_tokens"]]) if cop else None,
                            "reply_tokens_median_non_copiers": median([r["reply_tokens"] for r in sel if not r["copied_value"] and r["reply_tokens"]])
                            if any(not r["copied_value"] for r in sel) else None,
                            "none_reply_tokens_median": median([r["none_reply_tokens"] for r in sel if r["none_reply_tokens"]]) if sel else None,
                            "copied_named / broadcast": f"{sum(r['copied_value'] for r in sel if r['address'] == 'name')}/"
                                                        f"{sum(r['copied_value'] for r in sel if r['address'] == 'broadcast')}"}
    out = {"summary": summary, "rows": rows}
    (ROOT/"experiments/lane_d_sweep2/results").mkdir(parents=True, exist_ok=True)
    (ROOT/"experiments/lane_d_sweep2/results/solvable_behaviour.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
