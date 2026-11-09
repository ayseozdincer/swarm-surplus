#!/usr/bin/env python3
"""Lane G. Merge run_plants.py output directories (one per leak address, so no two processes wrote one directory) into the
single experiment directory the replay and the scorers expect (runs/lane_g/sweep2_leak2). Seed directories whose leak2 and none
branches are both complete are hard-linked in; the task manifests are merged by task_id and the merge refuses to continue
if one task_id has two different entries.

  python experiments/lane_g_two_notes/merge_runs.py runs/leak2_name runs/leak2_broad runs/lane_g/sweep2_leak2
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    *sources, dest = map(Path, sys.argv[1:])
    dest.mkdir(parents=True, exist_ok=True)
    (dest/"tasks").mkdir(exist_ok=True)
    manifests: dict[str, dict[str, dict]] = {}
    seeds = 0
    for src in sources:
        for seed_dir in sorted(src.glob("seed*")):
            complete = all((seed_dir/v/"run.json").exists() and json.loads((seed_dir/v/"run.json").read_text()).get("complete")
                           for v in ("leak2", "none"))
            if not complete:
                print(f"skip {seed_dir}: a branch is missing or incomplete")
                continue
            target = dest/seed_dir.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(seed_dir, target, copy_function=os.link)
            seeds += 1
        for manifest in sorted((src/"tasks").glob("manifest_*.json")):
            entries = manifests.setdefault(manifest.name, {})
            for entry in json.loads(manifest.read_text()):
                if entry["task_id"] in entries and entries[entry["task_id"]] != entry:
                    raise SystemExit(f"two different entries for {entry['task_id']} ({manifest})")
                entries[entry["task_id"]] = entry
    for name, entries in manifests.items():
        (dest/"tasks"/name).write_text(json.dumps(sorted(entries.values(), key=lambda e: e["task_id"]), indent=1))
    print(f"merged {seeds} seeds into {dest}; manifests {sorted(manifests)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
