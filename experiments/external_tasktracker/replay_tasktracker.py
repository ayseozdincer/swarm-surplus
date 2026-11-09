#!/usr/bin/env python3
"""Replay the TaskTracker-shaped turns (swarm/external_tasktracker.py) teacher-forced, one model load.

Every turn runs two full passes on the explicit 4D-mask kernel. The actual pass is the turn as served. The reference pass masks
the external span (the injected sentence) as keys; a clean twin has nothing to mask, so its reference is the plain kernel and
its per-token delta is the kernel floor. Both passes keep the per-token relative surplus at every lens layer over the reply
(trace_delta_reference, reply tokens x 39 layers) and the per-token log-probability of the forced reply token (logprob_actual,
logprob_reference). Read positions include the first writing of the target string and of the model's own answer anywhere in
the reply (target_0, target_pre, own_0, own_pre). Beside each .npz a slim <sample_id>.trace.npz holds the trace and log-prob
arrays only, for syncing down.

  SWARM_MODEL=qwen3_14b SWARM_ARTIFACTS=/root/artifacts python experiments/external_tasktracker/replay_tasktracker.py data/external_tt/replies.jsonl \
      --out data/replay_tasktracker [--stop-at <unix time>]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts"), str(ROOT/"third_party/maj-lens")]
import replay  # noqa: E402

BUSY = Path("/workspace/busy")


def slim(path: Path) -> None:
    data = np.load(path)
    keep = {k: data[k] for k in data.files if k.startswith("trace_") or k.startswith("logprob_")}
    np.savez(path.with_suffix(".trace.npz"), **keep)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replies", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT/"data/replay_tasktracker")
    parser.add_argument("--stop-at", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    from detector.jspace import Lens
    model, tok = replay.load_model()
    lens = Lens(replay.MODEL_DIR, replay.LENS_FILE)
    capture = replay.Capture(model, lens.layers)
    rows = [json.loads(l) for l in open(args.replies, encoding="utf-8")]
    rows = [r for r in rows if r.get("response")]
    # poisoned twins first so a partial run still has positives, then their clean twins in the same order
    order = sorted(rows, key=lambda r: (r["twin"] != "poisoned", r["sample_id"]))
    if args.limit:
        order = order[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out/"index.json"
    old = json.loads(index_path.read_text())["turns"] if index_path.exists() else []
    done = {t["file"] for t in old}
    started = time.time()
    new = []
    for i, r in enumerate(order):
        path = args.out/f"{r['sample_id']}.npz"
        if path.name in done and path.exists():
            if not path.with_suffix(".trace.npz").exists():
                slim(path)
            continue
        if args.stop_at and time.time() > args.stop_at:
            print("== stop-at reached", flush=True)
            break
        lab = r.get("label") or {}
        mentions_reply = {k: lab[v] for k, v in (("own", "own_answer"), ("final", "final_answer")) if lab.get(v)}
        if r.get("target"):
            mentions_reply["target"] = r["target"]
        row = replay.replay_turn(model, tok, capture, r, lens, path, shared_prefix=False, compacted=False, mentions=None, shuffled=False,
                                 trace=True, trace_direction=False, logprobs=True, mentions_reply=mentions_reply)
        row.update({"sample_id": r["sample_id"], "paragraph_id": r["paragraph_id"], "twin": r["twin"], "template": r.get("template"),
                    "strength": r.get("strength"), "target": r.get("target"), "label": r.get("label")})
        slim(path)
        path.unlink()           # the full arrays are not read by the scorer; keep the volume small
        new.append(row)
        if BUSY.parent.exists():
            BUSY.touch()
        print(f"{i + 1}/{len(order)} {r['sample_id']} tokens {row['tokens']} masked {row['masked_tokens']} {row['reference_kind']} {row['seconds']}s", flush=True)
        if (i + 1) % 20 == 0 or i + 1 == len(order):
            index_path.write_text(json.dumps({"model": replay.MODEL_NAME, "layers": capture.layers, "turns": old + new}, indent=1))
    index_path.write_text(json.dumps({"model": replay.MODEL_NAME, "layers": capture.layers, "turns": old + new}, indent=1))
    print(f"== done {len(new)} new turns in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
