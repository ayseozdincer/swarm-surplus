#!/usr/bin/env python3
"""W4. Teacher-forced replay of NARCBench Core public responses, actual against sanctioned reference.

Run from third_party/maj-lens on the pod. Rows and rendering are the J-space repo's and NARCBench's own (`_aligned`,
`_build_gen_text`). The reference for each row is detector.core_reference, teacher-forced on the same response tokens.

Actual and reference run through the same attention kernel, an explicit causal 4D mask, so a colluder's surplus holds no
kernel noise. Agents whose reference input equals their actual input (all control agents, honest agents in collusion runs)
would then be the same computation with surplus exactly zero by construction. Their reference pass runs on the plain kernel
instead, so they carry real replay noise. That is the conservative direction: the noise sits on the honest side.

  python ../../scripts/phase0.py replay --out ../../data/phase0/core
  python ../../scripts/phase0.py noise  --out ../../data/phase0/noise --n 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]
from detector.core_reference import honest_systems, reference_sample  # noqa: E402
from detector.render import token_at  # noqa: E402
from replay import LENS_LAYERS, Capture, attention_mask_4d, load_model  # noqa: E402

DATASET = Path("artifacts/narcbench")
SCENARIOS = DATASET/"scenarios/qwen3_32b/core"
THINK_END = "</think>\n\n"
POSITIONS = ("first_response", "end_of_response")


def core_samples():
    from utils.build.replay import _aligned, _upstream
    loader, _, render = _upstream()["core"]
    return [s for _, s in _aligned(DATASET, "core", loader, "qwen3_32b")], render


def encode(tok, render, sample):
    text = render(tok, sample)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
    start = token_at(offsets, text.rfind(THINK_END) + len(THINK_END))
    return ids, start


def forward(capture, model, ids, positions, explicit: bool):
    return capture.run(model, ids, positions, attention_mask_4d(len(ids)) if explicit else None)


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def replay(args, model, tok, capture):
    samples, render = core_samples()
    by_run: dict[str, list[dict]] = {}
    for s in samples:
        by_run.setdefault(s["run_id"], []).append(s)
    for run, rows in sorted(by_run.items()):
        directory = args.out/run
        if (directory/"index.json").exists():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        systems = honest_systems(SCENARIOS, run)
        turns, started = [], time.time()
        for s in rows:
            ids_a, start_a = encode(tok, render, s)
            ids_r, start_r = encode(tok, render, reference_sample(s, systems))
            if ids_a[start_a:] != ids_r[start_r:]:
                raise ValueError(f"response tokens differ: {run} {s['agent_name']} {s['round']}")
            same_input = ids_a == ids_r
            actual = forward(capture, model, ids_a, [start_a, len(ids_a) - 1], True)
            reference = forward(capture, model, ids_r, [start_r, len(ids_r) - 1], not same_input)
            name = f"{slug(s['agent_name'])}_r{s['round']}.npz"
            np.savez(directory/name, actual=actual.numpy(), reference=reference.numpy())
            turns.append({"file": name, "agent": s["agent_name"], "round": s["round"], "role": s["role"], "mode": s["mode"],
                          "domain": s["scenario_domain"], "tokens": len(ids_a), "reference_tokens": len(ids_r),
                          "response_tokens": len(ids_a) - start_a, "same_input": same_input,
                          "positions": dict(zip(POSITIONS, (start_a, len(ids_a) - 1))), "peer_token_spans": [[0, 0]]})
        (directory/"index.json").write_text(json.dumps({"run": run, "layers": capture.layers, "turns": turns}, indent=1))
        print(run, len(turns), f"{time.time() - started:.0f}s", flush=True)


def noise(args, model, tok, capture):
    """Twenty honest control agents at their last discussion round, each replayed three ways."""
    samples, render = core_samples()
    control = sorted({s["run_id"] for s in samples if s["mode"] == "control"})[:args.n]
    picks = []
    for i, run in enumerate(control):
        rows = [s for s in samples if s["run_id"] == run and s["round"] == 3]
        picks.append(sorted(rows, key=lambda s: s["agent_name"])[i % len(rows)])
    for name in ("repeat", "kernel"):
        (args.out/name).mkdir(parents=True, exist_ok=True)
    turns = {"repeat": [], "kernel": []}
    for s in picks:
        ids, start = encode(tok, render, s)
        positions = [start, len(ids) - 1]
        first = forward(capture, model, ids, positions, False)
        second = forward(capture, model, ids, positions, False)
        third = forward(capture, model, ids, positions, True)
        for label, other in (("repeat", second), ("kernel", third)):
            file = f"{s['run_id']}__{slug(s['agent_name'])}.npz"
            np.savez(args.out/label/file, actual=first.numpy(), reference=other.numpy())
            turns[label].append({"file": file, "run": s["run_id"], "agent": s["agent_name"], "round": s["round"], "tokens": len(ids),
                                 "positions": dict(zip(POSITIONS, positions)), "peer_token_spans": [[0, 0]]})
        print(s["run_id"], s["agent_name"], len(ids), flush=True)
    for label, rows in turns.items():
        (args.out/label/"index.json").write_text(json.dumps({"layers": capture.layers, "turns": rows}, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("replay", "noise"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    model, tok = load_model()
    capture = Capture(model, LENS_LAYERS)
    (replay if args.action == "replay" else noise)(args, model, tok, capture)
    return 0


if __name__ == "__main__":
    sys.exit(main())
