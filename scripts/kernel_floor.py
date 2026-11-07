#!/usr/bin/env python3
"""The kernel-noise floor of the replay model, from one honest turn, before any surplus is read.

  SWARM_MODEL=qwen3_14b python scripts/kernel_floor.py runs/sweep14b/seed3000/none carol --out results/lane_c/kernel_floor.json

The turn has no peer message, so its input is the same on every pass. Three passes over every token at every lens layer.
  explicit   the explicit 4D causal mask, the kernel both replay passes use
  again      the same, run a second time, which should be bit-identical
  plain      no mask, the plain kernel
Kernel noise is the relative residual delta per token, ||explicit - plain|| / ||plain||, by layer. With --masked RUN AGENT the
mask effect on a turn that has a peer message is added beside it, explicit actual against the masked reference over every token
after the first masked span, the same quantity 32B's noise-floor table reports.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]
from detector.render import find_turn, load_turns, render_turn  # noqa: E402
from replay import LENS_LAYERS, MODEL, Capture, attention_mask_4d, load_model  # noqa: E402


def relative(a, b):
    """[tokens, layers] relative delta, computed one layer at a time to keep memory flat."""
    return np.stack([np.linalg.norm(a[:, l] - b[:, l], axis=-1)/np.linalg.norm(b[:, l], axis=-1) for l in range(a.shape[1])], axis=1)


def summary(delta):
    return {"median": np.median(delta, axis=0).tolist(), "p95": np.percentile(delta, 95, axis=0).tolist(),
            "max": delta.max(axis=0).tolist()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("agent")
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--phase", default="solve")
    parser.add_argument("--masked", nargs=2, metavar=("RUN_DIR", "AGENT"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model, tok = load_model()
    capture = Capture(model, LENS_LAYERS)
    r = render_turn(tok, find_turn(load_turns(args.run_dir), args.agent, args.episode, args.phase))
    if r["peer_token_spans"]:
        raise ValueError("the floor turn must have nothing to mask")
    n = len(r["ids"])
    every = list(range(n))
    explicit = capture.run(model, r["ids"], every, attention_mask_4d(n)).numpy()
    again = capture.run(model, r["ids"], every, attention_mask_4d(n)).numpy()
    plain = capture.run(model, r["ids"], every, None).numpy()
    noise = relative(explicit, plain)
    out = {"model": MODEL, "layers": list(LENS_LAYERS), "turn": {"run": str(args.run_dir), "agent": args.agent, "phase": args.phase, "tokens": n},
           "repeat_bit_identical": bool((explicit == again).all()), "repeat_max_abs": float(np.abs(explicit - again).max()),
           "kernel_noise": summary(noise),
           "kernel_noise_at": {name: noise[p].tolist() for name, p in r["positions"].items()}}
    del again
    if args.masked:
        m = render_turn(tok, find_turn(load_turns(Path(args.masked[0])), args.masked[1], args.episode, args.phase))
        after = list(range(m["peer_token_spans"][0][1], len(m["ids"])))
        actual = capture.run(model, m["ids"], after, attention_mask_4d(len(m["ids"]))).numpy()
        reference = capture.run(model, m["ids"], after, attention_mask_4d(len(m["ids"]), m["mask"])).numpy()
        out["mask_effect"] = {"run": args.masked[0], "agent": args.masked[1], "tokens_after_span": len(after),
                              **summary(relative(actual, reference))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    show = [l for l in (0, 6, 9, 16, 18, 22, 25, 28, 31, 34, 37, 38) if l in LENS_LAYERS]
    print("repeat bit-identical", out["repeat_bit_identical"], "tokens", n)
    for l in show:
        line = f"L{l:2d} noise median {out['kernel_noise']['median'][l]:.4f} p95 {out['kernel_noise']['p95'][l]:.4f}"
        if args.masked:
            line += f" mask effect median {out['mask_effect']['median'][l]:.4f}"
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
