#!/usr/bin/env python3
"""Lane C. Does the position-blind trace measure uptake, or engagement with peer content, and does the direction carry uptake
or only attribution? Writes experiments/lane_c_sweep14b/results/users.json.

  SWARM_MODEL=qwen3_14b python experiments/lane_c_sweep14b/users.py runs/sweep14b/seed3??? --floor experiments/lane_c_sweep14b/results/floor.json
  python experiments/lane_c_sweep14b/users.py --analyze-only        (CPU, from the saved arrays)

Turns. Every free agent's verdict turn of episode 2 in leak and benign_k1 with a peer note masked. Roles: `user`, the blind
agent, who used the leak at its solve turn; `non-user`, the other agents free in the leak branch, exposed to the leak and not
using it; `other`, agents free only in benign_k1 (the leak's author), kept so every note signature has all its readers. Every
reply token is read in an actual and a kept-reference pass on the explicit 4D-mask kernel, as experiments/lane_c_sweep14b/blind.py.
Saved per turn: delta_reference [tokens, layers], the reference norm, and d = actual - reference for the first FIRST reply
tokens from first_generated at L16 to L38, float16.

Scores per turn, delta over floor averaged over L16 to L38, the floor the stage-1 one:
  first256 count    reply tokens above 10 times the floor among the first 256, the same length in every turn
  window 16         the largest 16-token sliding mean over the first 256; its window is the plateau
  drift min         the smallest 16-token sliding mean over the first 256 of cos(d_t, s_note), s_note the note's exposure
                    signature, the mean unit d at first_generated of the other readers of that note in that seed
  cos at plateau    the mean of cos(d_t, s_note) over the plateau window
  orthogonal        at the plateau, the norm of d minus its projection on s_note, over the reference norm, over the floor
  parallel          the same for the projection
Direction scores are per layer, then averaged over the late band (L31 to L38, 32B's L50 to L62) or the mid band (L16 to L18).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]

TRACE_BAND = list(range(16, 39))
DIRECTION_BANDS = {"late": list(range(31, 39)), "mid": [16, 17, 18]}
FIRST = 256
W = 16
EPISODE = 2


def unit(x):
    return x/np.linalg.norm(x, axis=-1, keepdims=True)


def windows(series, w=W):
    w = min(w, len(series))
    return np.convolve(series, np.ones(w)/w, mode="valid"), w


def auroc(pos, neg):
    pos, neg = [x for x in pos if x is not None], [x for x in neg if x is not None]
    return sum((p > n) + 0.5*(p == n) for p in pos for n in neg)/(len(pos)*len(neg)) if pos and neg else None


def replay(args, floor):
    from detector.render import find_turn, load_turns, render_turn
    from blind import GpuCapture, relative
    from replay import LENS_LAYERS, attention_mask_4d, load_model
    model, tok = load_model()
    capture = GpuCapture(model, list(LENS_LAYERS))
    rows = []
    for seed_dir in args.seed_dirs:
        leak = json.loads((seed_dir/"leak/run.json").read_text())["results"][EPISODE]["agents"]
        blind = next(a for a, x in leak.items() if x.get("blind"))
        answer = leak[blind]["submission"]["answer"]
        pattern = re.compile(rf"(?<![0-9A-Za-z]){re.escape(answer)}(?![0-9A-Za-z])")
        for variant in ("leak", "benign_k1"):
            agents = json.loads((seed_dir/variant/"run.json").read_text())["results"][EPISODE]["agents"]
            turns = load_turns(seed_dir/variant)
            for agent in sorted(a for a, x in agents.items() if not x.get("instructed")):
                r = render_turn(tok, find_turn(turns, agent, EPISODE, "verdict"))
                if not r["peer_token_spans"]:
                    continue
                started = time.perf_counter()
                ids, n = r["ids"], len(r["ids"])
                reply = list(range(r["positions"]["prompt_last"], n))
                A = capture.run(model, ids, reply, attention_mask_4d(n))
                R = capture.run(model, ids, reply, attention_mask_4d(n, r["mask"]))
                delta = np.stack([relative(A[:, li], R[:, li]) for li in range(len(LENS_LAYERS))], axis=1)
                rnorm = R.float().norm(dim=-1).cpu().numpy()
                start = r["positions"]["first_generated"] - reply[0]
                d = (A[start:start + FIRST][:, TRACE_BAND].float() - R[start:start + FIRST][:, TRACE_BAND].float()).half().cpu().numpy()
                del A, R
                hit = pattern.search(r["text"], r["prompt_chars"])
                offset = None
                if hit:
                    encoded = tok(r["text"], add_special_tokens=False, return_offsets_mapping=True)
                    offset = next(i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a <= hit.start() < b) - r["positions"]["first_generated"]
                role = "user" if agent == blind else "non-user" if not leak.get(agent, {}).get("instructed") else "other"
                out = args.out_root/seed_dir.name/variant
                out.mkdir(parents=True, exist_ok=True)
                np.savez(out/f"{agent}_e{EPISODE}_verdict_trace.npz", start=start, delta_reference=delta, rnorm=rnorm, d=d)
                row = {"seed": seed_dir.name, "variant": variant, "agent": agent, "role": role, "tokens": n,
                       "masked_tokens": len(r["mask"]) - sum(r["mask"]), "answer_offset": offset, "file": str(out/f"{agent}_e{EPISODE}_verdict_trace.npz"),
                       "seconds": round(time.perf_counter() - started, 1)}
                rows.append(row)
                print(row["seed"], variant, agent, role, n, row["seconds"], flush=True)
    (args.out_root/"rows.json").write_text(json.dumps(rows, indent=1))
    return rows


def analyze(args, floor, rows):
    data = {(r["seed"], r["variant"], r["agent"]): np.load(r["file"]) for r in rows}
    # every reader's unit d at first_generated, per layer of TRACE_BAND
    first = {k: unit(v["d"][0].astype(np.float32)) for k, v in data.items()}
    out = []
    for r in rows:
        key = (r["seed"], r["variant"], r["agent"])
        z = data[key]
        start = int(z["start"])
        ratio = z["delta_reference"][start:start + FIRST]/floor[None, :]
        series = ratio[:, TRACE_BAND].mean(1)
        means, w = windows(series)
        i = int(means.argmax())
        plateau = list(range(i, i + w))
        row = {**{k: r[k] for k in ("seed", "variant", "agent", "role", "answer_offset")},
               "first256 count": int((series > 10).sum()), "window 16": float(means[i]), "plateau centre": i + w//2}
        if r["answer_offset"] is not None:
            row["plateau centre - answer"] = i + w//2 - r["answer_offset"]
        others = [first[k] for k in first if k[:2] == key[:2] and k[2] != r["agent"]]
        d = z["d"].astype(np.float32)
        rnorm = z["rnorm"][start:start + FIRST][:, TRACE_BAND]
        if others:
            s = unit(np.mean(others, axis=0))                                    # [band layers, d_model]
            for name, layers in DIRECTION_BANDS.items():
                idx = [TRACE_BAND.index(l) for l in layers]
                c = np.sum(unit(d[:, idx])*s[None, idx], axis=-1).mean(1)          # cos per token, band mean
                cm, _ = windows(c)
                row[f"drift min | {name}"] = float(cm.min())
                row[f"cos at plateau | {name}"] = float(c[plateau].mean())
                par = np.sum(d[plateau][:, idx]*s[None, idx], axis=-1)            # [w, layers]
                perp = np.linalg.norm(d[plateau][:, idx] - par[..., None]*s[None, idx], axis=-1)
                fl = floor[layers][None, :]
                row[f"orthogonal | {name}"] = float((perp/rnorm[plateau][:, idx]/fl).mean())
                row[f"parallel | {name}"] = float((np.abs(par)/rnorm[plateau][:, idx]/fl).mean())
        out.append(row)

    keys = [k for k in out[0] if k in ("first256 count", "window 16") or " | " in k]
    def group(variant, role):
        return {(x["seed"], x["agent"]): x for x in out if x["variant"] == variant and x["role"] == role}
    lu, ln, bu, bn = group("leak", "user"), group("leak", "non-user"), group("benign_k1", "user"), group("benign_k1", "non-user")
    comparisons = {}
    for k in keys:
        # drift min: a user should drift away, so lower scores more uptake; scored negated so that 1.0 means users lower
        sign = -1 if k.startswith(("drift min", "cos at plateau")) else 1
        cell = {}
        p, q = [sign*x[k] for x in lu.values() if k in x], [sign*x[k] for x in ln.values() if k in x]
        cell["users vs non-users, leak"] = {"auroc": auroc(p, q), "n": [len(p), len(q)],
                                            "medians": [float(np.median([x[k] for x in lu.values()])), float(np.median([x[k] for x in ln.values()]))]}
        m = sorted(set(ln) & set(bn))
        p, q = [sign*ln[x][k] for x in m if k in ln[x] and k in bn[x]], [sign*bn[x][k] for x in m if k in ln[x] and k in bn[x]]
        cell["non-users, leak vs benign_k1"] = {"auroc": auroc(p, q), "paired_wins": sum(a > b for a, b in zip(p, q)), "pairs": len(p),
                                                "medians": [float(np.median([ln[x][k] for x in m])), float(np.median([bn[x][k] for x in m]))]}
        m = sorted(set(lu) & set(bu))
        p, q = [sign*lu[x][k] for x in m], [sign*bu[x][k] for x in m]
        cell["users, leak vs benign_k1"] = {"auroc": auroc(p, q), "paired_wins": sum(a > b for a, b in zip(p, q)), "pairs": len(p),
                                            "medians": [float(np.median([lu[x][k] for x in m])), float(np.median([bu[x][k] for x in m]))]}
        cell["sign"] = sign
        comparisons[k] = cell
    result = {"rows": out, "comparisons": comparisons, "first": FIRST, "window": W, "trace_band": [16, 38], "direction_bands": DIRECTION_BANDS,
              "non-users writing the answer, leak": sum(x["answer_offset"] is not None for x in ln.values()), "non-users, leak": len(ln)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    for k, cell in comparisons.items():
        print(k, {c: f"{v['auroc']:.2f} {v['medians'][0]:.3g}/{v['medians'][1]:.3g}" for c, v in cell.items() if isinstance(v, dict) and v["auroc"] is not None}, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seed_dirs", nargs="*", type=Path)
    parser.add_argument("--floor", type=Path, default=ROOT/"experiments/lane_c_sweep14b/results/floor.json")
    parser.add_argument("--out-root", type=Path, default=ROOT/"data/replay_14b/users/sweep14b")
    parser.add_argument("--out", type=Path, default=ROOT/"experiments/lane_c_sweep14b/results/users.json")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    floor = np.array(json.loads(args.floor.read_text())["floor"])
    rows = json.loads((args.out_root/"rows.json").read_text()) if args.analyze_only else replay(args, floor)
    analyze(args, floor, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
