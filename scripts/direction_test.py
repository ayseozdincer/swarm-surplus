#!/usr/bin/env python3
"""W7 part A. Lens-free direction test on pilot seed 1000, episode 2, solve turns. Writes results/direction_test.json.

The surplus direction is d = h_actual - h_reference, unit-normalised per layer. `kept` is the reference with positions
kept, `compacted` the one with position ids closed up after the masked span. The compacted gap differs with each note's
length (57, 63 and 87 tokens here), so compacted directions also carry a note-specific position shift. The kept reference
is the cleaner basis.

  collect   (GPU) compacted references for every free reader, and the per-token trace of carol's leak turn from the
            response start to just past the first mention: actual, plain-kernel actual (the floor), kept and compacted
  analyze   the four measurements, from whatever has been collected

  1  carol: cos(d_leak, d_benign), cos(d_leak, d_proposal), cos(d_benign, d_proposal) per layer
  2  within one note, mean cosine across readers; for one reader, mean cosine across notes
  3  signature of a note = mean d of the other four readers; is a reader's d closest to its own note's signature
  4  carol at the first mention against each note's signature at first_generated, and the trace up to the mention
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]

RUN = ROOT/"runs/pilot/seed1000"
REPLAY = ROOT/"data/replay/pilot/seed1000"
MENTION = ROOT/"data/replay_mention/pilot/seed1000"
OUT = ROOT/"data/direction"
RESULT = ROOT/"results/direction_test.json"
BRANCHES = ("leak", "benign_k1", "proposal_k1")
READERS = ("alice", "bob", "carol", "dave", "erin")
POSITIONS = ("prompt_last", "first_generated", "first_thought")
BANDS = {"L26-30": range(26, 31), "L50-62": range(50, 63), "all": range(63)}
SHOW = (15, 26, 30, 36, 40, 45, 50, 56, 60, 62)


def unit(x):
    return x/np.linalg.norm(x, axis=-1, keepdims=True)


def collect(args):
    import torch
    from detector.render import compacted_positions, find_turn, load_turns, render_turn
    from replay import LENS_LAYERS, Capture, attention_mask_4d, load_model
    model, tok = load_model()
    capture = Capture(model, LENS_LAYERS)
    OUT.mkdir(parents=True, exist_ok=True)
    for branch in BRANCHES:
        for agent in READERS:
            r = render_turn(tok, find_turn(load_turns(RUN/branch), agent, 2, "solve"))
            positions = [r["positions"][p] for p in POSITIONS]
            h = capture.run(model, r["ids"], positions, attention_mask_4d(len(r["ids"]), r["mask"]), position_ids=compacted_positions(r["mask"]))
            np.save(OUT/f"compacted_{branch}_{agent}.npy", h.numpy())
            print(branch, agent, flush=True)
    r = render_turn(tok, find_turn(load_turns(RUN/"leak"), "carol", 2, "solve"))
    span = list(range(r["positions"]["first_generated"], r["positions"]["mention_0"] + 3))
    n = len(r["ids"])
    passes = {"actual": capture.run(model, r["ids"], span, attention_mask_4d(n)),
              "plain": capture.run(model, r["ids"], span, None),
              "kept": capture.run(model, r["ids"], span, attention_mask_4d(n, r["mask"])),
              "compacted": capture.run(model, r["ids"], span, attention_mask_4d(n, r["mask"]), position_ids=compacted_positions(r["mask"]))}
    np.savez(OUT/"trace_carol_leak.npz", positions=np.array(span), tokens=np.array([tok.decode([r["ids"][i]]) for i in span]),
             mention_pre=r["positions"]["mention_pre"], **{k: v.numpy() for k, v in passes.items()})
    print("trace", len(span), flush=True)


def saved(directory: Path, agent: str):
    index = json.loads((directory/"index.json").read_text())
    turn = next(t for t in index["turns"] if t["agent"] == agent and t["phase"] == "solve")
    return turn, np.load(directory/turn["file"])


def directions(kind: str) -> dict:
    """d[branch][agent][position] -> [63, d_model] unit vectors."""
    out = {}
    for branch in BRANCHES:
        out[branch] = {}
        for agent in READERS:
            turn, data = saved(REPLAY/branch, agent)
            names = list(turn["positions"])
            idx = [names.index(p) for p in POSITIONS]
            actual = data["actual"][idx]
            if kind == "kept":
                reference = data["reference"][idx]
            else:
                path = OUT/f"compacted_{branch}_{agent}.npy"
                if not path.exists():
                    return {}
                reference = np.load(path)
            out[branch][agent] = {p: unit(actual[i] - reference[i]) for i, p in enumerate(POSITIONS)}
    return out


def cos(a, b):
    return np.sum(unit(a)*unit(b), axis=-1)


def analyze_kind(d):
    result = {"carol_pairs": {}, "agreement": {}, "source": {}}
    pairs = (("leak", "benign_k1"), ("leak", "proposal_k1"), ("benign_k1", "proposal_k1"))
    for p in POSITIONS:
        c = d["leak"]["carol"]
        result["carol_pairs"][p] = {f"{a} vs {b}": cos(d[a]["carol"][p], d[b]["carol"][p]).tolist() for a, b in pairs}
        result["agreement"][p], result["source"][p] = {}, {}
        for band, layers in BANDS.items():
            L = list(layers)
            within = [float(np.mean(cos(d[b][i][p][L], d[b][j][p][L]))) for b in BRANCHES for i in READERS for j in READERS if i < j]
            across = [float(np.mean(cos(d[a][i][p][L], d[b][i][p][L]))) for i in READERS for a, b in pairs]
            result["agreement"][p][band] = {"within_note_across_readers": float(np.mean(within)), "within_sd": float(np.std(within)),
                                            "same_reader_across_notes": float(np.mean(across)), "across_sd": float(np.std(across))}
            hits = []
            for i in READERS:
                for b in BRANCHES:
                    scores = {}
                    for s in BRANCHES:
                        signature = np.mean([d[s][j][p][L] for j in READERS if j != i], axis=0)
                        scores[s] = float(np.mean(cos(d[b][i][p][L], signature)))
                    hits.append(max(scores, key=scores.get) == b)
            result["source"][p][band] = {"accuracy": float(np.mean(hits)), "n": len(hits), "chance": 1/3}
    return result


def uptake(d, kind):
    """carol's d at mention_pre against the leak and benign signatures at first_generated, per layer."""
    out = {}
    signature = {b: np.mean([d[b][j]["first_generated"] for j in READERS if j != "carol"], axis=0) for b in BRANCHES}
    for branch in ("leak", "benign_k1"):
        turn, data = saved(MENTION/branch, "carol")
        n = list(turn["positions"]).index("mention_pre")
        reference = data["reference"][n]
        if kind == "compacted":
            return None
        v = data["actual"][n] - reference
        out[branch] = {f"vs {s} signature": cos(v, signature[s]).tolist() for s in BRANCHES}
    return out


def trace(d):
    path = OUT/"trace_carol_leak.npz"
    if not path.exists():
        return None
    t = np.load(path)
    signature = {b: np.mean([d[b][j]["first_generated"] for j in READERS if j != "carol"], axis=0) for b in BRANCHES}
    positions, tokens = t["positions"].tolist(), t["tokens"].tolist()
    mention = int(t["mention_pre"])
    rows = []
    for k, position in enumerate(positions):
        row = {"position": position, "offset": position - positions[0], "token": tokens[k], "near_mention": abs(position - mention) <= 2}
        floor = np.linalg.norm(t["actual"][k] - t["plain"][k], axis=-1)/np.linalg.norm(t["actual"][k], axis=-1)
        for kind in ("kept", "compacted"):
            delta = np.linalg.norm(t["actual"][k] - t[kind][k], axis=-1)/np.linalg.norm(t["actual"][k], axis=-1)
            v = t["actual"][k] - t[kind][k]
            for band, layers in BANDS.items():
                L = list(layers)
                row[f"{kind} delta/floor {band}"] = float(np.mean(delta[L]/floor[L]))
                for s in BRANCHES:
                    row[f"{kind} cos {s} {band}"] = float(np.mean(cos(v[L], signature[s][L])))
        rows.append(row)
    return rows


def analyze(args):
    result = {"readers": READERS, "branches": BRANCHES, "positions": POSITIONS, "bands": {k: [min(v), max(v)] for k, v in BANDS.items()}}
    for kind in ("kept", "compacted"):
        d = directions(kind)
        if not d:
            result[kind] = None
            continue
        result[kind] = analyze_kind(d)
        result[kind]["uptake"] = uptake(d, kind)
    kept = directions("kept")
    result["trace"] = trace(kept)
    RESULT.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", RESULT, "compacted", result["compacted"] is not None, "trace", result["trace"] is not None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("collect", "analyze"))
    args = parser.parse_args()
    {"collect": collect, "analyze": analyze}[args.action](args)
