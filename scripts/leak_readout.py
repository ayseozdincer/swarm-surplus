#!/usr/bin/env python3
"""W3's first result. Reads a replay of the leak branch and its matched branches, writes results/leak_readout.json.

Part 1, the blind agent's solve turn. Qwen3 splits numbers into single digits, so the planted 73 can only be read as the
tokens 7 and 3. For each position and layer: is the token in the top-10 readout, what is its rank over the full vocabulary,
and what share and surplus does it carry. All ten digits are reported, so a rise in digits generally can be told from a rise
in 7 and 3 specifically.

Part 2, every free agent's verdict turn: top-10 readout at the first generated token over the band, and surplus concepts.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]
from detector import jspace  # noqa: E402
from replay import LENS_FILE, MODEL_DIR  # noqa: E402

BAND = (26, 27, 28, 29, 30)
POSITIONS = ("first_generated", "prompt_last", "after_think")
DIGITS = "0123456789"


class Shares:
    """J-space shares of one run directory, addressed by file, pass and position."""

    def __init__(self, directory: Path):
        meta = json.loads((directory/"rows.json").read_text())
        data = np.load(directory/"jspace.npz")
        self.layers = meta["layers"]
        self.index = {(r["file"], r["pass"], r["position"]): i for i, r in enumerate(meta["rows"])}
        self.support, self.coefficients = data["support"], data["coefficients"]

    def z(self, file: str, label: str, position: str) -> list[dict[int, float]]:
        i = self.index.get((file, label, position))
        if i is None:
            i = self.index[(file, "actual", position)]
        return jspace.shares(self.support[i], self.coefficients[i])


def digit_table(lens, tok, H_actual, H_reference, layers, targets):
    """Per layer: rank and logit of each digit token in the actual and reference readouts."""
    ids = {d: tok.encode(d, add_special_tokens=False)[0] for d in targets}
    rows = []
    for li, layer in enumerate(layers):
        la = lens.logits(H_actual[None, li], layer)[0]
        lr = lens.logits(H_reference[None, li], layer)[0]
        row = {"layer": layer}
        for d, t in ids.items():
            row[d] = {"rank_actual": int((la > la[t]).sum()), "rank_reference": int((lr > lr[t]).sum()),
                      "logit_delta": float(la[t] - lr[t])}
        rows.append(row)
    return rows


def solve_turn(replay: Path, branch: str, agent: str, episode: int, lens, tok, positions=POSITIONS):
    directory = replay/branch
    index = json.loads((directory/"index.json").read_text())
    turn = next(t for t in index["turns"] if t["agent"] == agent and t["phase"] == "solve")
    data = np.load(directory/turn["file"])
    shares = Shares(directory)
    names = list(turn["positions"])
    digits = {d: tok.encode(d, add_special_tokens=False)[0] for d in DIGITS}
    result = {"branch": branch, "tokens": turn["tokens"], "masked_tokens": turn["masked_tokens"], "positions": {}}
    for position in positions:
        if position not in names:
            continue
        n = names.index(position)
        H_a, H_r = torch.from_numpy(data["actual"][n]), torch.from_numpy(data["reference"][n])
        top_a, top_r = data["actual_top_ids"][n], data["reference_top_ids"][n]
        z_a, z_r = shares.z(turn["file"], "actual", position), shares.z(turn["file"], "reference", position)
        per_digit = {}
        for d, t in digits.items():
            in_a = [l for li, l in enumerate(index["layers"]) if t in top_a[li]]
            in_r = [l for li, l in enumerate(index["layers"]) if t in top_r[li]]
            surplus_layers = [l for li, l in enumerate(index["layers"]) if z_a[li].get(t, 0) > z_r[li].get(t, 0)]
            per_digit[d] = {"top10_actual": in_a, "top10_reference": in_r, "surplus_layers": surplus_layers,
                            "mean_surplus": float(np.mean([max(z_a[li].get(t, 0) - z_r[li].get(t, 0), 0) for li in range(len(z_a))]))}
        result["positions"][position] = {"token": turn["position_tokens"][position], "digits": per_digit,
                                         "ranks": digit_table(lens, tok, H_a, H_r, index["layers"], DIGITS)}
    return result


def verdict_turns(replay: Path, branch: str, tok, free_agents):
    directory = replay/branch
    index = json.loads((directory/"index.json").read_text())
    shares = Shares(directory)
    band = [index["layers"].index(l) for l in BAND]
    rows = []
    for turn in index["turns"]:
        if turn["phase"] != "verdict" or turn["agent"] not in free_agents:
            continue
        data = np.load(directory/turn["file"])
        n = list(turn["positions"]).index("first_generated")
        counts = Counter(int(t) for li in band for t in data["actual_top_ids"][n, li])
        z_a, z_r = shares.z(turn["file"], "actual", "first_generated"), shares.z(turn["file"], "reference", "first_generated")
        pooled = defaultdict(float)
        for a, r in zip(z_a, z_r):
            for c, v in jspace.surplus(a, r).items():
                pooled[c] += v/len(z_a)
        rows.append({"agent": turn["agent"], "masked_tokens": turn["masked_tokens"],
                     "top10_band": [tok.decode([t]) for t, _ in counts.most_common(10)],
                     "surplus_top": [(tok.decode([c]), round(v, 5)) for c, v in sorted(pooled.items(), key=lambda x: -x[1])[:10]],
                     "surplus_total": round(sum(pooled.values()), 5)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", type=Path, default=ROOT/"data/replay/pilot/seed1000")
    parser.add_argument("--run", type=Path, default=ROOT/"runs/pilot/seed1000")
    parser.add_argument("--agent", default="carol")
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ROOT/"results/leak_readout.json")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    lens = jspace.Lens(MODEL_DIR, LENS_FILE)
    branches = sorted(p.name for p in args.replay.iterdir() if (p/"jspace.npz").exists())
    out = {"solve": {}, "verdict": {}}
    for branch in branches:
        out["solve"][branch] = solve_turn(args.replay, branch, args.agent, args.episode, lens, tok)
        run = json.loads((args.run/branch/"run.json").read_text())
        agents = run["results"][args.episode]["agents"]
        free = [a for a, r in agents.items() if not r.get("instructed")]
        out["verdict"][branch] = verdict_turns(args.replay, branch, tok, free)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
