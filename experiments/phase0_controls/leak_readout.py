#!/usr/bin/env python3
"""W3's first result. Reads a replay of the leak branch and its matched branches, writes experiments/phase0_controls/results/leak_readout.json.

Part 1, the blind agent's solve turn. Qwen3 splits numbers into single digits, so a planted 73 can only be read as the
tokens 7 and 3. For each token of the planted answer, position and layer: is it in the top-10 readout, what is its rank
over the full vocabulary, what share and surplus does it carry, and where does its change between the passes fall among
every token's change. All ten digits are reported as well, so a rise in digits generally can be told from a rise in the
planted ones. The reference is also compared with the `none` branch: with the note masked it should sit close to the
state the agent has when no note was ever written.

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

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]
from detector import jspace  # noqa: E402
from replay import LENS_FILE, MODEL_DIR, band  # noqa: E402

BAND = tuple(band(26, 30))          # NARCBench's Qwen3-32B probing band, at the same depth on other models
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
        i = self.index[(file, label, position)]
        return jspace.shares(self.support[i], self.coefficients[i])


def token_rows(lens, H_actual, H_reference, layers, tokens):
    """Per layer and token: rank in each readout, logit change and its percentile among all tokens' changes."""
    rows = []
    for li, layer in enumerate(layers):
        la = lens.logits(H_actual[None, li], layer)[0]
        lr = lens.logits(H_reference[None, li], layer)[0]
        delta = la - lr
        row = {"layer": layer}
        for name, t in tokens.items():
            row[name] = {"rank_actual": int((la > la[t]).sum()), "rank_reference": int((lr > lr[t]).sum()),
                         "logit_delta": float(delta[t]), "delta_percentile": float((delta < delta[t]).float().mean())}
        rows.append(row)
    return rows


def relative_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)/np.linalg.norm(b, axis=-1)


def solve_turn(replay: Path, branch: str, agent: str, lens, tok, answer: str, positions=POSITIONS):
    directory = replay/branch
    index = json.loads((directory/"index.json").read_text())
    turn = next(t for t in index["turns"] if t["agent"] == agent and t["phase"] == "solve")
    data = np.load(directory/turn["file"])
    shares = Shares(directory)
    names = list(turn["positions"])
    tokens = {d: tok.encode(d, add_special_tokens=False)[0] for d in DIGITS}
    planted = [tok.decode([t]) for t in tok.encode(answer, add_special_tokens=False)]
    tokens.update({p: t for p, t in zip(planted, tok.encode(answer, add_special_tokens=False))})
    result = {"branch": branch, "file": turn["file"], "tokens": turn["tokens"], "masked_tokens": turn["masked_tokens"],
              "answer": answer, "planted_tokens": planted, "positions": {}}
    for position in positions:
        if position not in names:
            continue
        n = names.index(position)
        H_a, H_r = torch.from_numpy(data["actual"][n]), torch.from_numpy(data["reference"][n])
        top_a, top_r = data["actual_top_ids"][n], data["reference_top_ids"][n]
        z_a, z_r = shares.z(turn["file"], "actual", position), shares.z(turn["file"], "reference", position)
        per_token = {}
        for name, t in tokens.items():
            in_a = [l for li, l in enumerate(index["layers"]) if t in top_a[li]]
            in_r = [l for li, l in enumerate(index["layers"]) if t in top_r[li]]
            held = [l for li, l in enumerate(index["layers"]) if t in z_a[li]]
            surplus_layers = [l for li, l in enumerate(index["layers"]) if z_a[li].get(t, 0) > z_r[li].get(t, 0)]
            per_token[name] = {"top10_actual": in_a, "top10_reference": in_r, "jspace_actual": held, "surplus_layers": surplus_layers,
                               "mean_surplus": float(np.mean([max(z_a[li].get(t, 0) - z_r[li].get(t, 0), 0) for li in range(len(z_a))]))}
        result["positions"][position] = {"token": turn["position_tokens"][position], "per_token": per_token,
                                         "ranks": token_rows(lens, H_a, H_r, index["layers"], tokens)}
    return result


def reference_check(replay: Path, agent: str, branches=("leak", "benign_k1", "proposal_k1")):
    """Distance of each branch's actual and reference state from the none branch at the prompt's last token."""
    base_dir = replay/"none"
    index = json.loads((base_dir/"index.json").read_text())
    turn = next(t for t in index["turns"] if t["agent"] == agent and t["phase"] == "solve")
    n = list(turn["positions"]).index("prompt_last")
    base = np.load(base_dir/turn["file"])["actual"][n]
    out = {}
    for branch in branches:
        if not (replay/branch/"index.json").exists():
            continue
        other = json.loads((replay/branch/"index.json").read_text())
        t = next(x for x in other["turns"] if x["agent"] == agent and x["phase"] == "solve")
        m = list(t["positions"]).index("prompt_last")
        data = np.load(replay/branch/t["file"])
        actual, reference = relative_distance(data["actual"][m], base), relative_distance(data["reference"][m], base)
        out[branch] = {"layers": index["layers"], "actual_vs_none": [float(x) for x in actual], "reference_vs_none": [float(x) for x in reference]}
    return out


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
    parser.add_argument("--out", type=Path, default=ROOT/"experiments/phase0_controls/results/leak_readout.json")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    lens = jspace.Lens(MODEL_DIR, LENS_FILE)
    branches = sorted(p.name for p in args.replay.iterdir() if (p/"jspace.npz").exists())
    answer = json.loads((args.run/"leak/run.json").read_text())["results"][args.episode]["agents"][args.agent]["submission"]["answer"]
    out = {"answer": answer, "solve": {}, "verdict": {}, "reference_check": reference_check(args.replay, args.agent)}
    for branch in branches:
        out["solve"][branch] = solve_turn(args.replay, branch, args.agent, lens, tok, answer)
        run = json.loads((args.run/branch/"run.json").read_text())
        agents = run["results"][args.episode]["agents"]
        free = [a for a, r in agents.items() if not r.get("instructed")]
        out["verdict"][branch] = verdict_turns(args.replay, branch, tok, free)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
