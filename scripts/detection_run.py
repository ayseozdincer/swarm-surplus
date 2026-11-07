#!/usr/bin/env python3
"""W7 part B. Score every free agent's solve turn at episode 2 of every replayed sweep1 seed, then the three detection claims
with matched-branch pairing. Writes results/detection_run.json.

Settings fixed in advance from the pilot (seed 1000, part A), which is not one of the scored seeds. Primary position
first_generated, mass over all layers, direction over L50 to L62, where leak separated from the other notes in the pilot.
Every other position and band is in the JSON.

  mass/floor   J-space surplus mass over the same agent's mass in the seed's none branch (plain-kernel pass, noise only)
  content      1 - cos(d, benign_k1 signature without this agent), mean over the band
  source       cos(d, leak signature) - max over the other notes' signatures, each without this agent
  jump         blind agent only: residual delta over floor at mention_pre, floor from none at its own mention_pre
  naming       bonus: best rank of the answer's first token at mention_pre in the actual pass, late layers (lens)

  claim 1  blind agent, leak against none                 uptake against nothing
  claim 2  blind agent, leak against benign_k1, benign_k3  uptake against exposure to a harmless note
  claim 3  the other free readers, leak against benign_k1  exposure without uptake, should sit at chance
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]
from replay import LENS_LAYERS, band, on_model  # noqa: E402
from surplus_table import mass, shares  # noqa: E402

REPLAY = ROOT/"data/replay/sweep1"
RUNS = ROOT/"runs/sweep1"
RESULT = ROOT/"results/detection_run.json"
POSITIONS = ("prompt_last", "first_generated", "first_thought")
BANDS = {"L26-30": band(26, 30), "L50-62": band(50, 62), "all": list(LENS_LAYERS)}   # 32B names, same depth on other models
PRIMARY = ("first_generated", "L50-62")
NOTES = ("leak", "benign_k1", "proposal_k1")
LATE = tuple(sorted({on_model(l) for l in (52, 54, 56, 58, 60, 61, 62)}))


def unit(x):
    return x/np.linalg.norm(x, axis=-1, keepdims=True)


def cos(a, b):
    return np.sum(unit(a)*unit(b), axis=-1)


def turn_of(directory, agent):
    index = json.loads((directory/"index.json").read_text())
    return next((t for t in index["turns"] if t["agent"] == agent and t["phase"] == "solve"), None)


def load_seed(seed_dir: Path):
    """per variant: free agents, d per agent and position, J-space mass per agent and position, mention data."""
    out = {}
    for directory in sorted(p for p in seed_dir.iterdir() if (p/"jspace.npz").exists() and (p/"index.json").exists()):
        run = json.loads((RUNS/seed_dir.name/directory.name/"run.json").read_text())
        agents = run["results"][2]["agents"]
        free = [a for a, r in agents.items() if not r.get("instructed")]
        meta = json.loads((directory/"rows.json").read_text())
        z = np.load(directory/"jspace.npz")
        look = {(r["file"], r["pass"], r["position"]): i for i, r in enumerate(meta["rows"])}
        variant = {"free": free, "blind": [a for a in free if agents[a].get("blind")], "d": {}, "mass": {}, "delta": {}, "turns": {}}
        for agent in free:
            turn = turn_of(directory, agent)
            if turn is None:
                continue
            data = np.load(directory/turn["file"])
            names = list(turn["positions"])
            variant["turns"][agent] = turn
            variant["d"][agent] = {p: data["actual"][names.index(p)] - data["reference"][names.index(p)] for p in names}
            variant["delta"][agent] = {p: np.linalg.norm(variant["d"][agent][p], axis=-1)/np.linalg.norm(data["actual"][names.index(p)], axis=-1) for p in names}
            variant["mass"][agent] = {}
            for p in names:
                i, j = look[(turn["file"], "actual", p)], look[(turn["file"], "reference", p)]
                variant["mass"][agent][p] = mass(shares(z["support"][i], z["coefficients"][i]), shares(z["support"][j], z["coefficients"][j]))
            variant["leak_used"] = {a: bool(agents[a].get("leak_used")) for a in free}
        out[directory.name] = variant
    return out


def signature(seed, note, position, exclude, band):
    v = seed.get(note)
    if not v:
        return None
    rows = [unit(v["d"][a][position][band]) for a in v["d"] if a != exclude and position in v["d"][a]]
    return np.mean(rows, axis=0) if rows else None


def scores(seed, variant, agent, position, band_name, lens=None, answer_token=None):
    band = BANDS[band_name]
    v = seed[variant]
    d = v["d"][agent][position][band]
    floor = seed["none"]["mass"].get(agent, {}).get(position)
    out = {"mass/floor": v["mass"][agent][position]/floor if floor else None}
    benign = signature(seed, "benign_k1", position, agent, band)
    out["content"] = float(1 - np.mean(cos(d, benign))) if benign is not None else None
    sig = {n: signature(seed, n, position, agent, band) for n in NOTES}
    if all(s is not None for s in sig.values()):
        sim = {n: float(np.mean(cos(d, s))) for n, s in sig.items()}
        out["source"] = sim["leak"] - max(sim[n] for n in NOTES if n != "leak")
        out["nearest"] = max(sim, key=sim.get)
    if "mention_pre" in v["delta"][agent] and "mention_pre" in seed["none"]["delta"].get(agent, {}):
        out["jump"] = float(np.mean(v["delta"][agent]["mention_pre"][band]/seed["none"]["delta"][agent]["mention_pre"][band]))
    return out


def auroc(pos, neg):
    pos, neg = [x for x in pos if x is not None], [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5*(p == n) for p in pos for n in neg)
    return wins/(len(pos)*len(neg))


def main() -> int:
    seeds = {p.name: load_seed(p) for p in sorted(REPLAY.glob("seed*")) if (p/"none"/"jspace.npz").exists()}
    seeds = {k: v for k, v in seeds.items() if all(n in v for n in ("none", "leak", "benign_k1"))}
    result = {"seeds": sorted(seeds), "primary": PRIMARY, "rows": [], "claims": {}}
    for seed_name, seed in seeds.items():
        for variant, v in seed.items():
            for agent in v["d"]:
                row = {"seed": seed_name, "variant": variant, "agent": agent, "blind": agent in v["blind"],
                       "leak_used": v.get("leak_used", {}).get(agent, False)}
                for position in POSITIONS:
                    for band in BANDS:
                        for k, x in scores(seed, variant, agent, position, band).items():
                            row[f"{k} | {position} | {band}"] = x
                result["rows"].append(row)
    rows = result["rows"]
    keys = sorted({k for r in rows for k in r if " | " in k and not k.startswith("nearest")})

    def pick(seed, variant, agent):
        return next((r for r in rows if (r["seed"], r["variant"], r["agent"]) == (seed, variant, agent)), None)

    claims = {"1 blind, leak vs none": [], "2 blind, leak vs benign": [], "3 other readers, leak vs benign_k1": []}
    for seed_name, seed in seeds.items():
        blind = seed["leak"]["blind"][0] if seed["leak"]["blind"] else None
        if blind:
            claims["1 blind, leak vs none"].append((pick(seed_name, "leak", blind), pick(seed_name, "none", blind)))
            for b in ("benign_k1", "benign_k3"):
                if b in seed and blind in seed[b]["d"]:
                    claims["2 blind, leak vs benign"].append((pick(seed_name, "leak", blind), pick(seed_name, b, blind)))
        for agent in seed["leak"]["d"]:
            if agent != blind and agent in seed["benign_k1"]["d"]:
                claims["3 other readers, leak vs benign_k1"].append((pick(seed_name, "leak", agent), pick(seed_name, "benign_k1", agent)))
    for name, pairs in claims.items():
        out = {"pairs": len(pairs), "seeds": len(seeds)}
        for key in keys:
            pos = [p.get(key) for p, _ in pairs]
            neg = [n.get(key) for _, n in pairs]
            if all(x is None for x in pos):
                continue
            paired = [(p, n) for p, n in zip(pos, neg) if p is not None and n is not None]
            out[key] = {"auroc": auroc(pos, neg), "paired_wins": sum(p > n for p, n in paired), "paired_n": len(paired),
                        "positive_median": float(np.median([x for x in pos if x is not None])),
                        "negative_median": float(np.median([x for x in neg if x is not None])) if any(x is not None for x in neg) else None}
        result["claims"][name] = out
    result["nearest_accuracy"] = {}
    for position in POSITIONS:
        for band in BANDS:
            hits = [r[f"nearest | {position} | {band}"] == r["variant"] for r in rows if r.get(f"nearest | {position} | {band}") and r["variant"] in NOTES]
            result["nearest_accuracy"][f"{position} | {band}"] = {"accuracy": float(np.mean(hits)) if hits else None, "n": len(hits)}
    RESULT.write_text(json.dumps(result, indent=1, default=float) + "\n")
    p, b = PRIMARY
    for name, out in result["claims"].items():
        print(name, out["pairs"], {k.split(" | ")[0]: (round(v["auroc"], 2) if v["auroc"] is not None else None, f"{v['paired_wins']}/{v['paired_n']}")
                                   for k, v in out.items() if isinstance(v, dict) and (f"| {p} |" in k and (b in k or k.startswith("mass")) or k.startswith("jump"))})
    return 0


if __name__ == "__main__":
    sys.exit(main())
