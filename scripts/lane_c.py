#!/usr/bin/env python3
"""Lane C. The 32B W3 and W7 readouts over every seed of a planted-channel sweep, on the model SWARM_MODEL names
(here Qwen3-14B with its converged lens). Writes one JSON, results/lane_c/readout.json by default.

  SWARM_MODEL=qwen3_14b python scripts/lane_c.py --replay data/replay_14b/sweep14b --runs runs/sweep14b

Episode 2 of every seed. The blind agent is the free agent run.json marks blind; in the leak branch it used the note.
`kept` is the reference with positions kept, `compacted` the one with position ids closed up after the masked span.
Bands are the 32B ones at the same relative depth (replay.band), named by their 32B layers.

  floor        kernel noise at the read positions (none turns, whose second pass is the plain kernel) against the mask
               effect on the blind agent's leak turn, per layer
  leak         W3 table 1. The planted answer's tokens at prompt_last, first_generated, after_think, and answer_pre as the
               positive control: top-10 layers, J-space layers, surplus layers, best rank over layers, per seed
  digits       W3 table 2, numeric answers only. Best rank of every digit at after_think
  mass         W3 surplus mass by branch and position, median over the free agents' solve and verdict turns
  recovery     distance of the blind agent's actual, kept and compacted states from the none branch's, per layer
  compacted    W3 follow-up 2. Is the compacted reference within the kernel floor of none
  verdict      W3's verdict-turn table. Top-10 over the mid band and the top surplus concepts at first_generated
  mention      W3 follow-up 1. The blind agent at mention_pre and first_generated, per layer
  direction    W7 part A on the solve turns of the free readers, kept reference (and compacted)
  detection    W7 part B's three claims, matched-branch pairing over seeds
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]
from replay import LENS_FILE, LENS_LAYERS, MODEL, MODEL_DIR, band, on_model  # noqa: E402
from surplus_table import mass, shares  # noqa: E402

EPISODE = 2
BANDS = {"L26-30": band(26, 30), "L50-62": band(50, 62), "all": list(LENS_LAYERS)}
LATE = sorted({on_model(l) for l in (52, 54, 56, 58, 60, 61, 62)})
SHOW = sorted({on_model(l) for l in (0, 15, 26, 30, 36, 40, 45, 50, 52, 54, 56, 58, 60, 61, 62)})
MASS_POSITIONS = ("prompt_last", "first_generated", "first_thought", "after_think", "answer_pre", "verdict_pre", "end_of_response")
LEAK_POSITIONS = ("first_generated", "after_think", "prompt_last", "answer_pre", "mention_pre")
DIRECTION_POSITIONS = ("prompt_last", "first_generated", "first_thought")
NOTES = ("leak", "benign_k1", "proposal_k1")
VARIANTS = ("leak", "none", "benign_k1", "benign_k3", "proposal_k1", "proposal_k3")
DIGITS = "0123456789"
REFS = ("kept", "compacted")


def unit(x):
    return x/np.linalg.norm(x, axis=-1, keepdims=True)


def cos(a, b):
    return np.sum(unit(a)*unit(b), axis=-1)


def relative(a, b):
    return np.linalg.norm(a - b, axis=-1)/np.linalg.norm(b, axis=-1)


class Branch:
    """One replayed variant run: its turns, residuals by pass and position, J-space shares, who is free and blind."""

    def __init__(self, directory: Path, run_dir: Path):
        self.directory = directory
        self.index = json.loads((directory/"index.json").read_text())
        self.layers = self.index["layers"]
        agents = json.loads((run_dir/"run.json").read_text())["results"][EPISODE]["agents"]
        self.agents = agents
        self.free = [a for a, r in agents.items() if not r.get("instructed")]
        self.blind = next((a for a in self.free if agents[a].get("blind")), None)
        self.submission = {a: (r.get("submission") or {}).get("answer") for a, r in agents.items()}
        self.turns = {(t["agent"], t["phase"]): t for t in self.index["turns"]}
        self.z_index = None
        if (directory/"jspace.npz").exists():
            meta = json.loads((directory/"rows.json").read_text())
            data = np.load(directory/"jspace.npz")
            self.support, self.coefficients = data["support"], data["coefficients"]
            self.z_index = {(r["file"], r["pass"], r["position"]): i for i, r in enumerate(meta["rows"])}
        self._arrays = {}

    def turn(self, agent, phase="solve"):
        return self.turns.get((agent, phase))

    def exposed(self, agent, phase="solve"):
        t = self.turn(agent, phase)
        return bool(t and t["masked_tokens"] > 0)

    def array(self, agent, phase, name):
        """One array of a turn's .npz, decompressed once."""
        key = (agent, phase)
        if key not in self._arrays:
            self._arrays[key] = [np.load(self.directory/self.turns[key]["file"]), {}]
        source, loaded = self._arrays[key]
        if name not in loaded:
            loaded[name] = source[name] if name in source.files else None
        return loaded[name]

    def h(self, agent, phase, label, position):
        """[layers, d_model] residual of a pass at a named position; `kept` is the stored reference."""
        t = self.turn(agent, phase)
        if t is None or position not in t["positions"]:
            return None
        a = self.array(agent, phase, {"kept": "reference"}.get(label, label))
        return None if a is None else a[list(t["positions"]).index(position)]

    def top_ids(self, agent, phase, label, position):
        t = self.turn(agent, phase)
        if t is None or position not in t["positions"]:
            return None
        name = {"kept": "reference"}.get(label, label)
        return self.array(agent, phase, f"{name}_top_ids")[list(t["positions"]).index(position)]

    def z(self, agent, phase, label, position):
        """J-space shares per layer, or None when that row was not decomposed. An unmasked turn's compacted pass is its
        reference."""
        t = self.turn(agent, phase)
        if t is None or self.z_index is None:
            return None
        name = {"kept": "reference"}.get(label, label)
        if name == "compacted" and t["masked_tokens"] == 0:
            name = "reference"
        i = self.z_index.get((t["file"], name, position))
        return None if i is None else shares(self.support[i], self.coefficients[i])

    def mass(self, agent, phase, position, ref="kept"):
        z_a, z_r = self.z(agent, phase, "actual", position), self.z(agent, phase, ref, position)
        return None if z_a is None or z_r is None else mass(z_a, z_r)

    def d(self, agent, phase, position, ref="kept"):
        a, r = self.h(agent, phase, "actual", position), self.h(agent, phase, ref, position)
        return None if a is None or r is None else a - r


class Ranks:
    """Lens ranks over the full vocabulary, every J on the GPU once."""

    def __init__(self):
        import torch
        from detector import jspace
        self.torch = torch
        self.lens = jspace.Lens(MODEL_DIR, LENS_FILE)
        self.lens.J = {l: J.to("cuda").float() for l, J in self.lens.J.items()}

    def __call__(self, H, layers, tokens):
        """[layers, len(tokens)] rank of each token in the readout of H [layers, d]."""
        out = np.zeros((len(layers), len(tokens)), dtype=np.int64)
        t = self.torch.as_tensor(tokens, device="cuda")
        for li, layer in enumerate(layers):
            logits = self.lens.logits(self.torch.from_numpy(np.ascontiguousarray(H[li]))[None], layer)[0]
            out[li] = (logits[None, :] > logits[t][:, None]).sum(1).cpu().numpy()
        return out


def answer_tokens(tok, answer):
    ids = tok.encode(answer, add_special_tokens=False)
    return list(dict.fromkeys(ids))[:8]


def load(replay: Path, runs: Path):
    seeds = {}
    for seed_dir in sorted(p for p in replay.iterdir() if p.name.startswith("seed")):
        branches = {}
        for v in VARIANTS:
            if (seed_dir/v/"index.json").exists() and (runs/seed_dir.name/v/"run.json").exists():
                branches[v] = Branch(seed_dir/v, runs/seed_dir.name/v)
        if all(v in branches for v in ("leak", "none", "benign_k1")):
            seeds[seed_dir.name] = branches
    return seeds


# ── W3 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

def floor_table(seeds):
    """Kernel noise (none turns) against the mask effect (blind agent's leak turn) at the read positions, per layer."""
    out = {}
    for position in ("prompt_last", "first_generated", "first_thought", "after_think", "mention_pre"):
        noise = [relative(b.h(a, "solve", "actual", position), b.h(a, "solve", "kept", position))
                 for s in seeds.values() for b in [s["none"]] for a in b.free if b.h(a, "solve", "actual", position) is not None]
        effect = [relative(s["leak"].h(s["leak"].blind, "solve", "actual", position), s["leak"].h(s["leak"].blind, "solve", "kept", position))
                  for s in seeds.values() if s["leak"].h(s["leak"].blind, "solve", "actual", position) is not None]
        out[position] = {"noise_turns": len(noise), "effect_turns": len(effect),
                         "noise_median": np.median(noise, axis=0).tolist() if noise else None,
                         "effect_median": np.median(effect, axis=0).tolist() if effect else None}
    return out


def leak_table(seeds, ranks, tok):
    rows = {}
    for name, s in seeds.items():
        leak = s["leak"]
        blind = leak.blind
        answer = leak.submission[blind]
        planted = answer_tokens(tok, answer)
        first = planted[0]
        row = {"blind": blind, "answer": answer, "planted": [tok.decode([t]) for t in planted], "numeric": answer.strip().isdigit(),
               "leak_used": bool(leak.agents[blind].get("leak_used")), "positions": {}}
        for position in LEAK_POSITIONS:
            if leak.h(blind, "solve", "actual", position) is None:
                row["positions"][position] = None
                continue
            cell = {"token": leak.turn(blind)["position_tokens"][position]}
            for label in ("actual", "kept"):
                top = leak.top_ids(blind, "solve", label, position)
                cell[f"top10_layers_{label}"] = [l for li, l in enumerate(leak.layers) if set(top[li].tolist()) & set(planted)]
            z_a = leak.z(blind, "solve", "actual", position)
            if z_a is not None:
                cell["jspace_layers"] = [l for li, l in enumerate(leak.layers) if any(t in z_a[li] for t in planted)]
                for ref in REFS:
                    z_r = leak.z(blind, "solve", ref, position)
                    cell[f"surplus_layers_{ref}"] = [l for li, l in enumerate(leak.layers) if any(z_a[li].get(t, 0) > z_r[li].get(t, 0) for t in planted)]
                    cell[f"surplus_first_{ref}"] = [max(z_a[li].get(first, 0) - z_r[li].get(first, 0), 0) for li in range(len(z_a))]
            best = {}
            for branch, label in (("leak", "actual"), ("leak", "kept"), ("leak", "compacted"), ("none", "actual"), ("benign_k1", "actual"), ("benign_k1", "kept")):
                b = s[branch]
                H = b.h(b.blind, "solve", label, position)
                if H is None:
                    continue
                r = ranks(H, b.layers, [first])[:, 0]
                best[f"{branch} {label}"] = {"best": int(r.min()), "best_layer": int(b.layers[int(r.argmin())]),
                                             "best_late": int(min(r[b.layers.index(l)] for l in LATE)), "by_layer": r.tolist()}
            cell["rank_first"] = best
            row["positions"][position] = cell
        rows[name] = row
    return rows


def digit_table(seeds, ranks, tok, position="after_think"):
    ids = [tok.encode(d, add_special_tokens=False)[0] for d in DIGITS]
    out = {}
    for name, s in seeds.items():
        answer = s["leak"].submission[s["leak"].blind]
        if not answer.strip().isdigit():
            continue
        row = {"answer": answer}
        for branch in ("leak", "benign_k1", "none"):
            b = s[branch]
            for label in ("actual", "kept"):
                H = b.h(b.blind, "solve", label, position)
                if H is not None and not (branch == "none" and label == "kept"):
                    row[f"{branch} {label}"] = ranks(H, b.layers, ids).min(axis=0).tolist()
        out[name] = row
    return out


def mass_table(seeds):
    cells = defaultdict(lambda: defaultdict(list))
    for s in seeds.values():
        for v, b in s.items():
            for (agent, phase), t in b.turns.items():
                if agent not in b.free:
                    continue
                for position in MASS_POSITIONS:
                    m = {ref: b.mass(agent, phase, position, ref) for ref in REFS}
                    if m["kept"] is not None:
                        cells[v][position].append((m["kept"], m["compacted"], t["reference_kind"]))
    return {v: {p: {"turns": len(x), "kept_median": float(np.median([a for a, _, _ in x])),
                    "compacted_median": float(np.median([c for _, c, _ in x if c is not None])) if any(c is not None for _, c, _ in x) else None,
                    "masked_turns": sum(k == "masked" for _, _, k in x)} for p, x in rows.items()} for v, rows in cells.items()}


def recovery_table(seeds, tok, runs):
    """Blind agent: distance from the none branch's actual state, and whether the history up to the position is the same."""
    from detector.render import find_turn, load_turns, render_turn
    out = defaultdict(lambda: defaultdict(list))
    renders = {}

    def render(seed, branch, agent):
        key = (seed, branch, agent)
        if key not in renders:
            renders[key] = render_turn(tok, find_turn(load_turns(runs/seed/branch), agent, EPISODE, "solve"))
        return renders[key]

    for seed, s in seeds.items():
        none = s["none"]
        blind = s["leak"].blind
        base_r = render(seed, "none", blind)
        for branch, b in s.items():
            if branch == "none":
                continue
            r = render(seed, branch, blind)
            for position in ("prompt_last", "first_generated", "first_thought", "after_think"):
                base, floor_ref = none.h(blind, "solve", "actual", position), none.h(blind, "solve", "kept", position)
                if base is None or b.h(blind, "solve", "actual", position) is None:
                    continue
                upto = r["positions"][position]
                history = [t for t, k in zip(r["ids"][:upto + 1], r["mask"][:upto + 1]) if k]
                floor = relative(floor_ref, base)
                dist = {label: relative(b.h(blind, "solve", label, position), base) for label in ("actual", "kept", "compacted")}
                out[branch][position].append({"seed": seed, "same_history": history == base_r["ids"][:base_r["positions"][position] + 1],
                                              "floor": floor.tolist(), **{k: v.tolist() for k, v in dist.items()},
                                              "layers_within_floor": int((dist["compacted"] <= floor).sum())})
    return out


def verdict_table(seeds, tok):
    mid = BANDS["L26-30"]
    out = {}
    for seed, s in seeds.items():
        out[seed] = {}
        for v, b in s.items():
            rows = []
            for agent in b.free:
                t = b.turn(agent, "verdict")
                if t is None:
                    continue
                top = b.top_ids(agent, "verdict", "actual", "first_generated")
                counts = Counter(int(x) for l in mid for x in top[b.layers.index(l)])
                z_a, z_r = b.z(agent, "verdict", "actual", "first_generated"), b.z(agent, "verdict", "kept", "first_generated")
                pooled = defaultdict(float)
                if z_a is not None:
                    for a, r in zip(z_a, z_r):
                        for c, val in a.items():
                            if val > r.get(c, 0.0):
                                pooled[c] += (val - r.get(c, 0.0))/len(z_a)
                rows.append({"agent": agent, "blind": agent == b.blind, "masked_tokens": t["masked_tokens"],
                             "top10_band": [tok.decode([x]) for x, _ in counts.most_common(10)],
                             "surplus_top": [(tok.decode([c]), round(val, 5)) for c, val in sorted(pooled.items(), key=lambda x: -x[1])[:10]],
                             "mass": b.mass(agent, "verdict", "first_generated")})
            out[seed][v] = rows
    return out


def mention_table(seeds, ranks, tok):
    """Blind agent at mention_pre and first_generated, per layer, every seed."""
    ids = {d: tok.encode(d, add_special_tokens=False)[0] for d in DIGITS}
    out = {}
    for seed, s in seeds.items():
        leak = s["leak"]
        blind = leak.blind
        answer = leak.submission[blind]
        planted = answer_tokens(tok, answer)
        numeric = answer.strip().isdigit()
        others = [ids[d] for d in DIGITS if ids[d] not in planted] if numeric else []
        row = {"answer": answer, "planted": [tok.decode([t]) for t in planted[:2]], "positions": {}}
        for position in ("mention_pre", "first_generated"):
            none_a, none_r = s["none"].h(blind, "solve", "actual", position), s["none"].h(blind, "solve", "kept", position)
            floor = relative(none_a, none_r) if none_a is not None else None
            cell = {"floor": None if floor is None else floor.tolist(), "branches": {}}
            for branch in ("leak", "benign_k1", "benign_k3", "proposal_k1", "none"):
                b = s.get(branch)
                if b is None or b.h(blind, "solve", "actual", position) is None:
                    continue
                t = b.turn(blind)
                a = b.h(blind, "solve", "actual", position)
                entry = {"token": t["position_tokens"][position],
                         "next_token": t["position_tokens"].get("mention_0") if position == "mention_pre" else None,
                         "same_text_as_leak": None}
                for ref in REFS:
                    r = b.h(blind, "solve", ref, position)
                    delta = relative(a, r)
                    entry[f"delta_{ref}"] = delta.tolist()
                    if floor is not None:
                        entry[f"ratio_{ref}"] = (delta/floor).tolist()
                tokens = planted[:2] + others
                ra = ranks(a, b.layers, tokens)
                rr = ranks(b.h(blind, "solve", "kept", position), b.layers, tokens)
                entry["rank_planted"] = [[ra[:, k].tolist(), rr[:, k].tolist()] for k in range(len(planted[:2]))]
                if others:
                    entry["best_other_digit"] = [ra[:, len(planted[:2]):].min(1).tolist(), rr[:, len(planted[:2]):].min(1).tolist()]
                z_a = b.z(blind, "solve", "actual", position)
                if z_a is not None:
                    for ref in REFS:
                        z_r = b.z(blind, "solve", ref, position)
                        entry[f"share_first_{ref}"] = [[x.get(planted[0], 0.0), y.get(planted[0], 0.0)] for x, y in zip(z_a, z_r)]
                        entry[f"mass_{ref}"] = mass(z_a, z_r)
                    entry["jspace_layers"] = [l for l, x in zip(b.layers, z_a) if any(p in x for p in planted)]
                    entry["surplus_layers"] = [l for l, x, y in zip(b.layers, z_a, b.z(blind, "solve", "kept", position)) if any(x.get(p, 0) > y.get(p, 0) for p in planted)]
                cell["branches"][branch] = entry
            row["positions"][position] = cell
        out[seed] = row
    return out


# ── W7 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

def readers(b):
    """Free agents whose solve turn had a peer note masked."""
    return [a for a in b.free if b.exposed(a)]


def signature(b, position, exclude, layers, ref="kept"):
    rows = [unit(b.d(a, "solve", position, ref)[layers]) for a in readers(b) if a != exclude]
    return np.mean(rows, axis=0) if rows else None


def direction_test(seeds, ref="kept"):
    notes = [n for n in NOTES if all(n in s for s in seeds.values())]
    pairs = [(a, b) for i, a in enumerate(notes) for b in notes[i + 1:]]
    result = {"notes": notes, "ref": ref, "blind_pairs": {}, "agreement": {}, "source": {}, "uptake": {}}
    for p in DIRECTION_POSITIONS:
        result["blind_pairs"][p] = {f"{a} vs {b}": np.mean([cos(s[a].d(s[a].blind, "solve", p, ref), s[b].d(s[b].blind, "solve", p, ref))
                                                             for s in seeds.values()], axis=0).tolist() for a, b in pairs}
        result["agreement"][p], result["source"][p] = {}, {}
        for band_name, layers in BANDS.items():
            L = [LENS_LAYERS.index(l) for l in layers]
            within, across, hits = [], [], []
            for s in seeds.values():
                for n in notes:
                    rs = readers(s[n])
                    within += [float(np.mean(cos(s[n].d(i, "solve", p, ref)[L], s[n].d(j, "solve", p, ref)[L]))) for i in rs for j in rs if i < j]
                for a, b in pairs:
                    both = set(readers(s[a])) & set(readers(s[b]))
                    across += [float(np.mean(cos(s[a].d(i, "solve", p, ref)[L], s[b].d(i, "solve", p, ref)[L]))) for i in sorted(both)]
                for n in notes:
                    for i in readers(s[n]):
                        sims = {m: float(np.mean(cos(s[n].d(i, "solve", p, ref)[L], signature(s[m], p, i, L, ref)))) for m in notes}
                        hits.append(max(sims, key=sims.get) == n)
            result["agreement"][p][band_name] = {"within_note_across_readers": float(np.mean(within)), "within_n": len(within),
                                                 "same_reader_across_notes": float(np.mean(across)), "across_n": len(across)}
            result["source"][p][band_name] = {"accuracy": float(np.mean(hits)), "n": len(hits), "chance": 1/len(notes)}
    every = list(range(len(LENS_LAYERS)))
    for branch in ("leak", "benign_k1"):
        rows = defaultdict(list)
        for s in seeds.values():
            b = s[branch]
            v = b.d(b.blind, "solve", "mention_pre", ref)
            if v is None:
                continue
            for n in notes:
                rows[f"vs {n} signature"].append(cos(v, signature(s[n], "first_generated", b.blind, every, ref)))
        result["uptake"][branch] = {k: {"mean": np.mean(x, axis=0).tolist(), "n": len(x)} for k, x in rows.items()}
    return result


def auroc(pos, neg):
    pos, neg = [x for x in pos if x is not None], [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5*(p == n) for p in pos for n in neg)/(len(pos)*len(neg))


def detection(seeds, ranks, tok):
    notes = [n for n in NOTES if all(n in s for s in seeds.values())]
    rows = []
    rank_cache = {}

    def naming(seed, b, agent, position, planted):
        key = (seed, id(b), agent, position)
        if key not in rank_cache:
            a, r = b.h(agent, "solve", "actual", position), b.h(agent, "solve", "kept", position)
            if a is None:
                rank_cache[key] = None
            else:
                late = [b.layers.index(l) for l in LATE]
                ra, rr = ranks(a[late], LATE, [planted])[:, 0], ranks(r[late], LATE, [planted])[:, 0]
                rank_cache[key] = (-float(ra.min()), float(np.max(np.log1p(rr) - np.log1p(ra))))
        return rank_cache[key]

    for seed, s in seeds.items():
        leak = s["leak"]
        blind = leak.blind
        planted = answer_tokens(tok, leak.submission[blind])[0]
        for variant, b in s.items():
            for agent in b.free:
                if b.turn(agent) is None:
                    continue
                row = {"seed": seed, "variant": variant, "agent": agent, "blind": agent == b.blind, "exposed": b.exposed(agent),
                       "leak_used": bool(b.agents[agent].get("leak_used"))}
                for position in DIRECTION_POSITIONS + ("mention_pre",):
                    for ref in REFS:
                        m, floor = b.mass(agent, "solve", position, ref), s["none"].mass(agent, "solve", position, "kept")
                        row[f"mass/floor {ref} | {position} | all"] = m/floor if m is not None and floor else None
                    got = naming(seed, b, agent, position, planted)
                    if got is not None:
                        row[f"naming rank | {position} | late"], row[f"naming lift | {position} | late"] = got
                    for band_name, layers in BANDS.items():
                        L = [LENS_LAYERS.index(l) for l in layers]
                        d = b.d(agent, "solve", position, "kept")
                        if d is None or position == "mention_pre":
                            continue
                        d = d[L]
                        benign = signature(s["benign_k1"], position, agent, L)
                        row[f"content | {position} | {band_name}"] = float(1 - np.mean(cos(d, benign))) if benign is not None else None
                        sig = {n: signature(s[n], position, agent, L) for n in notes}
                        if all(x is not None for x in sig.values()):
                            sim = {n: float(np.mean(cos(d, x))) for n, x in sig.items()}
                            row[f"source | {position} | {band_name}"] = sim["leak"] - max(sim[n] for n in notes if n != "leak")
                            row[f"nearest | {position} | {band_name}"] = max(sim, key=sim.get)
                if agent == b.blind:
                    for ref in REFS:
                        a, r = b.h(agent, "solve", "actual", "mention_pre"), b.h(agent, "solve", ref, "mention_pre")
                        na, nr = s["none"].h(agent, "solve", "actual", "mention_pre"), s["none"].h(agent, "solve", "kept", "mention_pre")
                        if a is None or na is None:
                            continue
                        ratio = relative(a, r)/relative(na, nr)
                        for band_name, layers in BANDS.items():
                            row[f"jump {ref} | mention_pre | {band_name}"] = float(np.mean(ratio[[LENS_LAYERS.index(l) for l in layers]]))
                rows.append(row)
    keys = sorted({k for r in rows for k in r if " | " in k and not k.startswith("nearest")})

    def pick(seed, variant, agent):
        return next((r for r in rows if (r["seed"], r["variant"], r["agent"]) == (seed, variant, agent)), None)

    claims = {"1 blind, leak vs none": [], "2 blind, leak vs benign_k1": [], "2b blind, leak vs benign_k3": [],
              "3 exposed non-users, leak vs benign_k1": []}
    for seed, s in seeds.items():
        blind = s["leak"].blind
        claims["1 blind, leak vs none"].append((pick(seed, "leak", blind), pick(seed, "none", blind)))
        claims["2 blind, leak vs benign_k1"].append((pick(seed, "leak", blind), pick(seed, "benign_k1", blind)))
        if "benign_k3" in s:
            claims["2b blind, leak vs benign_k3"].append((pick(seed, "leak", blind), pick(seed, "benign_k3", blind)))
        for agent in readers(s["leak"]):
            if agent != blind and agent in readers(s["benign_k1"]):
                claims["3 exposed non-users, leak vs benign_k1"].append((pick(seed, "leak", agent), pick(seed, "benign_k1", agent)))
    out = {}
    for name, pairs in claims.items():
        pairs = [(p, n) for p, n in pairs if p and n]
        cell = {"pairs": len(pairs), "seeds": len({p["seed"] for p, _ in pairs})}
        for key in keys:
            pos, neg = [p.get(key) for p, _ in pairs], [n.get(key) for _, n in pairs]
            if all(x is None for x in pos):
                continue
            paired = [(p, n) for p, n in zip(pos, neg) if p is not None and n is not None]
            cell[key] = {"auroc": auroc(pos, neg), "paired_wins": sum(p > n for p, n in paired), "paired_n": len(paired),
                         "positive_median": float(np.median([x for x in pos if x is not None])),
                         "negative_median": float(np.median([x for x in neg if x is not None])) if any(x is not None for x in neg) else None}
        out[name] = cell
    nearest = {}
    for position in DIRECTION_POSITIONS:
        for band_name in BANDS:
            k = f"nearest | {position} | {band_name}"
            hits = [r[k] == r["variant"] for r in rows if r.get(k) and r["variant"] in notes and r["exposed"]]
            nearest[f"{position} | {band_name}"] = {"accuracy": float(np.mean(hits)) if hits else None, "n": len(hits)}
    return {"notes": notes, "claims": out, "nearest_accuracy": nearest, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", type=Path, default=ROOT/"data/replay_14b/sweep14b")
    parser.add_argument("--runs", type=Path, default=ROOT/"runs/sweep14b")
    parser.add_argument("--out", type=Path, default=ROOT/"results/lane_c/readout.json")
    parser.add_argument("--parts", nargs="*", default=["floor", "leak", "digits", "mass", "recovery", "verdict", "mention", "direction", "detection"])
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    seeds = load(args.replay, args.runs)
    ranks = Ranks()
    result = json.loads(args.out.read_text()) if args.out.exists() else {}
    result.update({"model": MODEL, "seeds": sorted(seeds), "variants": {k: sorted(v) for k, v in seeds.items()},
                   "layers": list(LENS_LAYERS), "bands": BANDS, "late": LATE, "show": SHOW})
    parts = {"floor": lambda: floor_table(seeds), "leak": lambda: leak_table(seeds, ranks, tok),
             "digits": lambda: digit_table(seeds, ranks, tok), "mass": lambda: mass_table(seeds),
             "recovery": lambda: recovery_table(seeds, tok, args.runs), "verdict": lambda: verdict_table(seeds, tok),
             "mention": lambda: mention_table(seeds, ranks, tok),
             "direction": lambda: {ref: direction_test(seeds, ref) for ref in REFS},
             "detection": lambda: detection(seeds, ranks, tok)}
    for part in args.parts:
        result[part] = parts[part]()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1, default=float, ensure_ascii=False) + "\n")
        print("done", part, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
