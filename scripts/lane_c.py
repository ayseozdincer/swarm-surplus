#!/usr/bin/env python3
"""Lane C. The 32B W3 and W7 readouts over every seed of a planted-channel sweep, on the model SWARM_MODEL names
(here Qwen3-14B with its converged lens). Writes one JSON, results/lane_c/readout.json by default.

  SWARM_MODEL=qwen3_14b python scripts/lane_c.py --roots data/replay_14b/blind/sweep14b data/replay_14b/verdict/sweep14b

Episode 2 of every seed. The blind agent is the free agent run.json marks blind; in the leak branch it used the note. A
branch's turns can come from several replay roots: stage 1 (scripts/lane_c_blind.py) holds the blind agent's solve turn with
its placebo pass and per-token trace, stage 2 (replay.py) the free agents' verdict turns. `kept` is the reference with
positions kept, `placebo` the reference that masks a random designer span of the note's length instead of the note.
Bands are the 32B ones at the same relative depth (replay.band), named by their 32B layers.

  floor        kernel noise at the read positions (none turns, whose second pass is the plain kernel) against the mask
               effect on the blind agent's leak turn, per layer, and the same over every reply token from the traces
  leak         W3 table 1. The planted answer's tokens at prompt_last, first_generated, after_think, answer_pre (positive
               control) and mention_pre: top-10 layers, J-space layers, surplus layers, best rank over layers, per seed
  digits       W3 table 2, numeric answers only. Best rank of every digit at after_think
  mass         W3 surplus mass by branch and position, median over turns, kept and placebo
  recovery     distance of the blind agent's actual and kept states from the none branch's, per layer
  placebo      the placebo reference against the kept one at the read positions: its distance from the kept reference,
               the cosine of the two surplus directions, and both masses
  verdict      W3's verdict-turn table. Top-10 over the mid band and the top surplus concepts at first_generated
  mention      W3 follow-up 1. The blind agent at mention_pre and first_generated, per layer
  trace        the per-token trace of the blind agent's reply up to and past the first mention
  direction    W7 part A on the free readers' verdict turns (the blind agent's pairs on its solve turn)
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
MASS_POSITIONS = ("prompt_last", "first_generated", "first_thought", "after_think", "answer_pre", "verdict_pre", "end_of_response", "mention_pre")
LEAK_POSITIONS = ("first_generated", "after_think", "prompt_last", "answer_pre", "mention_pre")
DIRECTION_POSITIONS = ("prompt_last", "first_generated", "first_thought")
NOTES = ("leak", "benign_k1", "proposal_k1")
VARIANTS = ("leak", "none", "benign_k1", "benign_k3", "proposal_k1", "proposal_k3")
DIGITS = "0123456789"
REFS = ("kept", "placebo")
READER_PHASE = "verdict"


def unit(x):
    return x/np.linalg.norm(x, axis=-1, keepdims=True)


def cos(a, b):
    return np.sum(unit(a)*unit(b), axis=-1)


def relative(a, b):
    return np.linalg.norm(a - b, axis=-1)/np.linalg.norm(b, axis=-1)


class Source:
    """One replay directory: its index, J-space shares and lazily loaded arrays."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.index = json.loads((directory/"index.json").read_text())
        self.z_index = None
        if (directory/"jspace.npz").exists():
            meta = json.loads((directory/"rows.json").read_text())
            data = np.load(directory/"jspace.npz")
            self.support, self.coefficients = data["support"], data["coefficients"]
            self.z_index = {(r["file"], r["pass"], r["position"]): i for i, r in enumerate(meta["rows"])}
        self.loaded = {}

    def array(self, file, name):
        if file not in self.loaded:
            self.loaded[file] = [np.load(self.directory/file), {}]
        source, cache = self.loaded[file]
        if name not in cache:
            cache[name] = source[name] if name in source.files else None
        return cache[name]


class Branch:
    """One variant run: its turns from every replay root, residuals by pass and position, J-space shares, who is free and
    who is blind."""

    def __init__(self, directories, run_dir: Path):
        self.sources = [Source(d) for d in directories]
        self.layers = self.sources[0].index["layers"]
        agents = json.loads((run_dir/"run.json").read_text())["results"][EPISODE]["agents"]
        self.agents = agents
        self.free = [a for a, r in agents.items() if not r.get("instructed")]
        self.blind = next((a for a in self.free if agents[a].get("blind")), None)
        self.submission = {a: (r.get("submission") or {}).get("answer") for a, r in agents.items()}
        self.turns = {}
        for s in self.sources:
            for t in s.index["turns"]:
                self.turns.setdefault((t["agent"], t["phase"]), (s, t))

    def turn(self, agent, phase="solve"):
        hit = self.turns.get((agent, phase))
        return hit[1] if hit else None

    def exposed(self, agent, phase="solve"):
        t = self.turn(agent, phase)
        return bool(t and t["masked_tokens"] > 0)

    def h(self, agent, phase, label, position):
        """[layers, d_model] residual of a pass at a named position; `kept` is the stored reference."""
        hit = self.turns.get((agent, phase))
        if hit is None or position not in hit[1]["positions"]:
            return None
        s, t = hit
        a = s.array(t["file"], {"kept": "reference"}.get(label, label))
        return None if a is None else a[list(t["positions"]).index(position)]

    def top_ids(self, agent, phase, label, position):
        hit = self.turns.get((agent, phase))
        if hit is None or position not in hit[1]["positions"]:
            return None
        s, t = hit
        a = s.array(t["file"], {"kept": "reference"}.get(label, label) + "_top_ids")
        return None if a is None else a[list(t["positions"]).index(position)]

    def z(self, agent, phase, label, position):
        """J-space shares per layer, or None when that row was not decomposed."""
        hit = self.turns.get((agent, phase))
        if hit is None or hit[0].z_index is None:
            return None
        s, t = hit
        i = s.z_index.get((t["file"], {"kept": "reference"}.get(label, label), position))
        return None if i is None else shares(s.support[i], s.coefficients[i])

    def mass(self, agent, phase, position, ref="kept"):
        z_a, z_r = self.z(agent, phase, "actual", position), self.z(agent, phase, ref, position)
        return None if z_a is None or z_r is None else mass(z_a, z_r)

    def d(self, agent, phase, position, ref="kept"):
        a, r = self.h(agent, phase, "actual", position), self.h(agent, phase, ref, position)
        return None if a is None or r is None else a - r

    def trace(self, agent):
        hit = self.turns.get((agent, "solve"))
        if hit is None or "trace_file" not in hit[1]:
            return None
        s, t = hit
        if t["trace_file"] not in s.loaded:
            data = np.load(s.directory/t["trace_file"])
            s.loaded[t["trace_file"]] = [data, {k: data[k] for k in data.files}]
        return s.loaded[t["trace_file"]][1]


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


def planted(tok, leak):
    """The planted answer's tokens, first the one the blind agent's reasoning writes at its first mention (a string answer
    can take a leading space there, ` Ice` rather than `Ice`), then the answer's own tokens and the spaced form's first."""
    answer = leak.submission[leak.blind]
    t = leak.turn(leak.blind)
    ids = answer_tokens(tok, answer)
    first = t["position_token_ids"]["mention_0"] if t and "mention_0" in t["positions"] else ids[0]
    spaced = [t for t in answer_tokens(tok, " " + answer)[:1] if tok.decode([t]).strip()]
    return list(dict.fromkeys([first] + ids + spaced)), first


def load(roots, runs: Path):
    seeds = {}
    names = sorted({p.name for root in roots if root.exists() for p in root.iterdir() if p.name.startswith("seed")})
    for name in names:
        branches = {}
        for v in VARIANTS:
            dirs = [root/name/v for root in roots if (root/name/v/"index.json").exists()]
            if dirs and (runs/name/v/"run.json").exists():
                branches[v] = Branch(dirs, runs/name/v)
        if all(v in branches for v in ("leak", "none", "benign_k1")) and branches["leak"].turn(branches["leak"].blind):
            seeds[name] = branches
    return seeds


# ── W3 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

def token_floor(seeds):
    """Kernel noise per layer over every reply token of the none branch's blind turns, from the traces."""
    rows = [s["none"].trace(s["none"].blind)["delta_reference"] for s in seeds.values() if s["none"].trace(s["none"].blind) is not None]
    return np.median(np.concatenate(rows), axis=0) if rows else None


def floor_table(seeds):
    out = {}
    for position in ("prompt_last", "first_generated", "first_thought", "after_think", "mention_pre"):
        noise = [relative(b.h(a, ph, "actual", position), b.h(a, ph, "kept", position))
                 for s in seeds.values() for b in [s["none"]] for (a, ph) in b.turns if b.h(a, ph, "actual", position) is not None]
        effect = {ref: [relative(s["leak"].h(s["leak"].blind, "solve", "actual", position), s["leak"].h(s["leak"].blind, "solve", ref, position))
                        for s in seeds.values() if s["leak"].h(s["leak"].blind, "solve", ref, position) is not None] for ref in REFS}
        out[position] = {"noise_turns": len(noise), "effect_turns": len(effect["kept"]),
                         "noise_median": np.median(noise, axis=0).tolist() if noise else None,
                         "effect_median": np.median(effect["kept"], axis=0).tolist() if effect["kept"] else None,
                         "placebo_median": np.median(effect["placebo"], axis=0).tolist() if effect["placebo"] else None}
    f = token_floor(seeds)
    out["every_reply_token"] = None if f is None else {"noise_median": f.tolist(), "turns": sum(1 for s in seeds.values() if s["none"].trace(s["none"].blind) is not None)}
    return out


def leak_table(seeds, ranks, tok):
    rows = {}
    for name, s in seeds.items():
        leak = s["leak"]
        blind = leak.blind
        answer = leak.submission[blind]
        tokens, first = planted(tok, leak)
        row = {"blind": blind, "answer": answer, "planted": [tok.decode([t]) for t in tokens], "numeric": answer.strip().isdigit(),
               "leak_used": bool(leak.agents[blind].get("leak_used")), "positions": {}}
        for position in LEAK_POSITIONS:
            if leak.h(blind, "solve", "actual", position) is None:
                row["positions"][position] = None
                continue
            cell = {"token": leak.turn(blind)["position_tokens"][position]}
            for label in ("actual", "kept", "placebo"):
                top = leak.top_ids(blind, "solve", label, position)
                if top is not None:
                    cell[f"top10_layers_{label}"] = [l for li, l in enumerate(leak.layers) if set(top[li].tolist()) & set(tokens)]
            z_a = leak.z(blind, "solve", "actual", position)
            if z_a is not None:
                cell["jspace_layers"] = [l for li, l in enumerate(leak.layers) if any(t in z_a[li] for t in tokens)]
                for ref in REFS:
                    z_r = leak.z(blind, "solve", ref, position)
                    if z_r is None:
                        continue
                    cell[f"surplus_layers_{ref}"] = [l for li, l in enumerate(leak.layers) if any(z_a[li].get(t, 0) > z_r[li].get(t, 0) for t in tokens)]
                    cell[f"surplus_first_{ref}"] = [max(z_a[li].get(first, 0) - z_r[li].get(first, 0), 0) for li in range(len(z_a))]
            best = {}
            for branch, label in (("leak", "actual"), ("leak", "kept"), ("leak", "placebo"), ("none", "actual"), ("benign_k1", "actual"), ("benign_k1", "kept")):
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
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for s in seeds.values():
        for v, b in s.items():
            for (agent, phase), (_, t) in b.turns.items():
                if agent not in b.free:
                    continue
                group = "blind solve" if phase == "solve" and agent == b.blind else f"free {phase}"
                for position in MASS_POSITIONS:
                    for ref in REFS:
                        m = b.mass(agent, phase, position, ref)
                        if m is not None:
                            cells[group][v][f"{position} {ref}"].append(m)
    return {g: {v: {k: {"turns": len(x), "median": float(np.median(x))} for k, x in rows.items()} for v, rows in by.items()} for g, by in cells.items()}


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
                dist = {label: relative(b.h(blind, "solve", label, position), base) for label in ("actual", "kept")}
                out[branch][position].append({"seed": seed, "same_history": history == base_r["ids"][:base_r["positions"][position] + 1],
                                              "floor": relative(floor_ref, base).tolist(), **{k: v.tolist() for k, v in dist.items()}})
    return out


def placebo_table(seeds):
    """Blind solve turns: the placebo reference against the kept one at the read positions."""
    out = defaultdict(lambda: defaultdict(list))
    for seed, s in seeds.items():
        for v, b in s.items():
            agent = b.blind
            t = b.turn(agent)
            if t is None or "placebo_span" not in t:
                continue
            for position in MASS_POSITIONS:
                a, k, p = (b.h(agent, "solve", x, position) for x in ("actual", "kept", "placebo"))
                if a is None:
                    continue
                out[v][position].append({"seed": seed, "placebo_from_kept": relative(p, k).tolist(), "kept_delta": relative(a, k).tolist(),
                                         "placebo_delta": relative(a, p).tolist(), "cos_d": cos(a - k, a - p).tolist(),
                                         "mass_kept": b.mass(agent, "solve", position, "kept"), "mass_placebo": b.mass(agent, "solve", position, "placebo"),
                                         "placebo_span": t["placebo_span"], "placebo_text": t.get("placebo_text", "")[:80]})
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
        planted_ids, _ = planted(tok, leak)
        numeric = answer.strip().isdigit()
        others = [ids[d] for d in DIGITS if ids[d] not in planted_ids] if numeric else []
        row = {"answer": answer, "planted": [tok.decode([t]) for t in planted_ids[:2]], "positions": {}}
        for position in ("mention_pre", "first_generated"):
            none_a, none_r = s["none"].h(blind, "solve", "actual", position), s["none"].h(blind, "solve", "kept", position)
            floor = relative(none_a, none_r) if none_a is not None else None
            cell = {"floor": None if floor is None else floor.tolist(), "branches": {}}
            for branch in VARIANTS:
                b = s.get(branch)
                if b is None or b.h(blind, "solve", "actual", position) is None:
                    continue
                t = b.turn(blind)
                a = b.h(blind, "solve", "actual", position)
                entry = {"token": t["position_tokens"][position],
                         "next_token": t["position_tokens"].get("mention_0") if position == "mention_pre" else None}
                for ref in REFS:
                    r = b.h(blind, "solve", ref, position)
                    if r is None:
                        continue
                    delta = relative(a, r)
                    entry[f"delta_{ref}"] = delta.tolist()
                    if floor is not None:
                        entry[f"ratio_{ref}"] = (delta/floor).tolist()
                tokens = planted_ids[:2] + others
                ra = ranks(a, b.layers, tokens)
                rr = ranks(b.h(blind, "solve", "kept", position), b.layers, tokens)
                entry["rank_planted"] = [[ra[:, k].tolist(), rr[:, k].tolist()] for k in range(len(planted_ids[:2]))]
                if others:
                    entry["best_other_digit"] = [ra[:, len(planted_ids[:2]):].min(1).tolist(), rr[:, len(planted_ids[:2]):].min(1).tolist()]
                z_a = b.z(blind, "solve", "actual", position)
                if z_a is not None:
                    for ref in REFS:
                        z_r = b.z(blind, "solve", ref, position)
                        if z_r is None:
                            continue
                        entry[f"share_first_{ref}"] = [[x.get(planted_ids[0], 0.0), y.get(planted_ids[0], 0.0)] for x, y in zip(z_a, z_r)]
                        entry[f"mass_{ref}"] = mass(z_a, z_r)
                    z_k = b.z(blind, "solve", "kept", position)
                    entry["jspace_layers"] = [l for l, x in zip(b.layers, z_a) if any(p in x for p in planted_ids)]
                    entry["surplus_layers"] = [l for l, x, y in zip(b.layers, z_a, z_k) if any(x.get(p, 0) > y.get(p, 0) for p in planted_ids)]
                cell["branches"][branch] = entry
            row["positions"][position] = cell
        out[seed] = row
    return out


def trace_table(seeds):
    """Per seed and branch, the blind agent's reply token by token. Delta over the every-token kernel floor, mean over a band;
    the first token where the lens ranks the planted token in the top 10 at a late layer; the jump at the first mention."""
    floor = token_floor(seeds)
    late_idx = [LENS_LAYERS.index(l) for l in BANDS["L50-62"]]
    mid_idx = [LENS_LAYERS.index(l) for l in BANDS["L26-30"]]
    out = {}
    for seed, s in seeds.items():
        out[seed] = {}
        sig = {n: signature(s[n], "first_generated", s["leak"].blind, BANDS["L50-62"]) for n in NOTES if n in s}
        for v, b in s.items():
            tr = b.trace(b.blind)
            if tr is None:
                continue
            t = b.turn(b.blind)
            pos = tr["positions"].tolist()
            start = pos.index(t["positions"]["first_generated"])
            mention = pos.index(t["positions"]["mention_pre"]) if "mention_pre" in t["positions"] else None
            row = {"reply_tokens": len(pos), "first_generated": start, "mention_offset": None if mention is None else mention - start,
                   "mention_token": t["position_tokens"].get("mention_0"), "target": t.get("trace_target_text")}
            for ref in ("reference", "placebo"):
                ratio = tr[f"delta_{ref}"]/floor[None, :]
                for name, idx in (("late", late_idx), ("mid", mid_idx)):
                    series = ratio[:, idx].mean(1)
                    row[f"{ref} {name} max"] = float(series[start:].max())
                    row[f"{ref} {name} argmax_offset"] = int(series[start:].argmax())
                    row[f"{ref} {name} median"] = float(np.median(series[start:]))
                    if mention is not None:
                        row[f"{ref} {name} at mention"] = float(series[mention])
                        row[f"{ref} {name} before mention max"] = float(series[start:mention].max()) if mention > start else None
                        row[f"{ref} {name} first over 5x"] = next((i - start for i in range(start, len(series)) if series[i] > 5), None)
            for label in ("actual", "reference"):
                top = (tr[f"rank_{label}"][start:] < 10).any(1)
                row[f"first top10 offset {label}"] = int(np.argmax(top)) if top.any() else None
                row[f"top10 tokens {label}"] = int(top.sum())
            if mention is not None:
                row["rank at mention, actual"] = tr["rank_actual"][mention].tolist()
                row["rank at mention, reference"] = tr["rank_reference"][mention].tolist()
            # direction of each window token against the notes' verdict signatures at first_generated, late band
            window = tr["window"].tolist()
            bl = tr["band_layers"].tolist()
            li = [bl.index(l) for l in BANDS["L50-62"]]
            if window and all(x is not None for x in sig.values()) and sig:
                d = tr["d_reference"][:, li].astype(np.float32)
                cosines = {n: np.mean(np.sum(unit(d)*unit(x)[None], axis=-1), axis=1) for n, x in sig.items()}
                w_mention = window.index(t["positions"]["mention_pre"]) if "mention_pre" in t["positions"] and t["positions"]["mention_pre"] in window else None
                for n, c in cosines.items():
                    row[f"cos {n} signature, window mean"] = float(c.mean())
                    if w_mention is not None:
                        row[f"cos {n} signature, at mention"] = float(c[w_mention])
                        row[f"cos {n} signature, before mention"] = float(c[:w_mention].mean()) if w_mention else None
            out[seed][v] = row
    return out


# ── W7 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

def readers(b, phase=READER_PHASE):
    """Free agents whose turn in `phase` had a peer note masked."""
    return [a for a in b.free if b.exposed(a, phase)]


def signature(b, position, exclude, layers, ref="kept", phase=READER_PHASE):
    L = [LENS_LAYERS.index(l) for l in layers]
    rows = [unit(b.d(a, phase, position, ref)[L]) for a in readers(b, phase) if a != exclude and b.d(a, phase, position, ref) is not None]
    return np.mean(rows, axis=0) if rows else None


def direction_test(seeds, ref="kept", phase=READER_PHASE):
    notes = [n for n in NOTES if all(n in s for s in seeds.values())]
    pairs = [(a, b) for i, a in enumerate(notes) for b in notes[i + 1:]]
    result = {"notes": notes, "ref": ref, "phase": phase, "blind_pairs": {}, "agreement": {}, "source": {}, "uptake": {}}
    every = list(LENS_LAYERS)
    for p in DIRECTION_POSITIONS:
        result["blind_pairs"][p] = {f"{a} vs {b}": np.mean([cos(s[a].d(s[a].blind, "solve", p, ref), s[b].d(s[b].blind, "solve", p, ref))
                                                             for s in seeds.values()], axis=0).tolist() for a, b in pairs}
        result["agreement"][p], result["source"][p] = {}, {}
        for band_name, layers in BANDS.items():
            L = [LENS_LAYERS.index(l) for l in layers]
            within, across, hits = [], [], []
            for s in seeds.values():
                for n in notes:
                    rs = readers(s[n], phase)
                    within += [float(np.mean(cos(s[n].d(i, phase, p, ref)[L], s[n].d(j, phase, p, ref)[L]))) for i in rs for j in rs if i < j]
                for a, b in pairs:
                    both = set(readers(s[a], phase)) & set(readers(s[b], phase))
                    across += [float(np.mean(cos(s[a].d(i, phase, p, ref)[L], s[b].d(i, phase, p, ref)[L]))) for i in sorted(both)]
                for n in notes:
                    for i in readers(s[n], phase):
                        sims = {m: float(np.mean(cos(s[n].d(i, phase, p, ref)[L], signature(s[m], p, i, layers, ref, phase)))) for m in notes}
                        hits.append(max(sims, key=sims.get) == n)
            result["agreement"][p][band_name] = {"within_note_across_readers": float(np.mean(within)), "within_n": len(within),
                                                 "same_reader_across_notes": float(np.mean(across)), "across_n": len(across)}
            result["source"][p][band_name] = {"accuracy": float(np.mean(hits)), "n": len(hits), "chance": 1/len(notes)}
    for branch in ("leak", "benign_k1"):
        rows = defaultdict(list)
        for s in seeds.values():
            b = s[branch]
            v = b.d(b.blind, "solve", "mention_pre", ref)
            if v is None:
                continue
            for n in notes:
                rows[f"vs {n} signature"].append(cos(v, signature(s[n], "first_generated", b.blind, every, ref, phase)))
        result["uptake"][branch] = {k: {"mean": np.mean(x, axis=0).tolist(), "n": len(x)} for k, x in rows.items()}
    return result


def auroc(pos, neg):
    pos, neg = [x for x in pos if x is not None], [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5*(p == n) for p in pos for n in neg)/(len(pos)*len(neg))


def detection(seeds, ranks, tok):
    notes = [n for n in NOTES if all(n in s for s in seeds.values())]
    floor = token_floor(seeds)
    rows = []
    rank_cache = {}

    def naming(b, agent, phase, position, target):
        key = (id(b), agent, phase, position)
        if key not in rank_cache:
            a, r = b.h(agent, phase, "actual", position), b.h(agent, phase, "kept", position)
            if a is None:
                rank_cache[key] = None
            else:
                late = [b.layers.index(l) for l in LATE]
                ra, rr = ranks(a[late], LATE, [target])[:, 0], ranks(r[late], LATE, [target])[:, 0]
                rank_cache[key] = (-float(ra.min()), float(np.max(np.log1p(rr) - np.log1p(ra))))
        return rank_cache[key]

    for seed, s in seeds.items():
        leak = s["leak"]
        _, first = planted(tok, leak)
        for variant, b in s.items():
            for (agent, phase) in b.turns:
                if agent not in b.free:
                    continue
                row = {"seed": seed, "variant": variant, "agent": agent, "phase": phase, "blind": agent == b.blind,
                       "exposed": b.exposed(agent, phase), "leak_used": bool(b.agents[agent].get("leak_used"))}
                for position in DIRECTION_POSITIONS + ("mention_pre",):
                    floor_mass = s["none"].mass(agent, phase, position, "kept")
                    for ref in REFS:
                        m = b.mass(agent, phase, position, ref)
                        row[f"mass/floor {ref} | {position} | all"] = m/floor_mass if m is not None and floor_mass else None
                    got = naming(b, agent, phase, position, first)
                    if got is not None:
                        row[f"naming rank | {position} | late"], row[f"naming lift | {position} | late"] = got
                    if position == "mention_pre":
                        continue
                    d_full = b.d(agent, phase, position, "kept")
                    if d_full is None:
                        continue
                    for band_name, layers in BANDS.items():
                        d = d_full[[LENS_LAYERS.index(l) for l in layers]]
                        benign = signature(s["benign_k1"], position, agent, layers)
                        row[f"content | {position} | {band_name}"] = float(1 - np.mean(cos(d, benign))) if benign is not None else None
                        sig = {n: signature(s[n], position, agent, layers) for n in notes}
                        if all(x is not None for x in sig.values()):
                            sim = {n: float(np.mean(cos(d, x))) for n, x in sig.items()}
                            row[f"source | {position} | {band_name}"] = sim["leak"] - max(sim[n] for n in notes if n != "leak")
                            row[f"nearest | {position} | {band_name}"] = max(sim, key=sim.get)
                if agent == b.blind and phase == "solve":
                    a = b.h(agent, phase, "actual", "mention_pre")
                    na, nr = s["none"].h(agent, phase, "actual", "mention_pre"), s["none"].h(agent, phase, "kept", "mention_pre")
                    for ref in REFS:
                        r = b.h(agent, phase, ref, "mention_pre")
                        if a is None or na is None or r is None:
                            continue
                        ratio = relative(a, r)/relative(na, nr)
                        for band_name, layers in BANDS.items():
                            row[f"jump {ref} | mention_pre | {band_name}"] = float(np.mean(ratio[[LENS_LAYERS.index(l) for l in layers]]))
                    tr = b.trace(agent)
                    if tr is not None:
                        start = tr["positions"].tolist().index(b.turn(agent)["positions"]["first_generated"])
                        for ref in ("reference", "placebo"):
                            ratio = tr[f"delta_{ref}"][start:]/floor[None, :]
                            for band_name, layers in BANDS.items():
                                row[f"trace max {ref} | reply | {band_name}"] = float(ratio[:, [LENS_LAYERS.index(l) for l in layers]].mean(1).max())
                        row["trace naming | reply | late"] = -float(tr["rank_actual"][start:].min(1).min())
                rows.append(row)
    keys = sorted({k for r in rows for k in r if " | " in k and not k.startswith("nearest")})

    def pick(seed, variant, agent, phase):
        return next((r for r in rows if (r["seed"], r["variant"], r["agent"], r["phase"]) == (seed, variant, agent, phase)), None)

    claims = {"1 blind solve, leak vs none": [], "2 blind solve, leak vs benign_k1": [], "2b blind solve, leak vs benign_k3": [],
              "1v blind verdict, leak vs none": [], "2v blind verdict, leak vs benign_k1": [],
              "3 exposed non-users verdict, leak vs benign_k1": []}
    for seed, s in seeds.items():
        blind = s["leak"].blind
        claims["1 blind solve, leak vs none"].append((pick(seed, "leak", blind, "solve"), pick(seed, "none", blind, "solve")))
        claims["2 blind solve, leak vs benign_k1"].append((pick(seed, "leak", blind, "solve"), pick(seed, "benign_k1", blind, "solve")))
        if "benign_k3" in s:
            claims["2b blind solve, leak vs benign_k3"].append((pick(seed, "leak", blind, "solve"), pick(seed, "benign_k3", blind, "solve")))
        claims["1v blind verdict, leak vs none"].append((pick(seed, "leak", blind, "verdict"), pick(seed, "none", blind, "verdict")))
        claims["2v blind verdict, leak vs benign_k1"].append((pick(seed, "leak", blind, "verdict"), pick(seed, "benign_k1", blind, "verdict")))
        for agent in readers(s["leak"]):
            if agent != blind and agent in readers(s["benign_k1"]):
                claims["3 exposed non-users verdict, leak vs benign_k1"].append((pick(seed, "leak", agent, "verdict"), pick(seed, "benign_k1", agent, "verdict")))
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
            hits = [r[k] == r["variant"] for r in rows if r.get(k) and r["variant"] in notes and r["exposed"] and r["phase"] == READER_PHASE]
            nearest[f"{position} | {band_name}"] = {"accuracy": float(np.mean(hits)) if hits else None, "n": len(hits)}
    return {"notes": notes, "claims": out, "nearest_accuracy": nearest, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="+", type=Path, default=[ROOT/"data/replay_14b/blind/sweep14b", ROOT/"data/replay_14b/verdict/sweep14b"])
    parser.add_argument("--runs", type=Path, default=ROOT/"runs/sweep14b")
    parser.add_argument("--out", type=Path, default=ROOT/"results/lane_c/readout.json")
    parser.add_argument("--parts", nargs="*", default=["floor", "leak", "digits", "mass", "recovery", "placebo", "verdict", "mention", "trace",
                                                       "direction", "detection"])
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    seeds = load(args.roots, args.runs)
    ranks = Ranks()
    result = json.loads(args.out.read_text()) if args.out.exists() else {}
    result.update({"model": MODEL, "seeds": sorted(seeds), "variants": {k: sorted(v) for k, v in seeds.items()},
                   "turns": {k: {v: sorted(f"{a} {p}" for a, p in b.turns) for v, b in s.items()} for k, s in seeds.items()},
                   "layers": list(LENS_LAYERS), "bands": BANDS, "late": LATE, "show": SHOW})
    parts = {"floor": lambda: floor_table(seeds), "leak": lambda: leak_table(seeds, ranks, tok),
             "digits": lambda: digit_table(seeds, ranks, tok), "mass": lambda: mass_table(seeds),
             "recovery": lambda: recovery_table(seeds, tok, args.runs), "placebo": lambda: placebo_table(seeds),
             "verdict": lambda: verdict_table(seeds, tok), "mention": lambda: mention_table(seeds, ranks, tok),
             "trace": lambda: trace_table(seeds),
             "direction": lambda: {"kept": direction_test(seeds, "kept")},
             "detection": lambda: detection(seeds, ranks, tok)}
    for part in args.parts:
        try:
            result[part] = parts[part]()
        except Exception as error:  # a part that cannot run on what has been replayed so far is reported, not fatal
            result[part] = {"error": repr(error)}
            print("failed", part, repr(error), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1, default=float, ensure_ascii=False) + "\n")
        print("done", part, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
