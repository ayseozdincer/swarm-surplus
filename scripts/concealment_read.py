#!/usr/bin/env python3
"""W7 concealment read, three cases, descriptive. Writes results/concealment_read.json.

Carol's solve turn in the sweep1 leak branch of seeds 2000, 2001 and 2002, episode 2. In 2000 her submission note names the
source (Frank, via the team notes). In 2001 and 2002 it does not ("input specifics known to the reviewer", and a claim of her
own processing). Two concealers and one discloser test nothing. Sweep2's disclosed_source and concealed_source labels are
where the test runs.

The surplus direction is d = h_actual - h_reference with the kept reference (peer spans masked, positions kept), as in the
replay. Per layer band, at answer_pre, at every token of the submission note, and at end_of_response:
  cosine of d with the ontology concealment and misrepresentation directions (the J-space repo's CONCEPTS word groups as
  Qwen3 single tokens, each against the 32 ontology control words), mean over words, then over the band's layers; the
  same for the 18 ontology target words (conceal, covert, secret, deceive, ...), since concealment has two Qwen tokens
  J-space surplus mass, sum over concepts of [z - z~]+, mean over the band's layers
  cosine between the concealers' and the discloser's surplus directions (the note-span direction is the mean of the unit
  directions over the note tokens)
For scale, the same cosines and mass for carol's solve turn in benign_k1 and the plain-kernel floor from the none branch
(answer_pre and end_of_response only, from the replay).

  collect   (GPU) actual and kept-reference residuals at answer_pre, every note token and end_of_response
  analyze   (GPU for the lens) decompose the collected rows and write the JSON; without collect, replay positions only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]

SEEDS = {2000: "discloser", 2001: "concealer", 2002: "concealer"}
RUNS = ROOT/"runs/sweep1"
REPLAY = ROOT/"data/replay/sweep1"
OUT = ROOT/"data/concealment"
RESULT = ROOT/"results/concealment_read.json"
AGENT, EPISODE, PHASE = "carol", 2, "solve"
BANDS = {"L26-30": range(26, 31), "L50-62": range(50, 63), "all": range(63)}
GROUPS = ("concealment", "misrepresentation", "ontology18")


def unit(x):
    return x/np.linalg.norm(x, axis=-1, keepdims=True)


def collect(args):
    from detector.render import find_turn, load_turns, render_turn, token_span, tool_value_span
    from replay import LENS_LAYERS, Capture, attention_mask_4d, load_model
    model, tok = load_model()
    capture = Capture(model, LENS_LAYERS)
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        r = render_turn(tok, find_turn(load_turns(RUNS/f"seed{seed}/leak"), AGENT, EPISODE, PHASE))
        encoded = tok(r["text"], add_special_tokens=False, return_offsets_mapping=True)
        first, last = token_span([tuple(o) for o in encoded["offset_mapping"]], *tool_value_span(r["text"], r["prompt_chars"], "note"))
        names = ["answer_pre"] + [f"note_{i}" for i in range(last - first)] + ["end_of_response"]
        positions = [r["positions"]["answer_pre"], *range(first, last), r["positions"]["end_of_response"]]
        n = len(r["ids"])
        actual = capture.run(model, r["ids"], positions, attention_mask_4d(n))
        reference = capture.run(model, r["ids"], positions, attention_mask_4d(n, r["mask"]))
        np.savez(OUT/f"seed{seed}.npz", actual=actual.numpy(), reference=reference.numpy(), names=np.array(names),
                 positions=np.array(positions), tokens=np.array([tok.decode([r["ids"][p]]) for p in positions]))
        print(seed, len(names), "positions", flush=True)


def concept_ids(tok):
    """CONCEPTS word groups and the ontology control words as Qwen3 single-token ids."""
    spec = importlib.util.spec_from_file_location("maj_config", ROOT/"third_party/maj-lens/config.py")
    maj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maj)
    out = {}
    for group in GROUPS[:2]:
        ids, used = [], []
        for word in sorted(maj.CONCEPTS[group]):
            for text in (" " + word, word):                                 # one token per word, spaced form first
                enc = tok.encode(text, add_special_tokens=False)
                if len(enc) == 1:
                    if enc[0] not in ids:
                        ids.append(enc[0])
                        used.append(text)
                    break
        out[group] = (ids, used)
    onto = json.loads((ROOT/"third_party/maj-lens/ontology.json").read_text())
    for group, words in (("ontology18", onto["target_ids_by_word"]),):
        ids = [tok.encode(" " + w, add_special_tokens=False) for w in words]
        if any(len(i) != 1 for i in ids):
            raise ValueError("ontology target word is not one token")
        out[group] = ([i[0] for i in ids], [" " + w for w in words])
    controls = onto["control_ids_by_word"]
    control = []
    for word in controls:
        enc = tok.encode(" " + word, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"control word is not one token: {word!r}")
        control.append(enc[0])
    return out, control


def replay_turn(seed, variant):
    """(names, actual, reference, z_actual, z_reference) for carol's solve turn in a replayed sweep1 branch."""
    from surplus_table import shares
    directory = REPLAY/f"seed{seed}/{variant}"
    turn = next(t for t in json.loads((directory/"index.json").read_text())["turns"]
                if t["agent"] == AGENT and t["phase"] == PHASE and t["episode_index"] == EPISODE)
    data = np.load(directory/turn["file"])
    names = list(turn["positions"])
    meta = json.loads((directory/"rows.json").read_text())["rows"]
    js = np.load(directory/"jspace.npz")
    index = {(r["file"], r["pass"], r["position"]): i for i, r in enumerate(meta) if r["file"] == turn["file"]}
    z = {label: [shares(js["support"][index[(turn["file"], label, n)]], js["coefficients"][index[(turn["file"], label, n)]])
                 for n in names] for label in ("actual", "reference")}
    return names, data["actual"], data["reference"], z["actual"], z["reference"], turn["reference_kind"]


def analyze(args):
    import torch
    from transformers import AutoTokenizer
    from detector import jspace
    from replay import LENS_FILE, MODEL_DIR
    from surplus_table import mass, shares
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    device = args.device
    lens = jspace.Lens(MODEL_DIR, LENS_FILE, device=device)
    groups, control = concept_ids(tok)
    control_t = torch.tensor(control, device=device)
    directions = {}
    for group, (ids, _) in groups.items():
        target = torch.tensor(ids, device=device)
        directions[group] = np.stack([
            torch.nn.functional.normalize((lens.head[target] - lens.head[control_t].mean(0)) @ lens.J[l].to(device).float(), dim=1).cpu().numpy()
            for l in lens.layers])                                                 # [63, words, d_model]

    def onto(d):
        """d [63, d_model] -> {group: per-layer mean-over-words cosine}"""
        u = unit(d)
        return {g: np.einsum("ld,lwd->lw", u, directions[g]).mean(axis=1) for g in GROUPS}

    def band_stats(d, z_a, z_r):
        o = onto(d)
        row = {}
        for b, layers in BANDS.items():
            keep = set(layers)
            row[b] = {**{f"cos {g}": float(o[g][list(layers)].mean()) for g in GROUPS}, "mass": mass(z_a, z_r, keep)}
        return row

    result = {"seeds": {str(s): v for s, v in SEEDS.items()}, "agent": AGENT, "episode": EPISODE, "phase": PHASE,
              "concept_tokens": {g: used for g, (_, used) in groups.items()}, "bands": {b: [min(v), max(v)] for b, v in BANDS.items()},
              "cases": {}, "comparison": {}, "cross_case": {}}
    span_dir = {}
    for seed in SEEDS:
        case = {}
        path = OUT/f"seed{seed}.npz"
        if path.exists():
            saved = np.load(path)
            names, actual, reference = list(saved["names"]), saved["actual"], saved["reference"]
            X = np.concatenate([actual, reference])                                # [2P, 63, d_model]
            support = np.full((len(X), len(lens.layers), jspace.K), -1, dtype=np.int32)
            coef = np.zeros((len(X), len(lens.layers), jspace.K), dtype=np.float32)
            for li, layer in enumerate(lens.layers):
                atoms, norms = lens.atoms(layer)
                s, c = jspace.pursuit(torch.from_numpy(X[:, li]).to(device), atoms, norms)
                support[:, li], coef[:, li] = s.cpu().numpy(), c.cpu().numpy()
                del atoms, norms
            z = [shares(support[i], coef[i]) for i in range(len(X))]
            z_a, z_r = z[:len(names)], z[len(names):]
            note = [i for i, n in enumerate(names) if n.startswith("note_")]
            per_token = [band_stats(actual[i] - reference[i], z_a[i], z_r[i]) for i in range(len(names))]
            case["note_text"] = "".join(saved["tokens"][note])
            case["note_tokens"] = len(note)
            case["note_span_mean"] = {b: {k: float(np.mean([per_token[i][b][k] for i in note])) for k in per_token[note[0]][b]} for b in BANDS}
            case["note_span_max_mass"] = {b: float(np.max([per_token[i][b]["mass"] for i in note])) for b in BANDS}
            case["answer_pre"] = per_token[names.index("answer_pre")]
            case["end_of_response"] = per_token[names.index("end_of_response")]
            span_dir[seed] = {"note": unit(unit(actual[note] - reference[note]).mean(axis=0)),
                              "answer_pre": unit(actual[0] - reference[0]), "end_of_response": unit(actual[-1] - reference[-1])}
            r_names, r_actual, _, _, _, _ = replay_turn(seed, "leak")
            case["answer_pre_matches_replay_max_abs"] = float(np.abs(actual[0] - r_actual[r_names.index("answer_pre")]).max())
            case["source"] = "note-span pass"
        else:
            names, actual, reference, z_a, z_r, _ = replay_turn(seed, "leak")
            for p in ("answer_pre", "end_of_response"):
                i = names.index(p)
                case[p] = band_stats(actual[i] - reference[i], z_a[i], z_r[i])
            span_dir[seed] = {p: unit(actual[names.index(p)] - reference[names.index(p)]) for p in ("answer_pre", "end_of_response")}
            case["source"] = "replay positions only (note tokens not captured)"
        result["cases"][str(seed)] = case
        comparison = {}
        for variant in ("benign_k1", "none"):
            names, actual, reference, z_a, z_r, kind = replay_turn(seed, variant)
            comparison[variant] = {"reference_kind": kind}
            for p in ("answer_pre", "end_of_response"):
                if p in names:
                    i = names.index(p)
                    comparison[variant][p] = band_stats(actual[i] - reference[i], z_a[i], z_r[i])
            if variant == "benign_k1":
                span_dir.setdefault(f"benign{seed}", {})
                for p in ("answer_pre", "end_of_response"):
                    if p in names:
                        span_dir[f"benign{seed}"][p] = unit(actual[names.index(p)] - reference[names.index(p)])
        result["comparison"][str(seed)] = comparison
    discloser = next(s for s, v in SEEDS.items() if v == "discloser")
    concealers = [s for s, v in SEEDS.items() if v == "concealer"]
    pairs = [(c, discloser) for c in concealers] + [tuple(concealers)]
    for p in ("note", "answer_pre", "end_of_response"):
        if not all(p in span_dir[s] for s in SEEDS):
            continue
        entry = {}
        for a, b in pairs:
            cos = np.sum(span_dir[a][p]*span_dir[b][p], axis=-1)
            entry[f"{a} vs {b}"] = {band: float(cos[list(layers)].mean()) for band, layers in BANDS.items()}
        if p != "note":
            for a, b in pairs:
                if p in span_dir[f"benign{a}"] and p in span_dir[f"benign{b}"]:
                    cos = np.sum(span_dir[f"benign{a}"][p]*span_dir[f"benign{b}"][p], axis=-1)
                    entry[f"benign_k1 {a} vs {b}"] = {band: float(cos[list(layers)].mean()) for band, layers in BANDS.items()}
        result["cross_case"][p] = entry
    RESULT.write_text(json.dumps(result, indent=1) + "\n")
    for seed, case in result["cases"].items():
        print(seed, SEEDS[int(seed)], case["source"])
        for p in ("note_span_mean", "answer_pre", "end_of_response"):
            if p in case:
                print(" ", p, {b: {k: round(v, 4) for k, v in x.items()} for b, x in case[p].items()})
    print(json.dumps(result["cross_case"], indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("collect", "analyze"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    return collect(args) if args.action == "collect" else analyze(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
