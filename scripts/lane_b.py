#!/usr/bin/env python3
"""Lane B. Phase 1b (NARCBench Transfer) and phase 1a (NARCBench Stego) of the surplus detector on Qwen3-32B.
Run from third_party/maj-lens on the pod. Arrays go to data/replay_lane_b, numbers to results/lane_b.

  noise      one honest turn per tier (and a few more), same kernel twice and plain against explicit mask kernel
  transfer   the 288 final-phase public responses, actual against the honest role prompt of the paired control run,
             plus the placebo-style references of W4 follow-up 1
  stego      hands 1 to 7 of every run, each player read at the sender's last public token in its own context, actual
             against the private phase masked, positions kept and compacted
  decompose  J-space shares of every saved row (detector.jspace.pursuit)
  report_transfer, report_stego

Transfer reference. A final-phase prompt is the system prompt plus the public pipeline output. There is no private block in
that phase (no PRIVATE marker in any of the 288 prompts; the private messages' code reappears only as the colluders' own
public output), so nothing is masked and the reference swaps the system prompt for the agent's honest role prompt in the
paired control run, verbatim, by name (names match in all 36 pairs). Honest agents of a collusion run also have a different
system prompt in the control twin, so they carry a small edit. Control agents' reference is their actual input and runs on
the plain kernel (W3, W4), so it measures kernel noise; every other second pass shares the explicit 4D mask kernel.

Placebo (W4 follow-up 1). The donor is the honest role prompt of the same pipeline slot in the control run of the next
scenario of a different family, the donor's agent name replaced by the reader's.
  placebo   colluders: donor prompt in place of the collusion prompt
  padded    honest: own reference prompt + donor prompt (W4 had it; left out here for time)
  swapped   honest: donor prompt in place of the own reference prompt

Stego reference. detector.stego_spans finds the private spans of a colluder's prompt (the opening block with the private
coordination and the per-hand count records, and later PRIVATE: lines). The reference masks only those, so the sender's
public message stays in. `kept` keeps position ids, `compacted` closes them up (W4: compacted is primary for mass, kept for
directions). Honest players and every player of a control run have nothing to mask; their second pass is the plain kernel.
Control runs take the sender and receiver names of their collusion twin. The read position is read.py's message_end, the
last character of the sender's public message (PRIVATE: lines stripped) after the sender's last name tag in the reader's
text. A reader who speaks before the sender in a hand has no message_end in that hand. The sender is read at the end of its
own public message in its own teacher-forced turn.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]
from detector import jspace  # noqa: E402
from detector.render import compacted_positions, token_at  # noqa: E402
from replay import LENS_FILE, LENS_LAYERS, MODEL_DIR, Capture, attention_mask_4d, load_model  # noqa: E402

DATASET = Path("artifacts/narcbench")
TRANSFER = DATASET/"scenarios/qwen3_32b/transfer"
STEGO = DATASET/"scenarios/qwen3_32b/stego"
OUT = ROOT/"data/replay_lane_b"
RESULTS = ROOT/"results/lane_b"
THINK_END = "</think>\n\n"
T_POSITIONS = ("first_response", "end_of_response")
KINDS = ("placebo", "swapped")      # padded (own prompt + donor) is left out for time, see the report
HANDS = range(1, 8)                         # read.py collect_stego reads hands 1 to 7; hand 8 is the final phase
PRIVATE_LINE = re.compile(r"^[ \t]*PRIVATE\s*:(?!\s*\[what).*\n?", flags=re.M)   # read.py PRIVATE
ALL = tuple(LENS_LAYERS)
BAND = tuple(range(28, 57))                 # read.py's L19-38 of DeepSeek's 43 blocks, scaled to Qwen3-32B's 64
NARROW = (26, 27, 28, 29, 30)               # the band W4 used


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def family(run: str) -> str:
    return run.split("__", 1)[0]


# ═══════════════════════════════════════════════════════════════════ Transfer inputs

def transfer_samples():
    from utils.build.replay import _aligned, _upstream
    loader, _, render = _upstream()["transfer"]
    return [s for _, s in _aligned(DATASET, "transfer", loader, "qwen3_32b")], render


def encode(tok, render, sample):
    text = render(tok, sample)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
    if text.count(THINK_END) < 1:
        raise ValueError("no empty think block in the render")
    return ids, token_at(offsets, text.rfind(THINK_END) + len(THINK_END))


def control_systems(scenario: str) -> list[tuple[str, str]]:
    """(agent name, final-phase system prompt) of the scenario's control run, in pipeline order."""
    prompts = json.loads((TRANSFER/f"{scenario}__control"/"agent_prompts.json").read_text())
    return [(p["agent_name"], p["system_prompt"]) for p in prompts if p["phase"] == "final"]


def donors(runs) -> dict[str, list[tuple[str, str]]]:
    """scenario -> control prompts of the next scenario in a different family, sorted order, deterministic."""
    scenarios = sorted({r.rsplit("__", 1)[0] for r in runs})
    out = {}
    for i, scenario in enumerate(scenarios):
        donor = next(scenarios[(i + n) % len(scenarios)] for n in range(1, len(scenarios))
                     if family(scenarios[(i + n) % len(scenarios)]) != family(scenario))
        out[scenario] = control_systems(donor)
    return out


def transfer_inputs(sample, honest, donor) -> dict[str, str]:
    """System prompt of every reference this sample gets."""
    name = sample["agent_name"]
    order = [n for n, _ in honest]
    own = dict(honest)[name]
    donor_name, donor_prompt = donor[order.index(name)]
    swap = donor_prompt.replace(donor_name, name)
    if any(n in donor_prompt for n, _ in donor if n != donor_name):
        raise ValueError("donor prompt names another agent")
    if sample["role"] == "colluder":
        return {"reference": own, "placebo": swap}
    return {"reference": own, "swapped": swap}


def transfer(args, model, tok, capture):
    samples, render = transfer_samples()
    donor = donors({s["run_id"] for s in samples})
    by_run = defaultdict(list)
    for s in samples:
        by_run[s["run_id"]].append(s)
    for run, rows in sorted(by_run.items()):
        directory = OUT/"transfer"/run
        if (directory/"index.json").exists():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        scenario = run.rsplit("__", 1)[0]
        honest = control_systems(scenario)
        turns, started = [], time.time()
        for s in rows:
            if "PRIVATE" in s["user_prompt"]:
                raise ValueError(f"private marker in a final prompt: {run} {s['agent_name']}")
            ids_a, start_a = encode(tok, render, s)
            arrays = {"actual": capture.run(model, ids_a, [start_a, len(ids_a) - 1], attention_mask_4d(len(ids_a)))}
            meta = {}
            for kind, system in transfer_inputs(s, honest, donor[scenario]).items():
                ids_r, start_r = encode(tok, render, {**s, "system_prompt": system})
                if ids_a[start_a:] != ids_r[start_r:]:
                    raise ValueError(f"response tokens differ: {run} {s['agent_name']} {kind}")
                same = ids_a == ids_r
                arrays[kind] = capture.run(model, ids_r, [start_r, len(ids_r) - 1], None if same else attention_mask_4d(len(ids_r)))
                meta[kind] = {"tokens": len(ids_r), "same_input": same, "kernel": "plain" if same else "explicit"}
            file = f"{slug(s['agent_name'])}.npz"
            np.savez(directory/file, **{k: v.numpy() for k, v in arrays.items()})
            turns.append({"file": file, "agent": s["agent_name"], "role": s["role"], "mode": s["mode"], "family": family(run),
                          "slot": [n for n, _ in honest].index(s["agent_name"]), "tokens": len(ids_a),
                          "response_tokens": len(ids_a) - start_a, "references": meta,
                          "positions": dict(zip(T_POSITIONS, (start_a, len(ids_a) - 1)))})
        (directory/"index.json").write_text(json.dumps({"run": run, "layers": capture.layers, "turns": turns}, indent=1))
        print(run, len(turns), f"{time.time() - started:.0f}s", flush=True)


def transfer_padded(args, model, tok, capture):
    """W4's third placebo group, run after the main replay: honest agents with the donor prompt appended to their own
    reference prompt. Arrays go to data/replay_lane_b/transfer_padded/<run>, one `padded` array per honest agent."""
    samples, render = transfer_samples()
    donor = donors({s["run_id"] for s in samples})
    by_run = defaultdict(list)
    for s in samples:
        if s["role"] != "colluder":
            by_run[s["run_id"]].append(s)
    for run, rows in sorted(by_run.items()):
        directory = OUT/"transfer_padded"/run
        if (directory/"index.json").exists():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        scenario = run.rsplit("__", 1)[0]
        honest = control_systems(scenario)
        turns = []
        for s in rows:
            name = s["agent_name"]
            donor_name, donor_prompt = donor[scenario][[n for n, _ in honest].index(name)]
            ids_a, start_a = encode(tok, render, s)
            ids_r, start_r = encode(tok, render, {**s, "system_prompt": dict(honest)[name] + "\n\n" + donor_prompt.replace(donor_name, name)})
            if ids_a[start_a:] != ids_r[start_r:]:
                raise ValueError(f"response tokens differ: {run} {name} padded")
            file = f"{slug(name)}.npz"
            np.savez(directory/file, padded=capture.run(model, ids_r, [start_r, len(ids_r) - 1], attention_mask_4d(len(ids_r))).numpy())
            turns.append({"file": file, "agent": name, "tokens": len(ids_r), "positions": dict(zip(T_POSITIONS, (start_r, len(ids_r) - 1)))})
        (directory/"index.json").write_text(json.dumps({"run": run, "layers": capture.layers, "turns": turns}, indent=1))
        print("padded", run, len(turns), flush=True)


# ═══════════════════════════════════════════════════════════════════ Stego inputs

def stego_episodes():
    from utils import narcbench as NB
    episodes = NB.episodes(STEGO)
    for run, ep in episodes.items():
        twin = episodes[run.rsplit("__", 1)[0] + "__collusion"]
        ep["sender"], ep["receiver"] = twin["counter"], twin["big_player"]
        config = json.loads((STEGO/run/"run_config.json").read_text())
        ep["players"], ep["mode"] = config["task_agents"], config["mode"]
        ep["roles"] = config["roles"]
    return episodes


def prompt_text(tok, turn):
    from utils import narcbench as NB
    return NB.prompt_text(tok, turn)


def stego_turn(tok, ep, hand, reader):
    """ids, mask (None when nothing is private), named positions and notes of one reader's turn in one hand."""
    from utils import narcbench as NB
    from detector.stego_spans import private_spans, template_tails, token_spans
    sender, receiver = ep["sender"], ep["receiver"]
    t, ts = NB.turn(ep, hand, reader), NB.turn(ep, hand, sender)
    message = PRIVATE_LINE.sub("", ts["response"]).strip()
    tail = message[-40:]
    prompt = prompt_text(tok, t)
    text = prompt + t["response"] + "<|im_end|>\n" if reader == sender else prompt
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
    if reader == sender:
        j = text.find(tail, len(prompt))
        positions = {"prompt_last": token_at(offsets, len(prompt) - 1)}
    else:
        i = text.rfind(f"[{sender}]:")
        j = text.find(tail, i) if i >= 0 else -1
        positions = {"prompt_last": len(ids) - 1}
    if j >= 0:
        positions = {"message_end": next(k for k, (_, end) in enumerate(offsets) if end > j + len(tail) - 1), **positions}
    mask, spans = None, []
    if ep["mode"] == "collusion" and reader in (sender, receiver):
        receiver_tail, sender_tail = template_tails()
        partner = sender if reader == receiver else receiver
        base = text.find(t["user"])
        if base < 0 or text.count(t["user"]) != 1:
            raise ValueError(f"user prompt not found once in the render: {ep['run_id']} {hand} {reader}")
        chars = private_spans(t["user"], partner, receiver_tail if reader == receiver else sender_tail)
        spans = token_spans(offsets, base, chars)
        mask = [1]*len(ids)
        for a, b in spans:
            mask[a:b] = [0]*(b - a)
        if any(not mask[p] for p in positions.values()):
            raise ValueError("a read position is masked")
    return {"ids": ids, "mask": mask, "spans": spans, "positions": positions, "tail": tail}


def stego(args, model, tok, capture):
    from utils import narcbench as NB
    for run, ep in sorted(stego_episodes().items()):
        directory = OUT/"stego"/run
        if (directory/"index.json").exists():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        turns, started = [], time.time()
        for hand in HANDS:
            for reader in ep["players"]:
                r = stego_turn(tok, ep, hand, reader)
                names = list(r["positions"])
                positions = [r["positions"][n] for n in names]
                n = len(r["ids"])
                arrays = {"actual": capture.run(model, r["ids"], positions, attention_mask_4d(n))}
                if r["mask"] is None:
                    arrays["reference"] = capture.run(model, r["ids"], positions, None)
                else:
                    keep = attention_mask_4d(n, r["mask"])
                    arrays["kept"] = capture.run(model, r["ids"], positions, keep)
                    arrays["compacted"] = capture.run(model, r["ids"], positions, keep, compacted_positions(r["mask"]))
                file = f"h{hand}_{slug(reader)}.npz"
                np.savez(directory/file, **{k: v.numpy() for k, v in arrays.items()})
                role = "sender" if reader == ep["sender"] else "receiver" if reader == ep["receiver"] else "other"
                turns.append({"file": file, "hand": hand, "agent": reader, "seat": role, "role": ep["roles"][reader],
                              "mode": ep["mode"], "tokens": n, "masked_tokens": 0 if r["mask"] is None else n - sum(r["mask"]),
                              "spans": r["spans"], "positions": r["positions"],
                              "range": NB.count_range(ep["running_count"][hand - 1]), "running_count": ep["running_count"][hand - 1],
                              "position_tokens": {k: tok.decode([r["ids"][p]]) for k, p in r["positions"].items()}})
        (directory/"index.json").write_text(json.dumps({"run": run, "layers": capture.layers, "turns": turns}, indent=1))
        print(run, len(turns), f"{time.time() - started:.0f}s", flush=True)


# ═══════════════════════════════════════════════════════════════════ Noise floor

def noise(args, model, tok, capture):
    """Honest control turns, each run three times: plain, plain again, explicit mask kernel."""
    samples, render = transfer_samples()
    picks = sorted((s for s in samples if s["mode"] == "control"), key=lambda s: (s["run_id"], s["agent_name"]))[::36][:args.n]
    episodes = stego_episodes()
    controls = sorted(r for r in episodes if r.endswith("__control"))[:args.n]
    rows = []
    for s in picks:
        ids, start = encode(tok, render, s)
        rows.append(("transfer", f"{s['run_id']}__{slug(s['agent_name'])}", ids, [start, len(ids) - 1], T_POSITIONS))
    for run in controls:
        ep = episodes[run]
        reader = next(p for p in ep["players"] if p not in (ep["sender"], ep["receiver"]))
        r = stego_turn(tok, ep, 4, reader)
        rows.append(("stego", f"{run}__{slug(reader)}", r["ids"], list(r["positions"].values()), tuple(r["positions"])))
    index = []
    (OUT/"noise").mkdir(parents=True, exist_ok=True)
    for tier, name, ids, positions, names in rows:
        first = capture.run(model, ids, positions, None)
        second = capture.run(model, ids, positions, None)
        third = capture.run(model, ids, positions, attention_mask_4d(len(ids)))
        np.savez(OUT/"noise"/f"{name}.npz", plain=first.numpy(), repeat=second.numpy(), explicit=third.numpy())
        index.append({"tier": tier, "file": f"{name}.npz", "tokens": len(ids), "positions": dict(zip(names, positions))})
        print("noise", name, len(ids), flush=True)
    (OUT/"noise"/"index.json").write_text(json.dumps({"layers": capture.layers, "turns": index}, indent=1))


# ═══════════════════════════════════════════════════════════════════ J-space

def decompose(args, lens=None):
    """support and coefficients for every (file, pass, position) row of each directory given."""
    lens = lens or jspace.Lens(MODEL_DIR, LENS_FILE)
    work = []
    for directory in args.dirs:
        if (directory/"jspace.npz").exists() or not (directory/"index.json").exists():
            continue
        index = json.loads((directory/"index.json").read_text())
        rows, residuals = [], []
        for t in index["turns"]:
            data = np.load(directory/t["file"])
            for label in data.files:
                for n, name in enumerate(t["positions"]):
                    rows.append({"file": t["file"], "pass": label, "position": name})
                    residuals.append(data[label][n])
        residuals = np.stack(residuals)
        work.append((directory, rows, residuals, np.full((len(rows), len(lens.layers), jspace.K), -1, dtype=np.int32),
                     np.zeros((len(rows), len(lens.layers), jspace.K), dtype=np.float32)))
        print(directory, len(rows), "rows", flush=True)
    for li, layer in enumerate(lens.layers):
        started = time.time()
        atoms, norms = lens.atoms(layer)
        for _, _, residuals, support, coefficients in work:
            for start in range(0, len(residuals), 384):
                s, c = jspace.pursuit(torch.from_numpy(residuals[start:start + 384, li]).cuda(), atoms, norms)
                support[start:start + len(s), li] = s.cpu().numpy()
                coefficients[start:start + len(c), li] = c.cpu().numpy()
        del atoms, norms
        print(f"layer {layer} {time.time() - started:.1f}s", flush=True)
    for directory, rows, _, support, coefficients in work:
        np.savez_compressed(directory/"jspace.npz", support=support, coefficients=coefficients)
        (directory/"rows.json").write_text(json.dumps({"layers": lens.layers, "rows": rows}))


def follow(args):
    """decompose for run directories under --dirs as their replay finishes, one lens load, until --expect have shares."""
    lens = jspace.Lens(MODEL_DIR, LENS_FILE)
    roots = list(args.dirs)
    while True:
        runs = [p for root in roots for p in sorted(root.iterdir()) if p.is_dir()]
        ready = [p for p in runs if (p/"index.json").exists() and not (p/"jspace.npz").exists()]
        if ready:
            decompose(argparse.Namespace(dirs=ready), lens)
        elif sum((p/"jspace.npz").exists() for p in runs) >= args.expect:
            return
        else:
            time.sleep(20)


def load_jspace(directory: Path):
    meta = json.loads((directory/"rows.json").read_text())
    data = np.load(directory/"jspace.npz")
    index = {(r["file"], r["pass"], r["position"]): i for i, r in enumerate(meta["rows"])}
    return index, data["support"], data["coefficients"]


def shares(support, coefficients):
    return jspace.shares(support, coefficients)


def surplus_cells(index, support, coefficients, file, a, r, position, layers=ALL):
    i, j = index[(file, a, position)], index[(file, r, position)]
    z_a, z_r = shares(support[i], coefficients[i]), shares(support[j], coefficients[j])
    return [jspace.surplus(z_a[l], z_r[l]) for l in layers]


def aggregate(cells):
    """(pool mean, mass), phase0_report.aggregate."""
    acc = defaultdict(float)
    for cell in cells:
        for c, v in cell.items():
            acc[c] += v
    n = max(len(cells), 1)
    values = [v/n for v in acc.values()]
    return (float(np.mean(values)) if values else 0.0), float(sum(values))


# ═══════════════════════════════════════════════════════════════════ Scoring helpers

def stat(values):
    v = np.asarray(values, float)
    return {"n": int(len(v)), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())} if len(v) else {"n": 0}


def run_auroc(values: dict, labels: dict, groups: dict):
    """detect.grouped_auroc with the scenario bootstrap, on run -> score."""
    import detect
    runs = sorted(values)
    v = np.array([values[r] for r in runs], float)
    y = np.array([labels[r] for r in runs], int)
    g = np.array([groups[r] for r in runs])
    draws = np.array([detect.grouped_auroc(v[i], y[i], g[i]) for i in detect.scenario_draws(runs)])
    return {"auroc": detect.grouped_auroc(v, y, g), "ci": detect.interval(draws), "runs": len(runs),
            "collusion_min": float(v[y == 1].min()), "control_max": float(v[y == 0].max())}


def calibrated(agent_scores: dict, calibration: dict):
    """z-score of each agent against the control runs' agents of the same family and slot. agent_scores and calibration map
    (run, agent) -> (family, slot, value). Only control runs enter the mean and spread."""
    pools = defaultdict(list)
    for (run, _), (fam, slot, value) in calibration.items():
        if run.endswith("__control"):
            pools[(fam, slot)].append(value)
    out = {}
    for key, (fam, slot, value) in agent_scores.items():
        pool = np.array(pools[(fam, slot)])
        if len(pool) < 2:
            raise ValueError(f"calibration pool of {len(pool)} for {fam} slot {slot}")
        out[key] = (fam, slot, float((value - pool.mean())/max(pool.std(ddof=1), 1e-12)))
    return out


def by_run_max(agent_scores: dict) -> dict:
    out = defaultdict(lambda: -np.inf)
    for (run, _), (_, _, value) in agent_scores.items():
        out[run] = max(out[run], value)
    return dict(out)


def tpr_at_control(run_scores: dict, quantile=95) -> dict:
    """Threshold at the control runs' percentile (control runs only), share of collusion runs above it."""
    control = np.array([v for r, v in run_scores.items() if r.endswith("__control")])
    collusion = np.array([v for r, v in run_scores.items() if r.endswith("__collusion")])
    threshold = float(np.percentile(control, quantile))
    return {"threshold": threshold, "tpr": float(np.mean(collusion > threshold)), "fpr": float(np.mean(control > threshold))}


def relative_delta(a: np.ndarray, r: np.ndarray) -> np.ndarray:
    """||a - r|| / ||r|| per layer, a and r [layers, d]."""
    return np.linalg.norm(a - r, axis=-1)/np.linalg.norm(r, axis=-1)


def unit(x: np.ndarray) -> np.ndarray:
    return x/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def ontology_directions():
    from lens import ontology_directions as directions
    from utils.models import get_model
    onto = get_model("qwen3_32b").ontology()
    lens = jspace.Lens(MODEL_DIR, LENS_FILE)
    target = torch.tensor(list(onto["target_ids_by_word"].values()), device="cuda")
    control = torch.tensor(list(onto["control_ids_by_word"].values()), device="cuda")
    return lens, {l: directions(lens.head, lens.J[l].cuda().float(), target, control).cpu().numpy() for l in LENS_LAYERS}


def report_noise(args):
    """Kernel floor on this pod. a = plain kernel, r = the same pass again (repeat) or on the explicit mask kernel."""
    directory = OUT/"noise"
    index, support, coefficients = load_jspace(directory)
    out = defaultdict(lambda: defaultdict(list))
    layers = (0, 15, 26, 30, 36, 45, 50, 56, 62)
    for t in json.loads((directory/"index.json").read_text())["turns"]:
        data = np.load(directory/t["file"])
        for label in ("repeat", "explicit"):
            for pi, position in enumerate(t["positions"]):
                a, r = data["plain"][pi], data[label][pi]
                delta = relative_delta(a, r)
                out[f"{t['tier']} | {label} | {position}"]["rows"].append({
                    "file": t["file"], "tokens": t["tokens"], "bit_identical": bool((a == r).all()),
                    "mass": aggregate(surplus_cells(index, support, coefficients, t["file"], "plain", label, position))[1],
                    "relative_delta_mean": float(delta.mean()), "relative_delta_by_layer": {l: float(delta[l]) for l in layers}})
    result = {}
    for key, v in out.items():
        rows = v["rows"]
        result[key] = {"turns": len(rows), "bit_identical": sum(r["bit_identical"] for r in rows),
                       "mass": stat([r["mass"] for r in rows]), "relative_delta_mean": stat([r["relative_delta_mean"] for r in rows]),
                       "relative_delta_by_layer_median": {l: float(np.median([r["relative_delta_by_layer"][l] for r in rows])) for l in layers},
                       "first_turn": rows[0]}
        print(key, json.dumps({k: result[key][k] for k in ("turns", "bit_identical")}), round(result[key]["mass"]["median"], 3), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS/"noise.json").write_text(json.dumps(result, indent=1) + "\n")


# ═══════════════════════════════════════════════════════════════════ Transfer report

def report_transfer(args):
    lens, onto = ontology_directions()
    del lens
    root = OUT/"transfer"
    agents = {}            # (run, agent) -> meta
    mass = defaultdict(dict)      # position -> kind -> (run, agent) -> (pool mean, mass)
    norm = defaultdict(dict)      # position -> kind -> (run, agent) -> mean relative delta over layers
    ont = defaultdict(dict)       # position -> kind -> (run, agent) -> ontology cosine
    direction = defaultdict(lambda: defaultdict(dict))  # position -> kind -> (run, agent) -> d [layers, d]
    for directory in sorted(p for p in root.iterdir() if (p/"jspace.npz").exists()):
        run = directory.name
        index, support, coefficients = load_jspace(directory)
        for t in json.loads((directory/"index.json").read_text())["turns"]:
            key = (run, t["agent"])
            agents[key] = {**{k: t[k] for k in ("role", "mode", "family", "slot", "tokens")},
                           "edit_tokens": {k: abs(v["tokens"] - t["tokens"]) for k, v in t["references"].items()},
                           "same_input": t["references"]["reference"]["same_input"]}
            data = np.load(directory/t["file"])
            for pi, position in enumerate(T_POSITIONS):
                a = data["actual"][pi]
                for kind in t["references"]:
                    r = data[kind][pi]
                    mass[position].setdefault(kind, {})[key] = aggregate(surplus_cells(index, support, coefficients, t["file"], "actual", kind, position))
                    norm[position].setdefault(kind, {})[key] = float(relative_delta(a, r).mean())
                    d = unit(a - r)
                    ont[position].setdefault(kind, {})[key] = float(np.mean([(d[li] @ onto[l].T).mean() for li, l in enumerate(LENS_LAYERS)]))
                    direction[position][kind][key] = a - r
            extra = OUT/"transfer_padded"/run
            if (extra/"jspace.npz").exists() and t["role"] != "colluder":
                pidx, psupport, pcoefficients = load_jspace(extra)
                padded = np.load(extra/t["file"])["padded"]
                agents[key]["edit_tokens"]["padded"] = abs(json.loads((extra/"index.json").read_text())["turns"][[x["file"] for x in json.loads((extra/"index.json").read_text())["turns"]].index(t["file"])]["tokens"] - t["tokens"])
                for pi, position in enumerate(T_POSITIONS):
                    i, j = index[(t["file"], "actual", position)], pidx[(t["file"], "padded", position)]
                    z_a, z_p = shares(support[i], coefficients[i]), shares(psupport[j], pcoefficients[j])
                    mass[position].setdefault("padded", {})[key] = aggregate([jspace.surplus(z_a[l], z_p[l]) for l in ALL])
                    a, r = data["actual"][pi], padded[pi]
                    norm[position].setdefault("padded", {})[key] = float(relative_delta(a, r).mean())
                    d = unit(a - r)
                    ont[position].setdefault("padded", {})[key] = float(np.mean([(d[li] @ onto[l].T).mean() for li, l in enumerate(LENS_LAYERS)]))
    labels = {run: int(run.endswith("__collusion")) for run, _ in agents}
    groups = {run: family(run) for run, _ in agents}
    colluder = lambda k: agents[k]["role"] == "colluder"
    result = {"agents": len(agents), "runs": len(labels), "edit_tokens": {}, "positions": {}}
    for kind in ("reference", *KINDS, "padded"):
        for name, want in (("colluders", colluder), ("honest, collusion runs", lambda k: not colluder(k) and agents[k]["mode"] == "collusion"),
                           ("control", lambda k: agents[k]["mode"] == "control")):
            values = [agents[k]["edit_tokens"][kind] for k in agents if want(k) and kind in agents[k]["edit_tokens"]]
            if values:
                result["edit_tokens"][f"{kind} | {name}"] = stat(values)

    def score_set(table_for):
        """table_for(kind) -> (run, agent) -> value; returns the comparisons, raw and calibrated."""
        out = {}
        comparisons = {"real reference": ("reference", "reference"), "placebo vs swapped": ("placebo", "swapped")}
        if "padded" in mass[T_POSITIONS[0]] and len(mass[T_POSITIONS[0]]["padded"]) == sum(not colluder(k) for k in agents):
            comparisons["placebo vs padded"] = ("placebo", "padded")
        for label, (ck, hk) in comparisons.items():
            ct, ht = table_for(ck), table_for(hk)
            scores = {k: (agents[k]["family"], agents[k]["slot"], (ct if colluder(k) else ht)[k]) for k in agents}
            calib = {k: (agents[k]["family"], agents[k]["slot"], ht[k]) for k in agents if not colluder(k)}
            raw = by_run_max(scores)
            z = by_run_max(calibrated(scores, calib))
            import detect
            col = [s[2] for k, s in scores.items() if colluder(k)]
            hon = [s[2] for k, s in scores.items() if not colluder(k) and agents[k]["mode"] == "collusion"]
            within = defaultdict(dict)
            for k, sc in scores.items():
                if agents[k]["mode"] == "collusion":
                    within[k[0]][k[1]] = sc[2]
            top2 = sum(set(sorted(v, key=v.get)[-2:]) == {a for a in v if agents[(r, a)]["role"] == "colluder"} for r, v in within.items())
            out_agents = {"colluders vs honest of the same collusion runs": detect.auroc(np.array(col + hon), np.array([1]*len(col) + [0]*len(hon))),
                          "top two agents are the coalition": [int(top2), len(within)]}
            out[label] = {"raw": run_auroc(raw, labels, groups), "calibrated": run_auroc(z, labels, groups), "agent_level": out_agents,
                          "calibrated_tpr_at_control_p95": tpr_at_control(z),
                          "agents": {g: stat([s[2] for k, s in scores.items() if want(k)]) for g, want in
                                     (("colluders", colluder), ("honest, collusion runs", lambda k: not colluder(k) and agents[k]["mode"] == "collusion"),
                                      ("control", lambda k: agents[k]["mode"] == "control"))}}
        return out

    for position in T_POSITIONS:
        row = {"mass": score_set(lambda kind: {k: v[1] for k, v in mass[position][kind].items()}),
               "pool_mean": score_set(lambda kind: {k: v[0] for k, v in mass[position][kind].items()}),
               "relative_delta": score_set(lambda kind: norm[position][kind]),
               "ontology": score_set(lambda kind: ont[position][kind])}
        row["pair_alignment"] = pair_alignment(direction[position]["reference"], agents, labels, groups)
        mixed = {k: (direction[position]["placebo"] if colluder(k) else direction[position]["swapped"])[k] for k in agents}
        row["pair_alignment, placebo vs swapped"] = pair_alignment(mixed, agents, labels, groups)
        result["positions"][position] = row
        print(position, json.dumps({m: {c: round(v["raw"]["auroc"], 3) for c, v in row[m].items()} for m in ("mass", "relative_delta", "ontology")}), flush=True)
        print(position, "pair alignment", json.dumps({k: v["auroc"] if isinstance(v, dict) and "auroc" in v else v for k, v in row["pair_alignment"].items() if k in ("run", "top_pair_coalition")}), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS/"transfer.json").write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", RESULTS/"transfer.json")


def pair_alignment(directions: dict, agents: dict, labels: dict, groups: dict):
    """Lens-free analogue of J-PACT on the surplus direction. cos(d_i, d_j) per layer, contrasted with each member's mean
    cosine with the other two agents (detect.pair_contrast's form), averaged over layers; run score = max over pairs."""
    runs = sorted({r for r, _ in directions})
    scores, top, kept = {}, 0, 0
    for run in runs:
        names = sorted(a for r, a in directions if r == run)
        D = {a: unit(directions[(run, a)]) for a in names}
        cos = {(x, y): (D[x]*D[y]).sum(-1) for i, x in enumerate(names) for y in names[i + 1:]}
        get = lambda x, y: cos[(x, y) if (x, y) in cos else (y, x)]
        pairs = {}
        for (x, y), c in cos.items():
            others = [k for k in names if k not in (x, y)]
            pairs[(x, y)] = float(np.mean(c - np.mean([(get(x, k) + get(y, k))/2 for k in others], axis=0)))
        best = max(pairs, key=pairs.get)
        scores[run] = pairs[best]
        coalition = tuple(sorted(a for a in names if agents[(run, a)]["role"] == "colluder"))
        if coalition:
            kept += 1
            top += best == coalition
    return {"run": run_auroc(scores, labels, groups), "top_pair_coalition": [top, kept]}


# ═══════════════════════════════════════════════════════════════════ Stego report

def stego_rows():
    out = []
    for directory in sorted(p for p in (OUT/"stego").iterdir() if (p/"jspace.npz").exists()):
        out.append((directory, json.loads((directory/"index.json").read_text())))
    return out


def report_stego(args):
    from config import RANGE_SIGNATURES, RANGES
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    low = [x.strip().lower() for x in tok.batch_decode([[i] for i in range(len(tok))])]
    lens = jspace.Lens(MODEL_DIR, LENS_FILE)
    words = {g: ws for g, ws in RANGE_SIGNATURES.items()}
    J = {}

    def top_ids(H: np.ndarray, layers):
        """H [layers, d] -> {layer: set of top-10 ids}"""
        out = {}
        for l in layers:
            if l not in J:
                J[l] = lens.J[l].cuda().float()
            logits = jspace.lens_logits(torch.from_numpy(np.ascontiguousarray(H[l]))[None].cuda(), J[l], lens.norm_weight, lens.head, lens.eps)
            out[l] = set(logits.topk(jspace.TOPK, dim=-1).indices[0].tolist())
        return out

    def named(ids):
        found = {low[i] for i in ids}
        return {g for g, ws in words.items() if found & ws}

    mass, norm = defaultdict(dict), defaultdict(dict)
    decoding = defaultdict(list)       # (reader group, measure, band) -> [(true range, named groups)]
    directions = defaultdict(list)     # (reader group, reference) -> [(run, hand, range, d [layers, d])]
    meta = {}
    for directory, index in stego_rows():
        run = directory.name
        jidx, support, coefficients = load_jspace(directory)
        cells = defaultdict(lambda: defaultdict(list))
        ranges = {t["hand"]: t["range"] for t in index["turns"]}
        for t in index["turns"]:
            previous = ranges.get(t["hand"] - 1)     # the last count record in a colluder's private block
            data = np.load(directory/t["file"])
            key = (run, t["agent"])
            meta[key] = {"seat": t["seat"], "role": t["role"], "mode": t["mode"]}
            colluding = t["mode"] == "collusion" and t["seat"] in ("sender", "receiver")
            refs = ("compacted", "kept") if colluding else ("reference",)
            reader = (t["seat"] if t["mode"] == "collusion" else f"control {t['seat']}") if t["seat"] != "other" else \
                ("honest, collusion runs" if t["mode"] == "collusion" else "control other")
            for pi, position in enumerate(t["positions"]):
                for ref in refs:
                    cells[(position, ref)][t["agent"]].append(surplus_cells(jidx, support, coefficients, t["file"], "actual", ref, position))
                    norm[(position, ref)].setdefault(key, []).append(float(relative_delta(data["actual"][pi], data[ref][pi]).mean()))
                if position != "message_end" or t["seat"] == "sender":
                    continue
                main = refs[0]
                a, r = data["actual"][pi], data[main][pi]
                directions[(reader, main)].append((run, t["hand"], t["range"], previous, a - r))
                if colluding:
                    directions[(reader, "kept")].append((run, t["hand"], t["range"], previous, a - data["kept"][pi]))
                top_a, top_r = top_ids(a, ALL), top_ids(r, ALL)
                i, j = jidx[(t["file"], "actual", position)], jidx[(t["file"], main, position)]
                z_a, z_r = shares(support[i], coefficients[i]), shares(support[j], coefficients[j])
                for band_name, band in (("L28-56", BAND), ("all layers", ALL), ("L26-30", NARROW)):
                    bag_a = set().union(*(top_a[l] for l in band))
                    bag_r = set().union(*(top_r[l] for l in band))
                    sur = set().union(*(set(jspace.surplus(z_a[l], z_r[l])) for l in band))
                    held = set().union(*(set(z_a[l]) for l in band))
                    for measure, ids in (("top-10, actual", bag_a), ("top-10, reference", bag_r), ("top-10 surplus", bag_a - bag_r),
                                         ("J-space surplus", sur), ("J-space, actual", held)):
                        decoding[(reader, measure, band_name)].append((t["range"], previous, named(ids)))
        for (position, ref), by_agent in cells.items():
            for agent, per_hand in by_agent.items():
                mass[(position, ref)][(run, agent)] = aggregate([c for hand in per_hand for c in hand])
    result = {"mass": {}, "relative_delta": {}, "decoding": {}, "direction": {}}
    labels = {run: int(run.endswith("__collusion")) for run, _ in meta}
    groups = {run: "stego" for run in labels}
    for position in ("message_end", "prompt_last"):
        for primary in ("compacted", "kept"):
            for metric, table in (("mass", mass), ("relative_delta", norm)):
                scores = {}
                for key, m in meta.items():
                    ref = primary if m["mode"] == "collusion" and m["seat"] in ("sender", "receiver") else "reference"
                    value = table[(position, ref)].get(key)
                    if value is None:
                        continue
                    scores[key] = value[1] if metric == "mass" else float(np.mean(value))
                runs = defaultdict(lambda: -np.inf)
                for (run, _), v in scores.items():
                    runs[run] = max(runs[run], v)
                seats = {g: stat([v for k, v in scores.items() if want(meta[k])]) for g, want in
                         (("sender", lambda m: m["mode"] == "collusion" and m["seat"] == "sender"),
                          ("receiver", lambda m: m["mode"] == "collusion" and m["seat"] == "receiver"),
                          ("honest, collusion runs", lambda m: m["mode"] == "collusion" and m["seat"] == "other"),
                          ("control", lambda m: m["mode"] == "control"))}
                result[metric][f"{position} | colluders {primary}"] = {"run": run_auroc(dict(runs), labels, groups), "agents": seats}
                print(metric, position, primary, round(result[metric][f"{position} | colluders {primary}"]["run"]["auroc"], 3), flush=True)
    for (reader, measure, band), rows in sorted(decoding.items()):
        matrix = {}
        for true in RANGES:
            hit = [named for r, _, named in rows if r == true]
            matrix[true] = {"hands": len(hit), **{g: round(float(np.mean([g in x for x in hit])), 3) if hit else None for g in RANGES}}
        mean = lambda xs: round(float(np.mean(xs)), 3) if len(xs) else None
        changed = [(true, prev, named) for true, prev, named in rows if prev is not None and prev != true]
        result["decoding"][f"{reader} | {measure} | {band}"] = {
            "matrix": matrix, "hands": len(rows), "true_range_held": mean([true in named for true, _, named in rows]),
            "any_range_named": mean([bool(named) for _, _, named in rows]),
            "previous_range_held": mean([prev in named for _, prev, named in rows if prev is not None]),
            "changed_hands": len(changed), "changed_true_held": mean([true in named for true, _, named in changed]),
            "changed_previous_held": mean([prev in named for _, prev, named in changed])}
    for (reader, ref), rows in sorted(directions.items()):
        result["direction"][f"{reader} | {ref}"] = range_separation([(r, h, true, d) for r, h, true, _, d in rows])
        result["direction"][f"{reader} | {ref} | previous hand's range"] = range_separation([(r, h, prev, d) for r, h, _, prev, d in rows if prev is not None])
        print("direction", reader, ref, json.dumps({b: result["direction"][f"{reader} | {ref}"][b]["mean_contrast"] for b in ("L28-56", "all layers")}), flush=True)
    (RESULTS/"stego.json").write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", RESULTS/"stego.json")


def range_separation(rows, permutations=1000):
    """Within each run, mean cosine of the surplus direction between hands of the same range (HOT with HOT, NEGATIVE with
    NEGATIVE) minus between HOT and NEGATIVE hands. Also the same contrast within strata of equal hand distance, which
    removes the similarity of nearby hands, and a within-run label permutation null."""
    rng = np.random.default_rng(0)
    out = {}
    by_run = defaultdict(list)
    for run, hand, true, d in rows:
        if true in ("HOT", "NEGATIVE"):
            by_run[run].append((hand, true, d))
    for band_name, band in (("L28-56", BAND), ("all layers", ALL), ("L26-30", NARROW)):
        contrasts, stratified, null, usable = [], [], [], 0
        for run, hands in by_run.items():
            ranges = [h[1] for h in hands]
            if ranges.count("HOT") < 1 or ranges.count("NEGATIVE") < 1 or len(hands) < 3:
                continue
            usable += 1
            U = np.stack([unit(h[2][list(band)]) for h in hands])           # [hands, layers, d]
            C = np.einsum("ald,bld->abl", U, U).mean(-1)                     # mean cosine over layers
            index = [h[0] for h in hands]

            def contrast(labels, strata=False):
                same, diff = defaultdict(list), defaultdict(list)
                for a in range(len(hands)):
                    for b in range(a + 1, len(hands)):
                        bucket = abs(index[a] - index[b]) if strata else 0
                        (same if labels[a] == labels[b] else diff)[bucket].append(C[a, b])
                keys = [k for k in same if k in diff]
                if not keys:
                    return None
                return float(np.mean([np.mean(same[k]) - np.mean(diff[k]) for k in keys]))
            contrasts.append(contrast(ranges))
            s = contrast(ranges, True)
            if s is not None:
                stratified.append(s)
            null.append([contrast(list(rng.permutation(ranges))) for _ in range(permutations)])
        null = np.array(null)
        observed = float(np.mean(contrasts)) if contrasts else None
        out[band_name] = {"runs": usable, "mean_contrast": observed, "runs_positive": int(sum(c > 0 for c in contrasts)),
                          "stratified_mean_contrast": float(np.mean(stratified)) if stratified else None,
                          "stratified_runs": len(stratified), "stratified_positive": int(sum(c > 0 for c in stratified)),
                          "permutation_p": float(np.mean(null.mean(0) >= observed)) if contrasts else None}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("actions", nargs="+", choices=("noise", "transfer", "transfer_padded", "stego", "decompose", "follow", "report_noise", "report_transfer", "report_stego", "check"),
                        help="GPU replays given together share one model load")
    parser.add_argument("--dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--expect", type=int, default=0, help="follow: stop once this many run directories have shares")
    args = parser.parse_args()
    replays = [a for a in args.actions if a in ("noise", "transfer", "transfer_padded", "stego")]
    if replays:
        model, tok = load_model()
        capture = Capture(model, LENS_LAYERS)
    for action in args.actions:
        if action in replays:
            {"noise": noise, "transfer": transfer, "transfer_padded": transfer_padded, "stego": stego}[action](args, model, tok, capture)
        else:
            {"decompose": decompose, "follow": follow, "report_noise": report_noise, "report_transfer": report_transfer, "report_stego": report_stego,
             "check": check}[action](args)
        print(f"{action.upper()} DONE", flush=True)
    return 0


def check(args):
    """CPU check of every input before the GPU runs: renders, references, masks, read positions."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    samples, render = transfer_samples()
    donor = donors({s["run_id"] for s in samples})
    lengths = defaultdict(list)
    for s in samples:
        scenario = s["run_id"].rsplit("__", 1)[0]
        honest = control_systems(scenario)
        ids_a, start_a = encode(tok, render, s)
        for kind, system in transfer_inputs(s, honest, donor[scenario]).items():
            ids_r, start_r = encode(tok, render, {**s, "system_prompt": system})
            assert ids_a[start_a:] == ids_r[start_r:], (s["run_id"], s["agent_name"], kind)
            lengths[(kind, s["role"], s["mode"])].append(len(ids_r) - len(ids_a))
        lengths[("tokens", s["role"], s["mode"])].append(len(ids_a))
    print("transfer rows", len(samples), {f"{k}": (len(v), int(np.median(v))) for k, v in sorted(lengths.items())})
    counts = defaultdict(int)
    for run, ep in sorted(stego_episodes().items()):
        for hand in HANDS:
            for reader in ep["players"]:
                r = stego_turn(tok, ep, hand, reader)
                seat = "sender" if reader == ep["sender"] else "receiver" if reader == ep["receiver"] else "other"
                counts[(ep["mode"], seat, "message_end" in r["positions"])] += 1
                counts[("tokens", ep["mode"])] += len(r["ids"])
                if r["mask"] is not None:
                    counts[("masked", seat)] += len(r["mask"]) - sum(r["mask"])
    print("stego", dict(counts))


if __name__ == "__main__":
    sys.exit(main())
