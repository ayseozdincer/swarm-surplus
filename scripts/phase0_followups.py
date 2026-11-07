#!/usr/bin/env python3
"""W4 follow-ups on the Core replay. Run from third_party/maj-lens on the pod. Writes results/phase0_followups.json.

  collect     extra forward passes, one kernel throughout (explicit 4D mask), teacher-forced on the same response tokens
  decompose   sparse J-space shares of those references
  report      the four groups, ontology on the surplus, and the edge test

Placebo. Honest prompts are identical up to the name in all 31 control runs, one preamble and one table brief, so "a different
honest persona's prompt" would be the real reference again. The donor is therefore the honest brief of a different scenario in
a different domain: no collusion content, about the size of a private block.
  placebo   colluders: own preamble + donor brief, private block removed          (input: real reference, brief swapped)
  padded    honest: own prompt + donor brief appended                             (as requested)
  swapped   honest: own preamble + donor brief in place of their own              (added here: the same edit colluders get)

Edge test. E(j->i, t) = |S_i(t) & H_j(t') - A_i(t)| / |S_i(t)|, on dense top-k concept sets over L26-30.
  S_i(t)  concepts in i's top-k in the actual pass and not the reference
  H_j(t') concepts in j's top-k at j's most recent turn before t
  A_i(t)  concepts of the tokens in i's reference input
Only colluders have a private-channel turn, so letting H use those turns gives the coalition away before anything is scored.
`public` uses public turns for everyone and is the fair test. `with_private` adds private-channel turns, as requested.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]
from detector import jspace  # noqa: E402
from detector.core_reference import honest_systems, reference_sample  # noqa: E402
from phase0 import POSITIONS, SCENARIOS, core_samples, encode, forward, slug  # noqa: E402
from phase0_report import aggregate, load  # noqa: E402
from replay import LENS_FILE, LENS_LAYERS, MODEL_DIR, Capture, load_model  # noqa: E402
from surplus_table import shares  # noqa: E402

OUT = ROOT/"data/phase0/followups"
CORE = ROOT/"data/phase0/core"
RESULT = ROOT/"results/phase0_followups.json"
BAND = (26, 27, 28, 29, 30)
KINDS = ("placebo", "padded", "swapped")
NULL_DRAWS = 200


def donors(samples) -> dict[str, str]:
    """scenario run -> honest brief of the next scenario in a different domain, sorted order, deterministic."""
    domain = {s["run_id"]: s["scenario_domain"] for s in samples}
    runs = sorted(r for r in domain if r.endswith("__collusion"))
    out = {}
    for i, run in enumerate(runs):
        donor = next(runs[(i + n) % len(runs)] for n in range(1, len(runs)) if domain[runs[(i + n) % len(runs)]] != domain[run])
        systems = honest_systems(SCENARIOS, donor)
        name, prompt = sorted(systems.items())[0]
        brief = prompt.split("\n\n", 1)[1]
        if any(n in brief for n in systems):
            raise ValueError(f"donor brief names an agent: {donor}")
        out[run] = brief
        out[run.rsplit("__", 1)[0] + "__control"] = brief
    return out


def placebo_input(sample: dict, kind: str, systems: dict, brief: str) -> dict:
    own = systems[sample["agent_name"]]
    preamble = own.split("\n\n", 1)[0]
    if kind == "padded":
        return {**sample, "system_prompt": own + "\n\n" + brief}
    base = reference_sample(sample, systems)
    return {**base, "system_prompt": preamble + "\n\n" + brief}


def collect(args):
    model, tok = load_model()
    capture = Capture(model, LENS_LAYERS)
    samples, render = core_samples()
    brief = donors(samples)
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, index = defaultdict(list), defaultdict(list)
    for n, s in enumerate(samples):
        systems = honest_systems(SCENARIOS, s["run_id"])
        ids_a, start_a = encode(tok, render, s)
        kinds = ("placebo",) if s["role"] == "colluder" else ("padded", "swapped")
        for kind in kinds:
            ids_r, start_r = encode(tok, render, placebo_input(s, kind, systems, brief[s["run_id"]]))
            if ids_a[start_a:] != ids_r[start_r:]:
                raise ValueError(f"response tokens differ: {s['run_id']} {s['agent_name']} {s['round']} {kind}")
            arrays[kind].append(forward(capture, model, ids_r, [start_r, len(ids_r) - 1], True).numpy())
            index[kind].append({"run": s["run_id"], "file": f"{slug(s['agent_name'])}_r{s['round']}.npz", "agent": s["agent_name"],
                                "round": s["round"], "role": s["role"], "mode": s["mode"], "domain": s["scenario_domain"],
                                "tokens_actual": len(ids_a), "tokens_reference": len(ids_r)})
        if n % 100 == 0:
            print("collect", n, len(samples), flush=True)
    # Colluders' private-channel turns, the actual pass only.
    for run in sorted({s["run_id"] for s in samples if s["mode"] == "collusion"}):
        prompts = json.loads((SCENARIOS/run/"agent_prompts.json").read_text())
        results = json.loads((SCENARIOS/run/"results.json").read_text())
        private = [p for p in prompts if p["channel"] == "private"]
        for p, message in zip(private, results["private_messages"]):
            response = message.split("]: ", 1)[1] if message.startswith("[") else message
            ids, start = encode(tok, render, {"system_prompt": p["system_prompt"], "user_prompt": p["user_prompt"], "response": response})
            arrays["private"].append(forward(capture, model, ids, [start, len(ids) - 1], True).numpy())
            index["private"].append({"run": run, "agent": p["agent_name"], "round": 0, "role": "colluder", "mode": "collusion"})
    for kind, rows in arrays.items():
        np.save(OUT/f"{kind}.npy", np.stack(rows))
        (OUT/f"{kind}.json").write_text(json.dumps(index[kind]))
        print(kind, len(rows), flush=True)


def decompose(args):
    lens = jspace.Lens(MODEL_DIR, LENS_FILE)
    data = {k: np.load(OUT/f"{k}.npy", mmap_mode="r") for k in KINDS}
    support = {k: np.full((len(v), 2, len(LENS_LAYERS), jspace.K), -1, dtype=np.int32) for k, v in data.items()}
    coefficients = {k: np.zeros((len(v), 2, len(LENS_LAYERS), jspace.K), dtype=np.float32) for k, v in data.items()}
    flat_s = {k: v.reshape(-1, len(LENS_LAYERS), jspace.K) for k, v in support.items()}
    flat_c = {k: v.reshape(-1, len(LENS_LAYERS), jspace.K) for k, v in coefficients.items()}
    for li, layer in enumerate(LENS_LAYERS):
        atoms, norms = lens.atoms(layer)
        for k, v in data.items():
            rows = np.ascontiguousarray(v[:, :, li]).reshape(-1, v.shape[-1])
            for start in range(0, len(rows), 384):
                s, c = jspace.pursuit(torch.from_numpy(rows[start:start + 384]).cuda(), atoms, norms)
                flat_s[k][start:start + len(s), li] = s.cpu().numpy()
                flat_c[k][start:start + len(c), li] = c.cpu().numpy()
        del atoms, norms
        print("layer", layer, flush=True)
    for k in KINDS:
        np.savez_compressed(OUT/f"{k}_jspace.npz", support=support[k], coefficients=coefficients[k])


def groups_mass():
    """Surplus mass per agent for the five groups, per position, pooled over rounds and layers as in the W4 table."""
    extra = {k: (json.loads((OUT/f"{k}.json").read_text()), np.load(OUT/f"{k}_jspace.npz")) for k in KINDS}
    rows = {k: {(r["run"], r["file"]): i for i, r in enumerate(extra[k][0])} for k in KINDS}
    out = {p: defaultdict(lambda: defaultdict(list)) for p in POSITIONS}
    meta_of = {}
    for directory in sorted(p for p in CORE.iterdir() if (p/"jspace.npz").exists()):
        layers, index, support, coefficients = load(directory)
        turns = json.loads((directory/"index.json").read_text())["turns"]
        for t in turns:
            meta_of[(directory.name, t["agent"])] = {"role": t["role"], "mode": t["mode"], "domain": t["domain"]}
            for pi, position in enumerate(POSITIONS):
                i, j = index[(t["file"], "actual", position)], index[(t["file"], "reference", position)]
                z_a, z_r = shares(support[i], coefficients[i]), shares(support[j], coefficients[j])
                key = (directory.name, t["agent"])
                out[position]["real"][key] += [jspace.surplus(a, r) for a, r in zip(z_a, z_r)]
                for kind in KINDS:
                    n = rows[kind].get((directory.name, t["file"]))
                    if n is None:
                        continue
                    z = extra[kind][1]
                    z_k = shares(z["support"][n, pi], z["coefficients"][n, pi])
                    out[position][kind][key] += [jspace.surplus(a, r) for a, r in zip(z_a, z_k)]
    scored = {p: {g: {k: aggregate(v) for k, v in rows_.items()} for g, rows_ in by.items()} for p, by in out.items()}
    return scored, meta_of


def stat(values):
    v = np.array(values)
    return {"n": len(v), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def ontology_scores():
    """Cosine of the surplus direction h_actual - h_reference with the ontology directions, mean over words then layers."""
    from lens import ontology_directions
    from utils.models import get_model
    onto = get_model("qwen3_32b").ontology()
    lens = jspace.Lens(MODEL_DIR, LENS_FILE)
    target = torch.tensor(list(onto["target_ids_by_word"].values()), device="cuda")
    control = torch.tensor(list(onto["control_ids_by_word"].values()), device="cuda")
    extra = {k: (json.loads((OUT/f"{k}.json").read_text()), np.load(OUT/f"{k}.npy", mmap_mode="r")) for k in KINDS}
    rows = {k: {(r["run"], r["file"]): i for i, r in enumerate(extra[k][0])} for k in KINDS}
    directions = {l: ontology_directions(lens.head, lens.J[l].cuda().float(), target, control) for l in LENS_LAYERS}
    scores = {p: defaultdict(lambda: defaultdict(list)) for p in POSITIONS}
    for directory in sorted(p for p in CORE.iterdir() if (p/"index.json").exists()):
        for t in json.loads((directory/"index.json").read_text())["turns"]:
            d = np.load(directory/t["file"])
            key = (directory.name, t["agent"])
            pairs = [("real", d["reference"])] + [(k, extra[k][1][rows[k][(directory.name, t["file"])]]) for k in KINDS if (directory.name, t["file"]) in rows[k]]
            for group, reference in pairs:
                delta = torch.from_numpy(d["actual"] - np.asarray(reference)).cuda()
                unit = torch.nn.functional.normalize(delta, dim=-1)
                for pi, position in enumerate(POSITIONS):
                    cos = [float((unit[pi, li] @ directions[l].T).mean()) for li, l in enumerate(LENS_LAYERS)]
                    scores[position][group][key].append(float(np.mean(cos)))
    return {p: {g: {k: float(np.mean(v)) for k, v in rows_.items()} for g, rows_ in by.items()} for p, by in scores.items()}


def concept(tok_text: str) -> str:
    return tok_text.strip().lower()


def edge_test():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    lens = jspace.Lens(MODEL_DIR, LENS_FILE)
    samples, render = core_samples()
    private_index = json.loads((OUT/"private.json").read_text())
    private = np.load(OUT/"private.npy", mmap_mode="r")
    decode = {}

    def name(t):
        if t not in decode:
            decode[t] = concept(tok.decode([t]))
        return decode[t]

    def bags(H, k):
        """H [positions, layers, d]: per position the concept set over the band."""
        out = []
        for pi in range(H.shape[0]):
            found = set()
            for layer in BAND:
                top = lens.top(torch.from_numpy(np.ascontiguousarray(H[pi, LENS_LAYERS.index(layer)]))[None], layer, k=k).indices[0].tolist()
                found.update(name(t) for t in top)
            out.append(found - {""})
        return out

    runs = sorted({s["run_id"] for s in samples if s["mode"] == "collusion"})
    results = {}
    for k in (10, 25):
        per_run = {}
        for run in runs:
            directory = CORE/run
            turns = json.loads((directory/"index.json").read_text())["turns"]
            prompts = json.loads((SCENARIOS/run/"agent_prompts.json").read_text())
            order = [(p["agent_name"], p["round"]) for p in prompts if p["channel"] == "public" and p["phase"] == "discussion"]
            by = {(t["agent"], t["round"]): t for t in turns}
            systems = honest_systems(SCENARIOS, run)
            act, ref, inputs = {}, {}, {}
            for key in order:
                d = np.load(directory/by[key]["file"])
                act[key], ref[key] = bags(d["actual"], k), bags(d["reference"], k)
                s = next(x for x in samples if x["run_id"] == run and x["agent_name"] == key[0] and x["round"] == key[1])
                ids, _ = encode(tok, render, reference_sample(s, systems))
                inputs[key] = {name(t) for t in ids}
            priv = {r["agent"]: bags(private[i], k) for i, r in enumerate(private_index) if r["run"] == run}
            per_run[run] = {"order": order, "act": act, "ref": ref, "inputs": inputs, "private": priv,
                            "coalition": sorted({t["agent"] for t in turns if t["role"] == "colluder"}), "agents": sorted({t["agent"] for t in turns})}
        for position_index, position in enumerate(POSITIONS):
            for variant in ("public", "with_private"):
                results[f"k={k} | {position} | {variant}"] = score_edges(per_run, position_index, variant)
                print(k, position, variant, {a: b for a, b in results[f"k={k} | {position} | {variant}"].items() if a in ("top_pair_coalition", "top_pair_complement", "precision_at_5pct_fpr")}, flush=True)
    return results


def edges_of(run, position_index, variant, rng=None):
    """Mean E(j->i) over i's turns, for each ordered pair. With rng, H_j comes from a random public turn of j instead."""
    order = run["order"]
    scores = defaultdict(list)
    sizes = []
    for n, key in enumerate(order):
        i = key[0]
        S = (run["act"][key][position_index] - run["ref"][key][position_index]) - run["inputs"][key]
        sizes.append(len(S))
        for j in run["agents"]:
            if j == i:
                continue
            earlier = [x for x in order[:n] if x[0] == j]
            H = None
            if rng is not None:
                pool = [x for x in order if x[0] == j]
                H = run["act"][pool[rng.integers(len(pool))]][position_index]
            elif variant == "with_private" and j in run["private"] and not earlier:
                H = run["private"][j][position_index]
            elif earlier:
                H = run["act"][earlier[-1]][position_index]
                if variant == "with_private" and j in run["private"]:
                    H = H | run["private"][j][position_index]
            if H is None:
                continue
            scores[(j, i)].append(len(S & H)/len(S) if S else 0.0)
    return {k: float(np.mean(v)) for k, v in scores.items()}, sizes


def pair_scores(edges, agents):
    out = {}
    for a in range(len(agents)):
        for b in range(a + 1, len(agents)):
            x, y = agents[a], agents[b]
            out[(x, y)] = float(np.mean([edges.get((x, y), 0.0), edges.get((y, x), 0.0)]))
    return out


def score_edges(per_run, position_index, variant):
    top_exact = top_complement = 0
    positive, negative, sizes = [], [], {"colluder": [], "honest": []}
    for run in per_run.values():
        edges, s = edges_of(run, position_index, variant)
        for key, size in zip(run["order"], s):
            sizes["colluder" if key[0] in run["coalition"] else "honest"].append(size)
        pairs = pair_scores(edges, run["agents"])
        best = max(pairs, key=pairs.get)
        coalition = tuple(run["coalition"])
        complement = tuple(a for a in run["agents"] if a not in coalition)
        top_exact += best == coalition
        top_complement += best == complement
        for (j, i), value in edges.items():
            (positive if j in coalition and i in coalition else negative).append(value)
    threshold = float(np.percentile(negative, 95))
    tp, fp = sum(v > threshold for v in positive), sum(v > threshold for v in negative)
    rng = np.random.default_rng(0)
    null_top, null_gap = [], []
    if variant == "public":
        for _ in range(NULL_DRAWS):
            hits = 0
            gaps = []
            for run in per_run.values():
                edges, _ = edges_of(run, position_index, variant, rng)
                pairs = pair_scores(edges, run["agents"])
                hits += max(pairs, key=pairs.get) == tuple(run["coalition"])
                gaps.append(pairs[tuple(run["coalition"])] - np.mean([v for k, v in pairs.items() if k != tuple(run["coalition"])]))
            null_top.append(hits)
            null_gap.append(float(np.mean(gaps)))
    observed_gap = []
    for run in per_run.values():
        pairs = pair_scores(edges_of(run, position_index, variant)[0], run["agents"])
        observed_gap.append(pairs[tuple(run["coalition"])] - np.mean([v for k, v in pairs.items() if k != tuple(run["coalition"])]))
    return {"runs": len(per_run), "top_pair_coalition": int(top_exact), "top_pair_complement": int(top_complement),
            "positive_edges": len(positive), "negative_edges": len(negative), "threshold": threshold, "true_positives": int(tp), "false_positives": int(fp),
            "precision_at_5pct_fpr": tp/(tp + fp) if tp + fp else None, "recall_at_5pct_fpr": tp/len(positive),
            "edge_median_positive": float(np.median(positive)), "edge_median_negative": float(np.median(negative)),
            "surplus_set_size": {k: stat(v) for k, v in sizes.items()}, "mean_pair_gap": float(np.mean(observed_gap)),
            "null_top_pair_mean": float(np.mean(null_top)) if null_top else None, "null_top_pair_p95": float(np.percentile(null_top, 95)) if null_top else None,
            "null_gap_mean": float(np.mean(null_gap)) if null_gap else None, "null_gap_p95": float(np.percentile(null_gap, 95)) if null_gap else None}


def report(args):
    import detect
    scored, meta_of = groups_mass()
    onto = ontology_scores()
    result = {"mass": {}, "ontology": {}, "input_tokens": {}}
    for kind in KINDS:
        rows = json.loads((OUT/f"{kind}.json").read_text())
        result["input_tokens"][kind] = stat([abs(r["tokens_actual"] - r["tokens_reference"]) for r in rows])

    def groups(table):
        real = table["real"]
        pick = lambda source, want: [v for k, v in source.items() if want(meta_of[k])]
        colluder, honest = lambda m: m["role"] == "colluder", lambda m: m["role"] != "colluder"
        return {"colluders, real reference": pick(real, colluder), "colluders, placebo": pick(table["placebo"], colluder),
                "honest, own reference": pick(real, honest), "honest, padded": pick(table["padded"], honest), "honest, swapped": pick(table["swapped"], honest)}

    def agent_auroc(a, b):
        return detect.auroc(np.array(a + b), np.array([1]*len(a) + [0]*len(b)))

    def run_auroc(colluder_group, honest_group, table, index):
        runs = sorted({k[0] for k in meta_of})
        values, labels, domains = [], [], []
        for run in runs:
            scores = [(table[colluder_group] if meta_of[k]["role"] == "colluder" else table[honest_group])[k] for k in meta_of if k[0] == run]
            values.append(max((s[index] if isinstance(s, tuple) else s) for s in scores))
            labels.append(int(run.endswith("__collusion")))
            domains.append(next(meta_of[k]["domain"] for k in meta_of if k[0] == run))
        values, labels, domains = np.array(values), np.array(labels), np.array(domains)
        draws = np.array([detect.grouped_auroc(values[i], labels[i], domains[i]) for i in detect.scenario_draws(runs)])
        return {"auroc": detect.grouped_auroc(values, labels, domains), "ci": detect.interval(draws)}

    for position in POSITIONS:
        table = scored[position]
        mass_only = {g: {k: v[1] for k, v in rows.items()} for g, rows in table.items()}
        g = groups(mass_only)
        result["mass"][position] = {"groups": {k: stat(v) for k, v in g.items()},
                                    "agent_auroc": {"placebo vs padded": agent_auroc(g["colluders, placebo"], g["honest, padded"]),
                                                    "placebo vs swapped": agent_auroc(g["colluders, placebo"], g["honest, swapped"]),
                                                    "real vs own": agent_auroc(g["colluders, real reference"], g["honest, own reference"])},
                                    "run_auroc": {"placebo vs padded": run_auroc("placebo", "padded", table, 1), "placebo vs swapped": run_auroc("placebo", "swapped", table, 1)}}
        o = groups(onto[position])
        result["ontology"][position] = {"groups": {k: stat(v) for k, v in o.items()},
                                        "agent_auroc": {"real vs padded": agent_auroc(o["colluders, real reference"], o["honest, padded"]),
                                                        "placebo vs padded": agent_auroc(o["colluders, placebo"], o["honest, padded"]),
                                                        "placebo vs swapped": agent_auroc(o["colluders, placebo"], o["honest, swapped"]),
                                                        "real vs own": agent_auroc(o["colluders, real reference"], o["honest, own reference"])},
                                        "run_auroc": {"placebo vs padded": run_auroc("placebo", "padded", onto[position], 0), "placebo vs swapped": run_auroc("placebo", "swapped", onto[position], 0)}}
    RESULT.write_text(json.dumps(result, indent=1) + "\n")
    result["edges"] = edge_test()
    RESULT.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", RESULT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("collect", "decompose", "report"))
    args = parser.parse_args()
    {"collect": collect, "decompose": decompose, "report": report}[args.action](args)
