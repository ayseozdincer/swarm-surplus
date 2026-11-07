#!/usr/bin/env python3
"""Markdown tables for results/lane_b/REPORT.md from results/lane_b/{noise,transfer,stego}.json.

  python scripts/lane_b_tables.py noise|transfer|stego
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1]/"results/lane_b"


def ci(row):
    return f"{row['auroc']:.3f} [{row['ci'][0]:.2f}, {row['ci'][1]:.2f}]"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def spread(s, digits=3):
    return f"{s['median']:.{digits}f} ({s['min']:.{digits}f} to {s['max']:.{digits}f})" if s.get("n") else "-"


def noise():
    data = json.loads((RESULTS/"noise.json").read_text())
    rows = []
    for key, v in sorted(data.items()):
        tier, label, position = key.split(" | ")
        rows.append([tier, "same kernel twice" if label == "repeat" else "plain vs explicit mask", position, v["turns"],
                     v["bit_identical"], f"{v['mass']['median']:.3f}", f"{v['mass']['max']:.3f}",
                     f"{v['relative_delta_mean']['median']:.4f}",
                     f"{v['first_turn']['mass']:.3f} / {v['first_turn']['relative_delta_mean']:.4f}"])
    print(table(["Tier", "Comparison", "Position", "Turns", "Bit-identical", "Mass, median", "Mass, max",
                 "Relative residual delta, median", "First turn, mass / delta"], rows))
    print()
    layers = list(next(iter(data.values()))["relative_delta_by_layer_median"])
    rows = [[key] + [f"{v['relative_delta_by_layer_median'][l]:.4f}" for l in layers]
            for key, v in sorted(data.items()) if "explicit" in key]
    print(table(["Plain vs explicit, median by layer"] + [f"L{l}" for l in layers], rows))


def transfer():
    data = json.loads((RESULTS/"transfer.json").read_text())
    metrics = (("mass", "Surplus mass (J-space)"), ("pool_mean", "Surplus pool mean (J-space)"),
               ("relative_delta", "Surplus direction norm, lens-free"), ("ontology", "Ontology on the surplus"))
    rows = []
    for position, p in data["positions"].items():
        for key, label in metrics:
            for comparison in ("real reference", "placebo vs swapped"):
                c = p[key][comparison]
                rows.append([label, position, comparison, ci(c["raw"]), ci(c["calibrated"]),
                             f"{c['calibrated_tpr_at_control_p95']['tpr']:.2f}"])
        pa = p["pair_alignment"]
        rows.append(["Pair alignment of surplus directions, lens-free", position, "real reference", ci(pa["run"]), "-",
                     f"top pair is the coalition {pa['top_pair_coalition'][0]} of {pa['top_pair_coalition'][1]}"])
    print(table(["Detector", "Position", "References", "Run AUROC, raw", "Run AUROC, calibrated on control runs",
                 "TPR at control p95, calibrated"], rows))
    print()
    rows = []
    for position, p in data["positions"].items():
        for key, label in metrics:
            for comparison in ("real reference", "placebo vs swapped"):
                c = p[key][comparison]["agent_level"]
                t = c["top two agents are the coalition"]
                rows.append([label, position, comparison, f"{c['colluders vs honest of the same collusion runs']:.3f}", f"{t[0]} of {t[1]}"])
    print(table(["Detector", "Position", "References", "Agent AUROC, colluders vs honest agents of the same runs", "Top two agents are the coalition"], rows))
    print()
    rows = []
    for position, p in data["positions"].items():
        for comparison in ("real reference", "placebo vs swapped"):
            a = p["mass"][comparison]["agents"]
            r = p["relative_delta"][comparison]["agents"]
            for group in ("colluders", "honest, collusion runs", "control"):
                rows.append([position, comparison, group, a[group]["n"], spread(a[group]), spread(r[group], 4)])
    print(table(["Position", "References", "Agents", "n", "Mass, median (range)", "Direction norm, median (range)"], rows))
    print()
    rows = [[k, v["n"], v["median"], f"{v['min']} to {v['max']}"] for k, v in data["edit_tokens"].items()]
    print(table(["Reference and agents", "n", "Input length change, median tokens", "Range"], rows))
    print()
    rows = []
    for position, p in data["positions"].items():
        a = p["ontology"]["real reference"]["agents"]
        rows.append([position] + [spread(a[g], 4) for g in ("colluders", "honest, collusion runs", "control")])
    print(table(["Position", "Ontology cosine, colluders", "honest, collusion runs", "control"], rows))


def stego():
    data = json.loads((RESULTS/"stego.json").read_text())
    rows = []
    for metric, label in (("mass", "Surplus mass (J-space)"), ("relative_delta", "Surplus direction norm, lens-free")):
        for key, v in data[metric].items():
            position, reference = key.split(" | ")
            a = v["agents"]
            rows.append([label, position, reference.replace("colluders ", ""), ci(v["run"]), spread(a["receiver"]), spread(a["sender"]),
                         spread(a["honest, collusion runs"]), spread(a["control"])])
    print(table(["Detector", "Position", "Colluders' reference", "Run AUROC", "Receiver", "Sender", "Honest, collusion runs", "Control"], rows))
    print()
    rows = []
    for key, v in data["decoding"].items():
        reader, measure, band = key.split(" | ")
        m = v["matrix"]
        rows.append([reader, measure, band, v["hands"], v["true_range_held"], v["previous_range_held"],
                     f"{v['changed_true_held']} / {v['changed_previous_held']} ({v['changed_hands']})", v["any_range_named"]]
                    + [f"{m[t]['NEGATIVE']} / {m[t]['NEUTRAL']} / {m[t]['HOT']} ({m[t]['hands']})" for t in ("NEGATIVE", "NEUTRAL", "HOT")])
    print(table(["Reader", "Measure", "Band", "Hands", "True range held", "Previous hand's range held",
                 "Range changed: true / previous held (hands)", "Any range named",
                 "True NEGATIVE: names NEG / NEU / HOT (hands)", "True NEUTRAL", "True HOT"], rows))
    print()
    rows = []
    for key, v in data["direction"].items():
        for band, b in v.items():
            rows.append([key, band, b["runs"], f"{b['mean_contrast']:.4f}" if b["mean_contrast"] is not None else "-", b["runs_positive"],
                         f"{b['stratified_mean_contrast']:.4f}" if b["stratified_mean_contrast"] is not None else "-",
                         f"{b['stratified_positive']} of {b['stratified_runs']}", b["permutation_p"]])
    print(table(["Reader and reference", "Band", "Runs", "Same minus different range, mean cosine", "Runs positive",
                 "Within equal hand distance", "Runs positive", "Permutation p"], rows))


if __name__ == "__main__":
    {"noise": noise, "transfer": transfer, "stego": stego}[sys.argv[1]]()
