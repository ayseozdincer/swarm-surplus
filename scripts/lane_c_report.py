#!/usr/bin/env python3
"""Markdown tables for the lane C report from results/lane_c/readout*.json and kernel_floor.json, in the shapes of the 32B
W3 and W7 sections of results/OVERNIGHT.md. CPU only.

  python scripts/lane_c_report.py results/lane_c/readout.json > results/lane_c/tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("none", "benign_k1", "benign_k3", "leak", "proposal_k1", "proposal_k3")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(out)


def f(x, n=3):
    return "-" if x is None else f"{x:.{n}f}"


def med(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def code(text):
    return "`" + str(text).replace("\n", "\\n").replace("|", "\\|").replace("`", "'") + "`"


def floor_section(kernel, R):
    show = R["show"]
    out = [f"Kernel floor, one honest turn ({kernel['turn']['run']}, {kernel['turn']['agent']}, {kernel['turn']['tokens']} tokens), every token. "
           f"Same kernel twice bit-identical: {kernel['repeat_bit_identical']}. Mask effect on {kernel['mask_effect']['run']}, "
           f"{kernel['mask_effect']['agent']}, every token after the first masked span ({kernel['mask_effect']['tokens_after_span']}).", ""]
    fl = R.get("floor") or {}
    every = (fl.get("every_reply_token") or {}).get("noise_median")
    rows = []
    for l in show:
        n, e = kernel["kernel_noise"]["median"][l], kernel["mask_effect"]["median"][l]
        rows.append([l, f(n, 4), f(kernel["kernel_noise"]["p95"][l], 4), f(e, 4), f(n/e, 2), f(every[l], 4) if every else "-"])
    out.append(table(["Layer", "Kernel noise, median", "Kernel noise, p95", "Mask effect, median", "Noise / effect",
                      f"Kernel noise, every reply token of the {fl.get('every_reply_token', {}).get('turns', 0)} none blind turns, median"], rows))
    positions = [p for p in ("prompt_last", "first_generated", "first_thought", "after_think", "mention_pre") if fl.get(p, {}).get("noise_median") and fl[p].get("effect_median")]
    if positions:
        out += ["", "At the read positions, kernel noise / mask effect / placebo effect, medians. Noise over the none branch's turns "
                "(plain-kernel second pass), the two effects over the blind agent's leak turn of every seed.", ""]
        rows = [[l] + [f"{fl[p]['noise_median'][l]:.4f} / {fl[p]['effect_median'][l]:.4f} / " + (f"{fl[p]['placebo_median'][l]:.4f}" if fl[p].get("placebo_median") else "-")
                       for p in positions] for l in show]
        out.append(table(["Layer"] + positions, rows))
    return "\n".join(out)


def leak_section(R):
    leak = R["leak"]
    seeds = sorted(leak)
    out = []
    rows = []
    for position, label in (("first_generated", "1, first generated"), ("after_think", "2, after `</think>`"), ("prompt_last", "4, last prompt token"),
                            ("answer_pre", "control, `answer_pre`"), ("mention_pre", "first mention, `mention_pre`")):
        cells = [leak[s]["positions"].get(position) for s in seeds]
        cells = [c for c in cells if c]
        if not cells:
            continue

        def count(key):
            have = [c for c in cells if c.get(key) is not None]
            return f"{sum(bool(c[key]) for c in have)} of {len(have)} ({sum(len(c[key]) for c in have)})" if have else "-"
        best = {k: med([c["rank_first"].get(k, {}).get("best") for c in cells]) for k in ("leak actual", "leak kept", "leak placebo", "none actual", "benign_k1 actual")}
        late = {k: med([c["rank_first"].get(k, {}).get("best_late") for c in cells]) for k in ("leak actual", "leak kept", "none actual", "benign_k1 actual")}
        tokens = sorted({code(c["token"]) for c in cells})
        rows.append([label, ", ".join(tokens[:3]) + (" ..." if len(tokens) > 3 else ""), count("top10_layers_actual"), count("top10_layers_kept"), count("jspace_layers"),
                     count("surplus_layers_kept"), count("surplus_layers_placebo"),
                     f"{f(best['leak actual'], 0)} / {f(best['leak kept'], 0)} / {f(best['leak placebo'], 0)}", f(best["none actual"], 0), f(best["benign_k1 actual"], 0),
                     f"{f(late['leak actual'], 0)} / {f(late['leak kept'], 0)}; {f(late['none actual'], 0)}; {f(late['benign_k1 actual'], 0)}"])
    out.append(table(["Position", "Token", "Seeds with an answer token in top-10, actual (layers)", "Same, kept reference", "Seeds with it in J-space (layers)",
                      "Seeds with surplus on it, kept (layers)", "Same, placebo", "Median best rank of the first answer token over layers, leak actual / kept / placebo",
                      "none", "benign_k1", "Median best late-layer rank, leak actual / kept; none; benign_k1"], rows))
    out += ["", "Per seed, best rank over layers of the answer's first token (layer), leak actual / kept reference; none; benign_k1 actual.", ""]
    rows = []
    for s in seeds:
        row = leak[s]
        cells = []
        for position in ("first_generated", "after_think", "prompt_last", "answer_pre", "mention_pre"):
            c = row["positions"].get(position)
            if not c:
                cells.append("-")
                continue
            r = c["rank_first"]
            g = lambda k: f"{r[k]['best']} (L{r[k]['best_layer']})" if k in r else "-"
            cells.append(f"{g('leak actual')} / {g('leak kept')}; {r.get('none actual', {}).get('best', '-')}; {r.get('benign_k1 actual', {}).get('best', '-')}")
        rows.append([s, row["blind"], code(row["answer"]), ", ".join(code(x) for x in row["planted"][:2])] + cells)
    out.append(table(["Seed", "Blind", "Answer", "First tokens", "first_generated", "after_think", "prompt_last", "answer_pre (control)", "mention_pre"], rows))
    return "\n".join(out)


def digit_section(R):
    d = R.get("digits") or {}
    if not d:
        return "No numeric answers."
    rows = []
    for s, row in sorted(d.items()):
        first = int(row["answer"].strip()[0])

        def cell(k):
            if k not in row:
                return "-"
            ranks = row[k]
            return f"{ranks[first]} / {min(r for i, r in enumerate(ranks) if i != first)}"
        rows.append([s, code(row["answer"]), cell("leak actual"), cell("leak kept"), cell("benign_k1 actual"), cell("benign_k1 kept"), cell("none actual")])
    return table(["Seed", "Answer", "leak actual, first digit / best other digit", "leak kept", "benign_k1 actual", "benign_k1 kept", "none"], rows)


def mass_section(R):
    out = []
    positions = ("prompt_last", "first_generated", "first_thought", "after_think", "answer_pre", "verdict_pre", "end_of_response", "mention_pre")
    for group, by in R["mass"].items():
        shown = [p for p in positions if any(f"{p} kept" in by.get(v, {}) for v in VARIANTS)]
        rows = []
        for v in VARIANTS:
            if v not in by:
                continue
            cells = []
            for p in shown:
                k, pl = by[v].get(f"{p} kept"), by[v].get(f"{p} placebo")
                cells.append("-" if not k else f"{f(k['median'])}" + (f" / {f(pl['median'])}" if pl else "") + f" ({k['turns']})")
            rows.append([v + (" (kernel noise only)" if v == "none" else "")] + cells)
        out += [f"{group.capitalize()} turns, median mass, kept" + (" / placebo" if group == "blind solve" else "") + " (turns).", "",
                table(["Branch"] + list(shown), rows), ""]
    return "\n".join(out)


def recovery_section(R):
    rec = R["recovery"]
    show = R["show"]
    branches = [b for b in VARIANTS if b in rec]
    out = ["Relative distance of the blind agent from the none branch at `prompt_last`, median over seeds, actual / kept. "
           "Kernel floor is the none turn's own plain-kernel delta.", ""]
    rows = []
    for l in show:
        cells = [" / ".join(f(med([x[k][l] for x in rec[b]["prompt_last"]])) for k in ("actual", "kept")) for b in branches]
        rows.append([l] + cells + [f(med([x["floor"][l] for x in rec["leak"]["prompt_last"]]), 4)])
    out.append(table(["Layer"] + branches + ["Kernel floor"], rows))
    out += ["", "Median over layers, then over seeds.", ""]
    rows = []
    for b in branches:
        for p in ("prompt_last", "first_generated", "first_thought", "after_think"):
            xs = rec[b].get(p) or []
            if xs:
                m = {k: med([float(np.median(x[k])) for x in xs]) for k in ("actual", "kept", "floor")}
                rows.append([b, p, f(m["actual"]), f(m["kept"]), f(m["floor"]), f"{sum(x['same_history'] for x in xs)} of {len(xs)}"])
    out.append(table(["Branch", "Position", "Actual", "Kept reference", "Kernel floor", "Same history as none"], rows))
    return "\n".join(out)


def placebo_section(R):
    P = R["placebo"]
    out = ["The placebo reference masks a random designer span (system prompt or task text) of the note span's token length, one "
           "draw per turn, instead of the note. Blind solve turns, median over seeds and layers. In none the kept reference is the "
           "plain kernel and the placebo the explicit one, so there the placebo also carries kernel noise.", ""]
    rows = []
    for v in VARIANTS:
        for p in ("prompt_last", "first_generated", "first_thought", "after_think", "answer_pre", "mention_pre", "end_of_response"):
            xs = P.get(v, {}).get(p) or []
            if not xs:
                continue
            m = lambda k: med([float(np.median(x[k])) for x in xs])
            rows.append([v, p, f(m("kept_delta")), f(m("placebo_delta")), f(m("placebo_from_kept")), f(m("cos_d"), 2),
                         f(med([x["mass_kept"] for x in xs])), f(med([x["mass_placebo"] for x in xs])), len(xs)])
    out.append(table(["Branch", "Position", "Delta, kept", "Delta, placebo", "Placebo reference from kept reference", "cos(d kept, d placebo)",
                      "Mass, kept", "Mass, placebo", "Seeds"], rows))
    return "\n".join(out)


def verdict_section(R, seed):
    v = R["verdict"][seed]
    rows = []
    for branch in ("none", "benign_k1", "leak", "proposal_k1"):
        for row in v.get(branch, []):
            rows.append([branch, row["agent"] + (" (blind)" if row["blind"] else ""), f(row["mass"]),
                         ", ".join(code(c) for c, _ in row["surplus_top"][:5]), ", ".join(code(c) for c in row["top10_band"][:5])])
    return table(["Branch", "Agent", "Surplus mass", "Top surplus concepts", "Top-10 over the mid band"], rows)


def mention_section(R, position):
    M = R["mention"]
    seeds = sorted(M)
    show = R["show"]
    rows = []
    for l in show:
        def collect(branch, key, i=None, j=None):
            xs = []
            for s in seeds:
                c = M[s]["positions"][position]["branches"].get(branch)
                if not c or key not in c:
                    continue
                v = c[key]
                if i is not None:
                    v = v[i]
                if j is not None:
                    v = v[j]
                xs.append(v[l])
            return xs
        floors = [M[s]["positions"][position]["floor"][l] for s in seeds if M[s]["positions"][position]["floor"]]
        ra, rr = collect("leak", "rank_planted", 0, 0), collect("leak", "rank_planted", 0, 1)
        other = collect("leak", "best_other_digit", 0)
        shares = [M[s]["positions"][position]["branches"]["leak"]["share_first_kept"][l] for s in seeds
                  if "share_first_kept" in M[s]["positions"][position]["branches"].get("leak", {})]
        rows.append([l, f(med(collect("leak", "delta_kept"))), f(med(floors)), f(med(collect("leak", "ratio_kept")), 1), f(med(collect("leak", "ratio_placebo")), 1),
                     f(med(collect("benign_k1", "ratio_kept")), 1), f"{sum(x == 0 for x in ra)} / {sum(x == 0 for x in rr)} of {len(ra)}", f"{f(med(ra), 0)} / {f(med(rr), 0)}",
                     f(med(other), 0) if other else "-", f(med(collect("benign_k1", "rank_planted", 0, 0)), 0), f(med(collect("none", "rank_planted", 0, 0)), 0),
                     f"{sum(a > b for a, b in shares)} of {len(shares)}", f(med([a for a, _ in shares])) + " / " + f(med([b for _, b in shares]))])
    out = [table(["Layer", "Delta, leak", "Floor", "Delta / floor, leak", "Same, placebo reference", "Delta / floor, benign_k1",
                  "Seeds with the first answer token at rank 0, actual / reference", "Median rank of it, actual / reference", "Best other digit, numeric seeds",
                  "Rank of it, benign_k1", "Rank of it, none", "Seeds with J-space surplus on it", "Median share, actual / reference"], rows)]
    out += ["", "Per seed.", ""]
    rows = []
    for s in seeds:
        P = M[s]["positions"][position]["branches"]
        c = P.get("leak")
        if not c:
            rows.append([s, code(M[s]["answer"]), "no such position", "", "", "", "", ""])
            continue
        ra, rr = np.array(c["rank_planted"][0][0]), np.array(c["rank_planted"][0][1])
        rows.append([s, code(M[s]["answer"]), code(c["token"]) + " → " + code(c.get("next_token") or ""),
                     f"{int(ra[R['late']].min())} / {int(rr[R['late']].min())}", ", ".join(f"L{l}" for l in np.flatnonzero(ra == 0)) or "none",
                     ", ".join(f"L{l}" for l in c.get("surplus_layers", [])) or "none",
                     " / ".join(f(P[b].get("mass_kept")) if b in P else "-" for b in ("leak", "benign_k1", "none")),
                     " / ".join(f(P[b].get("mass_placebo")) if b in P else "-" for b in ("leak", "benign_k1", "none"))])
    out.append(table(["Seed", "Answer", "Token → next", "Best late rank, actual / kept", "Layers at rank 0, actual", "J-space surplus layers, kept",
                      "Mass leak / benign_k1 / none, kept", "Same, placebo"], rows))
    return "\n".join(out)


def trace_section(R):
    T = R["trace"]
    seeds = sorted(T)
    out = ["Every reply token of the blind agent's solve turn. Delta over the every-token kernel floor, mean over the late band "
           "(L50-62 on 32B), from `first_generated`. Offsets are tokens after `first_generated`.", ""]
    rows = []
    for s in seeds:
        r = T[s].get("leak")
        if not r:
            continue
        rows.append([s, r["mention_offset"], code(r.get("mention_token") or ""), f(r.get("reference late before mention max"), 1), f(r.get("reference late at mention"), 1),
                     f(r.get("reference late max"), 1), r.get("reference late argmax_offset"), r.get("reference late first over 5x"),
                     f(r.get("placebo late max"), 1), f(r.get("placebo late at mention"), 1),
                     r.get("first top10 offset actual"), r.get("first top10 offset reference"),
                     f(T[s].get("none", {}).get("reference late max"), 1), f(T[s].get("benign_k1", {}).get("reference late max"), 1)])
    out.append(table(["Seed", "First mention offset", "Mention token", "Max before mention", "At mention_pre", "Max over the reply", "Its offset",
                      "First token over 5x", "Placebo max", "Placebo at mention", "First top-10 of the answer token, actual",
                      "Same, reference", "none, max", "benign_k1, max"], rows))
    keys = [k for k in ("cos leak signature, before mention", "cos leak signature, at mention", "cos benign_k1 signature, before mention",
                        "cos benign_k1 signature, at mention", "cos proposal_k1 signature, at mention") if any(k in T[s].get("leak", {}) for s in seeds)]
    if keys:
        out += ["", "Direction of each token's surplus against the notes' verdict-turn signatures at `first_generated`, late band, leak branch, "
                "median over seeds.", "", table(["Measure", "Median"], [[k, f(med([T[s].get("leak", {}).get(k) for s in seeds]))] for k in keys])]
    return "\n".join(out)


def direction_section(R, ref="kept"):
    D = R["direction"][ref]
    if "error" in D:
        return f"not run: {D['error']}"
    show = R["show"]
    out = [f"Readers are the free agents' {D['phase']} turns; the blind agent's own pairs use its solve turn.", ""]
    for p in ("prompt_last", "first_generated", "first_thought"):
        out += [f"`{p}`", ""]
        rows = [[k] + [f"{v[l]:.2f}" for l in show] for k, v in D["blind_pairs"][p].items()]
        out += [table(["Pair, blind agent, mean over seeds"] + [f"L{l}" for l in show], rows), ""]
    rows = [[p, band, f"{c['within_note_across_readers']:.3f} ({c['within_n']})", f"{c['same_reader_across_notes']:.3f} ({c['across_n']})"]
            for p in ("prompt_last", "first_generated", "first_thought") for band, c in D["agreement"][p].items()]
    out += ["Same content, different readers. Mean cosine (pairs).", "", table(["Position", "Band", "Within one note, across readers", "Same reader, across notes"], rows), ""]
    bands = list(D["source"]["prompt_last"])
    rows = [[p] + [f"{D['source'][p][b]['accuracy']:.2f}" for b in bands] for p in ("prompt_last", "first_generated", "first_thought")]
    n = D["source"]["prompt_last"][bands[0]]["n"]
    out += [f"Source without naming. Nearest note signature, {n} decisions per cell, chance {D['source']['prompt_last'][bands[0]]['chance']:.2f}.", "",
            table(["Position"] + bands, rows), ""]
    rows = [[branch, k + f" ({c['n']})"] + [f"{c['mean'][l]:.2f}" for l in show] for branch, cells in D["uptake"].items() for k, c in cells.items()]
    out += ["Uptake against exposure. Blind agent's direction at `mention_pre` against each note's signature at `first_generated`, mean over seeds.", "",
            table(["Blind agent's turn", "Signature"] + [f"L{l}" for l in show], rows)]
    return "\n".join(out)


def detection_section(R, position="first_generated", band="L50-62"):
    C = R["detection"]["claims"]
    keys = [("mass/floor, kept", f"mass/floor kept | {position} | all"), ("mass/floor, placebo", f"mass/floor placebo | {position} | all"),
            ("content", f"content | {position} | {band}"), ("source", f"source | {position} | {band}"),
            ("jump at mention_pre, kept", f"jump kept | mention_pre | {band}"), ("same, placebo", f"jump placebo | mention_pre | {band}"),
            ("trace max, reference", f"trace max reference | reply | {band}"), ("trace max, placebo", f"trace max placebo | reply | {band}"),
            ("naming rank (bonus)", f"naming rank | {position} | late"), ("naming lift (bonus)", f"naming lift | {position} | late"),
            ("naming over the reply (bonus)", "trace naming | reply | late")]
    rows = []
    for name, c in C.items():
        if not c["pairs"]:
            continue
        cells = []
        for _, k in keys:
            v = c.get(k)
            cells.append("-" if not v or v["auroc"] is None else f"{v['auroc']:.2f} ({v['paired_wins']}/{v['paired_n']})")
        rows.append([name, c["pairs"]] + cells)
    return table(["Claim", "Pairs"] + [k for k, _ in keys], rows)


def main() -> int:
    R = json.loads(Path(sys.argv[1]).read_text())
    kernel = json.loads((ROOT/"results/lane_c/kernel_floor.json").read_text())
    parts = [("Floor", lambda: floor_section(kernel, R)), ("Leak readout", lambda: leak_section(R)), ("Digits at after_think", lambda: digit_section(R)),
             ("Surplus mass", lambda: mass_section(R)), ("Reference recovery", lambda: recovery_section(R)), ("Placebo reference", lambda: placebo_section(R)),
             ("Verdict turns, first seed", lambda: verdict_section(R, R["seeds"][0])),
             ("First mention, mention_pre", lambda: mention_section(R, "mention_pre")), ("First mention, first_generated", lambda: mention_section(R, "first_generated")),
             ("Per-token trace", lambda: trace_section(R)), ("Direction test, kept", lambda: direction_section(R, "kept")),
             ("Detection, first_generated, late band", lambda: detection_section(R)),
             ("Detection, prompt_last, late band", lambda: detection_section(R, "prompt_last")),
             ("Detection, first_generated, mid band", lambda: detection_section(R, "first_generated", "L26-30")),
             ("Detection, mention_pre, late band", lambda: detection_section(R, "mention_pre"))]
    for title, make in parts:
        try:
            print(f"### {title}\n\n{make()}\n")
        except (KeyError, TypeError, ValueError, IndexError) as error:
            print(f"### {title}\n\nnot available: {error!r}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
