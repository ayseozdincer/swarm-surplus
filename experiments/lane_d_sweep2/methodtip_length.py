#!/usr/bin/env python3
"""Lane D. Is the method_tip plateau the note's length or its content? A tip note is three times a leak note's length, so the plateau
count alone cannot say. New references on the target's solve turn, every reply token, delta over floor as experiments/lane_d_sweep2/methodtip_found.py:
  tip branch    note    only the tip note masked
                head    only the note's first N tokens masked, N the length of the leak note of the same seed
                tail    only its last N tokens masked
                designer  a random designer span as long as the note masked instead (lane C's placebo, experiments/lane_c_sweep14b/blind.py on its
                          branch: any designer-provenance system or user message, never the first 8 tokens)
  leak branch   note, and designer at the leak note's length
If head and tail hold the plateau, length does not set it; if designer stays near zero at the tip's length, masking that much does not.
Writes experiments/lane_d_sweep2/results/methodtip_length.json. With --csv-seeds it also writes lane C's placebo for the blind target's solve turn in leak
and benign_k1 of those sweep2 seeds, as trace_placebo_lanec.npz beside the replay, for experiments/lane_d_sweep2/trace_csv.py.

  SWARM_MODEL=qwen3_14b python experiments/lane_d_sweep2/methodtip_length.py --csv-seeds seed5034 seed5046
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]
from detector.render import find_turn, load_turns, render_turn, token_span  # noqa: E402
import replay  # noqa: E402

REPLAY = Path(os.environ.get("SWEEP2_REPLAY", ROOT/"data/replay_sweep2"))
RUNS = Path(os.environ.get("SWEEP2_RUNS", ROOT/"runs"))
ARM = "sweep2_methodtip"
BAND = list(range(16, 39))
FIRST, WINDOW, FACTOR = 256, 16, 10
SEED, FIRST_KEEP = 20261107, 8


def designer_spans(tok, turn, full, offsets):
    """Token spans of the content of every designer-provenance system or user message, wrapper tokens excluded (lane C's)."""
    messages, meta, tools = turn["messages"], turn["message_meta"], turn["tools"]
    spans = []
    for i, tag in enumerate(meta):
        if tag["provenance"] != "designer" or messages[i]["role"] not in ("system", "user"):
            continue
        upto = tok.apply_chat_template(messages[:i + 1], tools=tools, tokenize=False, add_generation_prompt=False)
        if not full.startswith(upto) or not upto.endswith("<|im_end|>\n"):
            continue
        block = upto.rfind("<|im_start|>")
        start, end = upto.index("\n", block) + 1, len(upto) - len("<|im_end|>\n")
        if end > start:
            spans.append((i, *token_span(offsets, start, end)))
    return spans


def designer_keep(tok, turn, r, length, key):
    offsets = [tuple(o) for o in tok(r["text"], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]]
    starts = [s for _, a, b in designer_spans(tok, turn, r["text"], offsets) for s in range(max(a, FIRST_KEEP), b - length + 1)]
    start = starts[np.random.default_rng([SEED, zlib.crc32(key.encode())]).integers(len(starts))]
    return keep_all_but(len(r["ids"]), start, start + length)


def keep_all_but(n, a, b):
    keep = [1]*n
    keep[a:b] = [0]*(b - a)
    return keep


def reply_states(model, capture, ids, start, keep=None):
    """The band's residuals at every token from `start`, [tokens, layers, d_model] on the GPU."""
    capture.trace_from = start
    capture.run(model, ids, [len(ids) - 1], replay.attention_mask_4d(len(ids), keep))
    states = capture.trace()
    capture.trace_from, capture.traced = None, {}
    return states


def relative(actual, reference):
    """||actual - reference|| / ||actual|| per token and band layer, layer by layer since a long reply does not fit twice as float32."""
    rows = [torch.linalg.vector_norm(actual[:, i].float() - reference[:, i].float(), dim=-1)/torch.linalg.vector_norm(actual[:, i].float(), dim=-1)
            for i in range(actual.shape[1])]
    return torch.stack(rows, dim=1).cpu().numpy().astype(np.float32)


def scores(delta, floor):
    series = (delta/floor[None, BAND]).mean(axis=1)
    w = min(WINDOW, len(series))
    means = np.convolve(series, np.ones(w)/w, mode="valid")
    return {"count": int((series[:FIRST] > FACTOR).sum()), "count_all": int((series > FACTOR).sum()), "window": float(means.max()),
            "mean_first256": float(series[:FIRST].mean()), "mean_after": float(series[FIRST:].mean()) if len(series) > FIRST else None}


def med(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-seeds", nargs="*", default=[])
    args = parser.parse_args()
    behaviour = json.loads((ROOT/"experiments/lane_d_sweep2/results/methodtip_behaviour.json").read_text())["rows"]
    floor = np.array(json.loads((ROOT/"experiments/lane_d_sweep2/results/methodtip_scores.json").read_text())["floor"])
    model, tok = replay.load_model()
    capture = replay.Capture(model, BAND)
    for seed in args.csv_seeds:
        for branch in ("leak", "benign_k1"):
            d = REPLAY/"sweep2"/seed/branch
            agent = next(t for t in json.loads((d/"index.json").read_text())["turns"] if t["phase"] == "solve" and t.get("trace_from") is not None)["agent"]
            turn = find_turn(load_turns(RUNS/"sweep2"/seed/branch), agent, 2, "solve")
            r = render_turn(tok, turn)
            actual = reply_states(model, capture, r["ids"], r["prompt_tokens"])
            length = len(r["mask"]) - sum(r["mask"])
            placebo = reply_states(model, capture, r["ids"], r["prompt_tokens"], designer_keep(tok, turn, r, length, f"{seed}/{branch}/{agent}"))
            np.savez(d/"trace_placebo_lanec.npz", delta=relative(actual, placebo), band=np.array(BAND), masked_tokens=length)
            del actual, placebo
            print("csv", seed, branch, length, flush=True)
    rows = []
    for b in behaviour:
        seed, target = b["seed"], b["target"]
        spans = {}
        for branch in ("leak_solvable", "method_tip"):
            turn = find_turn(load_turns(RUNS/ARM/seed/branch), target, 2, "solve")
            r = render_turn(tok, turn)
            note = json.loads((RUNS/ARM/seed/branch/"run.json").read_text())["plant"]["entries"][0]["text"]
            span = replay.text_token_span(tok, r, note)
            assert span, (seed, branch)
            spans[branch] = (turn, r, span)
        n_leak = spans["leak_solvable"][2][1] - spans["leak_solvable"][2][0]
        for branch, (turn, r, (a, z)) in spans.items():
            ids, n, start = r["ids"], len(r["ids"]), r["prompt_tokens"]
            references = {"note": keep_all_but(n, a, z), "designer": designer_keep(tok, turn, r, z - a, f"{seed}/{branch}/{target}")}
            if branch == "method_tip":
                references["head"] = keep_all_but(n, a, a + n_leak)
                references["tail"] = keep_all_but(n, z - n_leak, z)
            actual = reply_states(model, capture, ids, start)
            row = {"seed": seed, "branch": branch, "note_tokens": z - a, "leak_note_tokens": n_leak, "followed": b["followed"], "cites": b["cites"]}
            for name, keep in references.items():
                row[name] = scores(relative(actual, reply_states(model, capture, ids, start, keep)), floor)
            rows.append(row)
            del actual
            torch.cuda.empty_cache()
            print(seed, branch, z - a, {k: row[k]["count"] for k in references}, flush=True)
    table = {}
    for branch, names in (("method_tip", ("note", "head", "tail", "designer")), ("leak_solvable", ("note", "designer"))):
        sel = [r for r in rows if r["branch"] == branch]
        table[branch] = {"n": len(sel), "note_tokens": med([r["note_tokens"] for r in sel]),
                         **{name: {k: med([r[name][k] for r in sel]) for k in ("count", "count_all", "window", "mean_first256", "mean_after")} for name in names}}
    tips = {r["seed"]: r for r in rows if r["branch"] == "method_tip"}
    leaks = {r["seed"]: r for r in rows if r["branch"] == "leak_solvable"}
    paired = {"head above leak note": sum(tips[s]["head"]["count"] > leaks[s]["note"]["count"] for s in tips),
              "tail above leak note": sum(tips[s]["tail"]["count"] > leaks[s]["note"]["count"] for s in tips),
              "tip note above its designer": sum(tips[s]["note"]["count"] > tips[s]["designer"]["count"] for s in tips),
              "leak note above its designer": sum(leaks[s]["note"]["count"] > leaks[s]["designer"]["count"] for s in tips), "of": len(tips)}
    out = {"table": table, "paired": paired, "rows": rows}
    (ROOT/"experiments/lane_d_sweep2/results/methodtip_length.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"table": table, "paired": paired}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
