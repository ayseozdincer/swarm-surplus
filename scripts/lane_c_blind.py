#!/usr/bin/env python3
"""Lane C stage 1. The blind agent's solve turn of one episode in every branch of every seed given, with every token of its
reply read in the same passes, and a placebo reference.

  SWARM_MODEL=qwen3_14b python scripts/lane_c_blind.py runs/sweep14b/seed3000 ... --out-root data/replay_14b/blind

Passes, all on the explicit 4D-mask kernel except where replay.py also uses the plain one.
  actual     as served
  reference  peer messages masked, positions kept. With nothing masked, the plain kernel, marked plain_kernel, as replay.py
  placebo    instead of the note, a random span of designer tokens (system prompt or task text) of the note span's token length
             is masked, one draw per turn with a fixed seed. In none, whose turn has no note, the length is the leak branch's
             note span for the same seed. It measures what masking anything at all costs.

Per run directory, <out>/<seed>/<variant>/<agent>_e<episode>_solve.npz holds actual, reference and placebo at the named
positions of detector.render with their top-10 readouts (replay.py's format plus `placebo`), and index.json the turn. The
trace, <agent>_e<episode>_solve_trace.npz, covers every reply token (prompt_last onward):
  delta_reference, delta_placebo   ||actual - ref|| / ||ref|| per token and lens layer
  rank_actual, rank_reference      lens rank of the planted answer's first token per token at the late layers; the token is
                                   the one the leak branch's reasoning writes at its first mention, the same in every branch
  d_reference, d_placebo           actual - ref at the band layers, float16, from first_generated to 3 tokens past the first
                                   mention, at most WINDOW tokens
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]
from detector.render import find_turn, load_turns, render_turn, token_span  # noqa: E402
from replay import LENS_FILE, LENS_LAYERS, MODEL, MODEL_DIR, Capture, attention_mask_4d, band, load_model, on_model  # noqa: E402

VARIANTS = ("leak", "none", "benign_k1", "benign_k3", "proposal_k1", "proposal_k3")
WINDOW = 1024
SEED = 20261107
FIRST_KEEP = 8                       # the placebo never masks the first tokens of the sequence
BAND_LAYERS = sorted(set(band(26, 30)) | set(band(50, 62)))
LATE = sorted({on_model(l) for l in (52, 54, 56, 58, 60, 61, 62)})


class GpuCapture(Capture):
    """Capture whose pass returns bf16 [positions, layers, d_model] on the GPU, so every reply token fits."""

    def _hook(self, layer):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            self.saved[layer] = h[0, self.index]
        return hook

    def run(self, model, ids, positions, mask=None, position_ids=None):
        import torch
        device = model.device
        self.index = torch.as_tensor(positions, device=device)
        position_ids = torch.arange(len(ids), device=device)[None]
        with torch.no_grad():
            model.model(input_ids=torch.tensor([ids], device=device), position_ids=position_ids,
                        attention_mask=None if mask is None else mask.to(device), use_cache=False)
        return torch.stack([self.saved[layer] for layer in self.layers], dim=1)


def designer_spans(tokenizer, turn, full, offsets):
    """Token spans of the content of every designer-provenance system or user message, wrapper tokens excluded."""
    messages, meta, tools = turn["messages"], turn["message_meta"], turn["tools"]
    spans = []
    for i, tag in enumerate(meta):
        if tag["provenance"] != "designer" or messages[i]["role"] not in ("system", "user"):
            continue
        upto = tokenizer.apply_chat_template(messages[:i + 1], tools=tools, tokenize=False, add_generation_prompt=False)
        if not full.startswith(upto) or not upto.endswith("<|im_end|>\n"):
            continue
        block = upto.rfind("<|im_start|>")
        start, end = upto.index("\n", block) + 1, len(upto) - len("<|im_end|>\n")
        if end > start:
            spans.append((i, *token_span(offsets, start, end)))
    return spans


def placebo_span(spans, length, rng):
    starts = [(i, s) for i, a, b in spans for s in range(max(a, FIRST_KEEP), b - length + 1)]
    if not starts:
        raise ValueError(f"no designer span of {length} tokens")
    i, s = starts[rng.integers(len(starts))]
    return i, s, s + length


def relative(a, b):
    return ((a.float() - b.float()).norm(dim=-1)/b.float().norm(dim=-1)).cpu().numpy()


def ranks(lens, H, layer, target, chunk=1024):
    """Rank of `target` in the lens readout of every row of H [rows, d] bf16 on the GPU."""
    import torch
    out = []
    for start in range(0, len(H), chunk):
        logits = lens.logits(H[start:start + chunk].float(), layer)
        out.append((logits > logits[:, target:target + 1]).sum(1))
    return torch.cat(out).cpu().numpy().astype(np.int32)


def replay_blind(model, tok, capture, lens, turn, out_dir, target, note_length, rng_key):
    import torch
    r = render_turn(tok, turn)
    ids, names = r["ids"], list(r["positions"])
    named = [r["positions"][n] for n in names]
    encoded = tok(r["text"], add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != ids:
        raise ValueError("re-encoding differs")
    offsets = [tuple(o) for o in encoded["offset_mapping"]]
    reply = list(range(r["positions"]["prompt_last"], len(ids)))
    n, k = len(ids), len(named)
    started = time.perf_counter()
    masked = bool(r["peer_token_spans"])
    length = (len(r["mask"]) - sum(r["mask"])) if masked else note_length
    rng = np.random.default_rng([SEED, zlib.crc32(rng_key.encode())])
    message, a, b = placebo_span(designer_spans(tok, turn, r["text"], offsets), length, rng)
    placebo_keep = [1]*n
    placebo_keep[a:b] = [0]*(b - a)
    index = named + reply
    passes = {"actual": capture.run(model, ids, index, attention_mask_4d(n)),
              "reference": capture.run(model, ids, index, attention_mask_4d(n, r["mask"]) if masked else None),
              "placebo": capture.run(model, ids, index, attention_mask_4d(n, placebo_keep))}
    arrays = {}
    for label, H in passes.items():
        top_ids = np.zeros((k, len(capture.layers), 10), dtype=np.int32)
        top_logits = np.zeros((k, len(capture.layers), 10), dtype=np.float32)
        for li, layer in enumerate(capture.layers):
            top = lens.top(H[:k, li].float(), layer)
            top_ids[:, li], top_logits[:, li] = top.indices.cpu().numpy(), top.values.cpu().numpy()
        arrays[label] = H[:k].float().cpu().numpy()
        arrays[f"{label}_top_ids"], arrays[f"{label}_top_logits"] = top_ids, top_logits
    path = out_dir/f"{turn['agent']}_e{turn['episode_index']}_solve.npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)

    A, R, P = (passes[x][k:] for x in ("actual", "reference", "placebo"))
    first = r["positions"]["first_generated"] - reply[0]
    stop = r["positions"]["mention_0"] + 3 - reply[0] if "mention_0" in r["positions"] else first + 256
    window = list(range(first, min(stop, first + WINDOW, len(reply))))
    bands = [capture.layers.index(l) for l in BAND_LAYERS]
    trace = {"positions": np.array(reply, dtype=np.int32), "layers": np.array(capture.layers), "late": np.array(LATE),
             "band_layers": np.array(BAND_LAYERS), "window": np.array([reply[i] for i in window], dtype=np.int32),
             "delta_reference": np.stack([relative(A[:, li], R[:, li]) for li in range(len(capture.layers))], axis=1),
             "delta_placebo": np.stack([relative(A[:, li], P[:, li]) for li in range(len(capture.layers))], axis=1),
             "rank_actual": np.stack([ranks(lens, A[:, capture.layers.index(l)], l, target) for l in LATE], axis=1),
             "rank_reference": np.stack([ranks(lens, R[:, capture.layers.index(l)], l, target) for l in LATE], axis=1),
             "d_reference": (A[window][:, bands].float() - R[window][:, bands].float()).half().cpu().numpy(),
             "d_placebo": (A[window][:, bands].float() - P[window][:, bands].float()).half().cpu().numpy()}
    trace_path = out_dir/f"{turn['agent']}_e{turn['episode_index']}_solve_trace.npz"
    np.savez(trace_path, **trace)
    del passes, A, R, P
    torch.cuda.empty_cache()
    return {"agent": turn["agent"], "episode_index": turn["episode_index"], "phase": "solve", "file": path.name,
            "trace_file": trace_path.name, "trace_target": int(target), "trace_target_text": tok.decode([target]),
            "tokens": n, "prompt_tokens": r["prompt_tokens"], "reply_tokens": len(reply),
            "masked_tokens": len(r["mask"]) - sum(r["mask"]), "reference_kind": "masked" if masked else "plain_kernel",
            "peer_token_spans": r["peer_token_spans"], "placebo_span": [a, b], "placebo_message": message,
            "placebo_text": tok.decode(ids[a:b])[:200], "positions": dict(zip(names, named)),
            "position_tokens": {x: tok.decode([ids[p]]) for x, p in zip(names, named)},
            "position_token_ids": {x: ids[p] for x, p in zip(names, named)},
            "seconds": round(time.perf_counter() - started, 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seed_dirs", nargs="+", type=Path)
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--out-root", type=Path, default=ROOT/"data/replay_14b/blind")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from detector.jspace import Lens
    model, tok = load_model()
    lens = Lens(MODEL_DIR, LENS_FILE)
    lens.J = {layer: J.to(lens.device) for layer, J in lens.J.items()}
    capture = GpuCapture(model, lens.layers)
    for seed_dir in args.seed_dirs:
        leak_run = json.loads((seed_dir/"leak/run.json").read_text())["results"][args.episode]["agents"]
        blind = next(a for a, x in leak_run.items() if x.get("blind"))
        leak_turn = find_turn(load_turns(seed_dir/"leak"), blind, args.episode, "solve")
        lr = render_turn(tok, leak_turn)
        answer = leak_run[blind]["submission"]["answer"]
        target = lr["ids"][lr["positions"]["mention_0"]] if "mention_0" in lr["positions"] else tok.encode(answer, add_special_tokens=False)[0]
        note_length = len(lr["mask"]) - sum(lr["mask"])
        for variant in args.variants:
            out = args.out_root/seed_dir.name/variant
            if args.resume and (out/"index.json").exists():
                continue
            turn = find_turn(load_turns(seed_dir/variant), blind, args.episode, "solve")
            row = replay_blind(model, tok, capture, lens, turn, out, target, note_length, f"{seed_dir.name}/{variant}/{blind}")
            (out/"index.json").write_text(json.dumps({"run": str(seed_dir/variant), "model": MODEL, "layers": capture.layers,
                                                       "turns": [row]}, indent=1))
            print(seed_dir.name, variant, blind, row["tokens"], row["reply_tokens"], row["masked_tokens"], row["seconds"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
