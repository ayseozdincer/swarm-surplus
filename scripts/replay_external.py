#!/usr/bin/env python3
"""Replay the insider-trading turns teacher-forced, the actual pass against a reference with Sally's spans masked, whole-reply traces.

  SWARM_MODEL=qwen3_14b SWARM_ARTIFACTS=/root/artifacts python scripts/replay_external.py results/external/runs --out data/replay_external

Every turn under <runs>/<variant>/samples.jsonl (trade and report) gets two passes on the explicit 4D-mask kernel from a shared prefix
(scripts/replay.py): actual, and the reference with the peer spans masked as keys and every position id kept. A turn with nothing to
mask (the no-tip control) gets the plain kernel as its reference, which measures kernel noise and is the floor pool.

Saved per turn, slim, in <out>/<variant>/s<sample>_<phase>.npz
  trace_delta_reference   [reply tokens, lens layers]  ||h_actual - h_reference|| / ||h_actual||, the per-token relative surplus
  trace_actual_norm       [reply tokens, lens layers]
  logprob_actual          [reply tokens]  log-probability of the forced reply token under the actual pass (float32 head)
  logprob_reference       [reply tokens]  the same under the reference pass
  reference_argmax        [reply tokens]  the reference's most likely token at each position, and reference_argmax_logprob
  actual_argmax           [reply tokens]
  reply_ids               [reply tokens]  the forced token ids, so positions can be found without the model
and a row per turn in <out>/index.json with the spans, the labels carried by the sample and the timing. Turns already on disk are
skipped, so the script resumes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import replay  # noqa: E402
from detector.render import render_turn  # noqa: E402

BUSY = Path("/workspace/busy")
CHUNK = 256


class Heads:
    """Forward hook on the model's final norm: keeps the normed hidden states that predict the reply tokens."""

    def __init__(self, model):
        self.from_ = None
        self.saved = None
        self.handle = model.model.norm.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        if self.from_ is not None:
            self.saved = output[0, self.from_ - 1:-1].detach()


def logprobs(head_weight, hidden, targets):
    """Per-token log-probability of each target under a float32 head, the argmax id and its log-probability."""
    import torch
    lp_t, arg, lp_arg = [], [], []
    tgt = torch.as_tensor(targets, device=hidden.device)
    for i in range(0, hidden.shape[0], CHUNK):
        logits = hidden[i:i + CHUNK].float() @ head_weight.T
        lp = torch.log_softmax(logits, dim=-1)
        lp_t.append(lp.gather(1, tgt[i:i + CHUNK, None])[:, 0])
        best = lp.argmax(dim=-1)
        arg.append(best)
        lp_arg.append(lp.gather(1, best[:, None])[:, 0])
        del logits, lp
    return (torch.cat(lp_t).cpu().numpy().astype(np.float32), torch.cat(arg).cpu().numpy().astype(np.int32),
            torch.cat(lp_arg).cpu().numpy().astype(np.float32))


def replay_sample(model, tokenizer, capture, heads, head_weight, turn, out_path):
    import torch
    started = time.perf_counter()
    r = render_turn(tokenizer, turn)
    ids, mask, P, N = r["ids"], r["mask"], r["prompt_tokens"], len(r["ids"])
    masked = bool(r["peer_token_spans"])
    split = min(a for a, _ in r["peer_token_spans"]) if masked else 0
    positions = [P - 1, N - 1]
    targets = ids[P:]
    capture.trace_from = heads.from_ = P - split
    traces, lps = {}, {}
    if split:
        cache = capture.prefix(model, ids[:split])
        capture.run_suffix(model, cache, ids, split, positions)
        traces["actual"], lps["actual"] = capture.trace(), logprobs(head_weight, heads.saved, targets)
        capture.run_suffix(model, cache, ids, split, positions, mask)
        traces["reference"], lps["reference"] = capture.trace(), logprobs(head_weight, heads.saved, targets)
        del cache
    else:
        capture.run(model, ids, positions, replay.attention_mask_4d(N))
        traces["actual"], lps["actual"] = capture.trace(), logprobs(head_weight, heads.saved, targets)
        capture.run(model, ids, positions, None)                # plain kernel: the floor
        traces["reference"], lps["reference"] = capture.trace(), logprobs(head_weight, heads.saved, targets)
    capture.trace_from = heads.from_ = None
    capture.traced, heads.saved = {}, None
    A, R = traces["actual"], traces["reference"]
    layers = range(A.shape[1])
    nA = torch.stack([torch.linalg.vector_norm(A[:, li].float(), dim=-1) for li in layers], dim=1)
    nD = torch.stack([torch.linalg.vector_norm(A[:, li].float() - R[:, li].float(), dim=-1) for li in layers], dim=1)
    arrays = {"trace_delta_reference": (nD / nA).cpu().numpy().astype(np.float32), "trace_actual_norm": nA.cpu().numpy().astype(np.float32),
              "logprob_actual": lps["actual"][0], "logprob_reference": lps["reference"][0],
              "actual_argmax": lps["actual"][1], "reference_argmax": lps["reference"][1], "reference_argmax_logprob": lps["reference"][2],
              "reply_ids": np.asarray(targets, dtype=np.int32)}
    del A, R, traces
    torch.cuda.empty_cache()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)
    surplus = arrays["logprob_actual"] - arrays["logprob_reference"]
    return {"variant": turn["variant"], "sample": turn["episode_index"], "phase": turn["phase"], "file": out_path.name,
            "tokens": N, "prompt_tokens": P, "reply_tokens": N - P, "masked_tokens": len(mask) - sum(mask),
            "reference_kind": "masked" if masked else "plain_kernel", "shared_prefix_tokens": split,
            "peer_token_spans": r["peer_token_spans"], "positions": r["positions"], "prompt_chars": r["prompt_chars"],
            "traded_on_tip": turn.get("traded_on_tip"), "action": turn.get("action"), "amount": turn.get("amount"),
            "finish_reason": turn["response"].get("finish_reason"),
            "logprob_surplus_max": float(surplus.max()), "logprob_surplus_sum_pos": float(surplus[surplus > 0].sum()),
            "delta_band_mean": float(arrays["trace_delta_reference"][:, 16:39].mean()),
            "seconds": round(time.perf_counter() - started, 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "data/replay_external")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--phases", nargs="*", default=["trade", "report"])
    parser.add_argument("--limit", type=int, default=None, help="at most this many turns a variant (smoke test)")
    parser.add_argument("--stop-at", type=float, default=None, help="unix time after which no new turn is started")
    args = parser.parse_args()
    import torch
    model, tokenizer = replay.load_model()
    capture = replay.Capture(model, replay.LENS_LAYERS)
    heads = Heads(model)
    head_weight = model.lm_head.weight.detach().float()
    index_path = args.out / "index.json"
    rows = json.loads(index_path.read_text())["turns"] if index_path.exists() else []
    done = {(row["variant"], row["sample"], row["phase"]) for row in rows}
    started, n = time.time(), 0
    for samples in sorted(args.runs_root.glob("*/samples.jsonl")):
        variant = samples.parent.name
        if args.variants and variant not in args.variants:
            continue
        turns = [json.loads(line) for line in samples.read_text(encoding="utf-8").splitlines() if line.strip()]
        turns = sorted((t for t in turns if t["phase"] in args.phases), key=lambda t: (t["episode_index"], t["phase"] != "trade"))
        if args.limit:
            turns = turns[:args.limit]
        for turn in turns:
            key = (variant, turn["episode_index"], turn["phase"])
            path = args.out / variant / f"s{turn['episode_index']:03d}_{turn['phase']}.npz"
            if key in done and path.exists():
                continue
            if args.stop_at and time.time() > args.stop_at:
                print("== stop-at reached", flush=True)
                break
            row = replay_sample(model, tokenizer, capture, heads, head_weight, {**turn, "variant": variant}, path)
            rows.append(row)
            n += 1
            if BUSY.parent.exists():
                BUSY.touch()
            print(f"{variant} s{turn['episode_index']} {turn['phase']} tokens {row['tokens']} reply {row['reply_tokens']} masked {row['masked_tokens']} "
                  f"{row['reference_kind']} lp_max {row['logprob_surplus_max']:.2f} band {row['delta_band_mean']:.4f} {row['seconds']}s", flush=True)
            if n % 10 == 0:
                args.out.mkdir(parents=True, exist_ok=True)
                index_path.write_text(json.dumps({"model": replay.MODEL_NAME, "layers": list(capture.layers), "turns": rows}, indent=1))
    args.out.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({"model": replay.MODEL_NAME, "layers": list(capture.layers), "turns": rows}, indent=1))
    print(f"== done {n} new turns, {len(rows)} total, {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
