#!/usr/bin/env python3
"""Replay chosen turns of a run teacher-forced, once as served and once with `peer` messages masked out.

  python scripts/replay.py runs/pilot/seed1000/leak --episode 2 --out data/replay/pilot/seed1000/leak

For each agent and phase of the episode, the actual pass and the reference pass read the residual stream at every
lens layer at the positions of detector.render. The reference pass uses the same token ids and position_ids with every
peer message masked as keys, so nothing can attend to those tokens and no position shifts.

Both passes run through one attention kernel, an explicit 4D mask (tests/gpu_checks.py: the same result as a 2D mask with
zeros, exactly). The plain kernel differs from it by 0.1 to 5 percent by layer, the size of the mask effect, so comparing
across kernels would read kernel noise as surplus. A turn with no peer message has nothing to mask: its `reference` is the
plain kernel instead, marked reference_kind plain_kernel, which measures that noise on the same turn.

Writes <out>/<agent>_e<episode>_<phase>.npz, arrays actual and reference [positions, layers, d_model] float32 and the
top-10 readout ids and logits for each, and <out>/index.json with the positions, the masked spans and the token text.
Run with --decompose to add the J-space shares (detector.jspace.pursuit) to index.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens")]
from detector.render import load_turns, render_turn  # noqa: E402

ARTIFACTS = Path("/workspace/artifacts")
MODEL_DIR = ARTIFACTS/"qwen3_32b"
LENS_FILE = ARTIFACTS/"lens_candidate/qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt"
PHASES = ("solve", "verdict", "reflection")
LENS_LAYERS = tuple(range(63))         # the released lens covers blocks 0 to 62 of 64


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda",
                                                 local_files_only=True).eval()
    return model, tokenizer


def attention_mask_4d(n: int, keep=None):
    """Causal [1, 1, n, n] boolean mask; `keep` zeros mark keys nothing may attend to."""
    import torch
    mask = torch.ones(n, n, dtype=torch.bool).tril()
    if keep is not None:
        mask = mask & torch.tensor(keep, dtype=torch.bool)[None, :]
    return mask[None, None]


class Capture:
    """Forward hooks on every lens layer; after a pass, `take` stacks the residual at the given positions."""

    def __init__(self, model, layers):
        self.layers, self.saved, self.handles = list(layers), {}, []
        for layer in self.layers:
            self.handles.append(model.model.layers[layer].register_forward_hook(self._hook(layer)))
        self.index = None

    def _hook(self, layer):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            self.saved[layer] = h[0, self.index].float().cpu()
        return hook

    def run(self, model, ids, positions, mask=None):
        import torch
        device = model.device
        self.index = torch.as_tensor(positions, device=device)
        input_ids = torch.tensor([ids], device=device)
        position_ids = torch.arange(len(ids), device=device)[None]
        attention_mask = None if mask is None else mask.to(device)
        with torch.no_grad():
            model.model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, use_cache=False)
        return torch.stack([self.saved[layer] for layer in self.layers], dim=1)


def replay_turn(model, tokenizer, capture, turn, lens, out_path):
    r = render_turn(tokenizer, turn)
    names = list(r["positions"])
    positions = [r["positions"][n] for n in names]
    started = time.perf_counter()
    masked = bool(r["peer_token_spans"])
    actual = capture.run(model, r["ids"], positions, attention_mask_4d(len(r["ids"])))
    reference = capture.run(model, r["ids"], positions, attention_mask_4d(len(r["ids"]), r["mask"]) if masked else None)
    arrays = {"actual": actual.numpy(), "reference": reference.numpy()}
    for label, H in (("actual", actual), ("reference", reference)):
        ids = np.zeros((len(names), len(capture.layers), 10), dtype=np.int32)
        logits = np.zeros((len(names), len(capture.layers), 10), dtype=np.float32)
        for li, layer in enumerate(capture.layers):
            top = lens.top(H[:, li], layer)
            ids[:, li], logits[:, li] = top.indices.cpu().numpy(), top.values.cpu().numpy()
        arrays[f"{label}_top_ids"], arrays[f"{label}_top_logits"] = ids, logits
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return {"agent": turn["agent"], "episode_index": turn["episode_index"], "phase": turn["phase"], "file": out_path.name,
            "tokens": len(r["ids"]), "prompt_tokens": r["prompt_tokens"], "masked_tokens": len(r["mask"]) - sum(r["mask"]),
            "reference_kind": "masked" if masked else "plain_kernel",
            "peer_token_spans": r["peer_token_spans"], "positions": dict(zip(names, positions)),
            "position_tokens": {n: tokenizer.decode([r["ids"][p]]) for n, p in zip(names, positions)},
            "position_token_ids": {n: r["ids"][p] for n, p in zip(names, positions)},
            "max_abs_delta_at_layer": float((actual - reference).abs().max()), "seconds": round(time.perf_counter() - started, 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--agents", nargs="*")
    parser.add_argument("--phases", nargs="*", default=list(PHASES))
    parser.add_argument("--out-root", type=Path, default=ROOT/"data/replay")
    args = parser.parse_args()
    from detector.jspace import Lens
    model, tokenizer = load_model()
    lens = Lens(MODEL_DIR, LENS_FILE)
    capture = Capture(model, lens.layers)
    for run_dir in args.run_dirs:
        out = args.out_root/Path(*run_dir.parts[-3:])
        rows = []
        for turn in load_turns(run_dir):
            if turn["episode_index"] != args.episode or turn["phase"] not in args.phases:
                continue
            if args.agents and turn["agent"] not in args.agents:
                continue
            path = out/f"{turn['agent']}_e{turn['episode_index']}_{turn['phase']}.npz"
            rows.append(replay_turn(model, tokenizer, capture, turn, lens, path))
            print(run_dir, turn["agent"], turn["phase"], rows[-1]["tokens"], rows[-1]["seconds"], flush=True)
        (out/"index.json").write_text(json.dumps({"run": str(run_dir), "layers": capture.layers, "turns": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
