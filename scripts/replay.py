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
--compacted adds a third pass, `compacted`, the reference with position ids closed up after the masked spans.
SWARM_MODEL=qwen3_14b replays Qwen3-14B with its own lens; the default is Qwen3-32B.
Run with --decompose to add the J-space shares (detector.jspace.pursuit) to index.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens")]
from detector.render import compacted_positions, load_turns, render_turn  # noqa: E402

ARTIFACTS = Path("/workspace/artifacts")
# The model replayed, chosen by SWARM_MODEL (default qwen3_32b). Every script that imports MODEL_DIR, LENS_FILE or
# LENS_LAYERS from here follows it. Each released lens covers every block but the last.
MODELS = {"qwen3_32b": (ARTIFACTS/"qwen3_32b", ARTIFACTS/"lens_candidate/qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt", 64),
          "qwen3_14b": (ARTIFACTS/"qwen3_14b", ARTIFACTS/"lens_candidate/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt", 40)}
MODEL = os.environ.get("SWARM_MODEL", "qwen3_32b")
MODEL_DIR, LENS_FILE, BLOCKS = MODELS[MODEL]
PHASES = ("solve", "verdict", "reflection")
LENS_LAYERS = tuple(range(BLOCKS - 1))


def on_model(layer: int) -> int:
    """A Qwen3-32B lens layer (0 to 62) at the same relative depth of this model's lens, so bands chosen on 32B carry over.
    The identity on 32B."""
    return round(layer*(len(LENS_LAYERS) - 1)/62)


def band(first: int, last: int) -> list[int]:
    """The 32B band first..last, both included, on this model."""
    return sorted({on_model(l) for l in range(first, last + 1)})


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
    """Forward hooks on every lens layer; after a pass, `take` stacks the residual at the given positions. The hooks keep
    their copies on the GPU and the pass moves them to the CPU once at the end: a copy to the CPU in every hook synchronises
    each layer, which made a 14B pass about six times slower, with bit-identical values."""

    def __init__(self, model, layers):
        self.layers, self.saved, self.handles = list(layers), {}, []
        for layer in self.layers:
            self.handles.append(model.model.layers[layer].register_forward_hook(self._hook(layer)))
        self.index = None

    def _hook(self, layer):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            self.saved[layer] = h[0, self.index].float()
        return hook

    def run(self, model, ids, positions, mask=None, position_ids=None):
        import torch
        device = model.device
        self.index = torch.as_tensor(positions, device=device)
        input_ids = torch.tensor([ids], device=device)
        position_ids = torch.arange(len(ids), device=device)[None] if position_ids is None else torch.tensor([position_ids], device=device)
        attention_mask = None if mask is None else mask.to(device)
        with torch.no_grad():
            model.model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, use_cache=False)
        return torch.stack([self.saved[layer] for layer in self.layers], dim=1).cpu()


def replay_turn(model, tokenizer, capture, turn, lens, out_path, compacted=False, compress=True):
    r = render_turn(tokenizer, turn)
    names = list(r["positions"])
    positions = [r["positions"][n] for n in names]
    started = time.perf_counter()
    masked = bool(r["peer_token_spans"])
    actual = capture.run(model, r["ids"], positions, attention_mask_4d(len(r["ids"])))
    reference = capture.run(model, r["ids"], positions, attention_mask_4d(len(r["ids"]), r["mask"]) if masked else None)
    passes = [("actual", actual), ("reference", reference)]
    if compacted:
        # Same mask and kernel as the reference, position ids closed up after the masked spans. With nothing masked it is
        # the reference itself.
        passes.append(("compacted", capture.run(model, r["ids"], positions, attention_mask_4d(len(r["ids"]), r["mask"]),
                                                position_ids=compacted_positions(r["mask"])) if masked else reference))
    arrays = {label: H.numpy() for label, H in passes}
    for label, H in passes:
        ids = np.zeros((len(names), len(capture.layers), 10), dtype=np.int32)
        logits = np.zeros((len(names), len(capture.layers), 10), dtype=np.float32)
        for li, layer in enumerate(capture.layers):
            top = lens.top(H[:, li], layer)
            ids[:, li], logits[:, li] = top.indices.cpu().numpy(), top.values.cpu().numpy()
        arrays[f"{label}_top_ids"], arrays[f"{label}_top_logits"] = ids, logits
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (np.savez_compressed if compress else np.savez)(out_path, **arrays)
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
    parser.add_argument("--resume", action="store_true", help="skip run directories that already have an index.json")
    parser.add_argument("--compacted", action="store_true", help="also run the compacted reference, saved as `compacted`")
    parser.add_argument("--free-only", action="store_true", help="only agents not instructed in the episode (run.json)")
    parser.add_argument("--uncompressed", action="store_true", help="plain .npz; compression costs about 9 s a turn on a network volume")
    parser.add_argument("--lens-on-gpu", action="store_true", help="keep every J on the GPU instead of copying one per readout")
    args = parser.parse_args()
    from detector.jspace import Lens
    model, tokenizer = load_model()
    lens = Lens(MODEL_DIR, LENS_FILE)
    if args.lens_on_gpu:
        lens.J = {layer: J.to(lens.device) for layer, J in lens.J.items()}
    capture = Capture(model, lens.layers)
    for run_dir in args.run_dirs:
        out = args.out_root/Path(*run_dir.parts[-3:])
        if args.resume and (out/"index.json").exists():
            continue
        rows = []
        instructed = set()
        if args.free_only:
            agents = json.loads((run_dir/"run.json").read_text())["results"][args.episode]["agents"]
            instructed = {a for a, r in agents.items() if r.get("instructed")}
        for turn in load_turns(run_dir):
            if turn["episode_index"] != args.episode or turn["phase"] not in args.phases:
                continue
            if (args.agents and turn["agent"] not in args.agents) or turn["agent"] in instructed:
                continue
            path = out/f"{turn['agent']}_e{turn['episode_index']}_{turn['phase']}.npz"
            rows.append(replay_turn(model, tokenizer, capture, turn, lens, path, args.compacted, not args.uncompressed))
            print(run_dir, turn["agent"], turn["phase"], rows[-1]["tokens"], rows[-1]["seconds"], flush=True)
        (out/"index.json").write_text(json.dumps({"run": str(run_dir), "model": MODEL, "layers": capture.layers, "turns": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
