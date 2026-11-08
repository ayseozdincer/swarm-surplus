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

The model comes from SWARM_MODEL (qwen3_32b, the default, or qwen3_14b) under SWARM_ARTIFACTS (default /workspace/artifacts),
so every script that imports MODEL_DIR, LENS_FILE and LENS_LAYERS from here follows the same switch.

--shared-prefix runs a masked turn's tokens before its first peer span once, on the explicit 4D-mask kernel, and every pass
continues from a copy of that cache, also on the 4D-mask kernel. Those tokens are identical in every pass, so this changes
no pass's inputs, only skips recomputing them. --compacted adds a third pass, the reference with position ids closed up
after each masked span (detector.render.compacted_positions), saved as `compacted`; an unmasked turn copies its reference.
--blind-only keeps the agents blind at the replayed episode. A blind agent's turn also reads where its reasoning first
writes its task's withheld answer (positions planted_pre and planted_0), whether or not it submits it. Turns whose .npz exists are skipped and index.json is merged,
so separate calls can add phases or agents to one run directory.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens")]
from detector.render import compacted_positions, load_turns, render_turn  # noqa: E402

ARTIFACTS = Path(os.environ.get("SWARM_ARTIFACTS", "/workspace/artifacts"))
# model directory, lens file, lens layers
MODELS = {
    # the released lens covers blocks 0 to 62 of 64; a partial checkpoint (running sums)
    "qwen3_32b": ("qwen3_32b", "lens_candidate/qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt", tuple(range(63))),
    # blocks 0 to 38 of 40; finished format with a `J` key, converged at 615 prompts
    "qwen3_14b": ("qwen3_14b", "lens_candidate/qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt", tuple(range(39))),
}
MODEL_NAME = os.environ.get("SWARM_MODEL", "qwen3_32b")
MODEL_DIR = ARTIFACTS/MODELS[MODEL_NAME][0]
LENS_FILE = ARTIFACTS/MODELS[MODEL_NAME][1]
LENS_LAYERS = MODELS[MODEL_NAME][2]
PHASES = ("solve", "verdict", "reflection")


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
        self.trace_from, self.traced = None, {}

    def _hook(self, layer):
        def hook(_module, _inputs, output):
            if self.index is None:
                return
            h = output[0] if isinstance(output, tuple) else output
            self.saved[layer] = h[0, self.index].float().cpu()
            if self.trace_from is not None:
                self.traced[layer] = h[0, self.trace_from:].detach().clone()
        return hook

    def trace(self):
        """[tokens, layers, d_model] residuals of the traced range from the last pass, on the GPU, in the model dtype."""
        import torch
        return torch.stack([self.traced[layer] for layer in self.layers], dim=1)

    def run(self, model, ids, positions, mask=None, position_ids=None):
        import torch
        device = model.device
        self.index = torch.as_tensor(positions, device=device)
        input_ids = torch.tensor([ids], device=device)
        position_ids = torch.arange(len(ids), device=device)[None] if position_ids is None else torch.tensor([position_ids], device=device)
        attention_mask = None if mask is None else mask.to(device)
        with torch.no_grad():
            model.model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, use_cache=False)
        return torch.stack([self.saved[layer] for layer in self.layers], dim=1)

    def prefix(self, model, ids):
        """KV cache of ids on the explicit 4D-mask kernel, nothing read."""
        import torch
        device = model.device
        self.index = None
        with torch.no_grad():
            out = model.model(input_ids=torch.tensor([ids], device=device), position_ids=torch.arange(len(ids), device=device)[None],
                              attention_mask=attention_mask_4d(len(ids)).to(device), use_cache=True)
        return out.past_key_values

    def run_suffix(self, model, cache, ids, split, positions, keep=None, position_ids=None):
        """Tokens ids[split:] after a copy of `cache` (ids[:split]), explicit 4D mask [1, 1, n - split, n] with the causal
        rows of the full sequence and `keep` zeros as masked keys. Positions are indices into the full sequence."""
        import torch
        device = model.device
        n = len(ids)
        self.index = torch.as_tensor([p - split for p in positions], device=device)
        mask = torch.ones(n - split, n, dtype=torch.bool).tril(diagonal=split)
        if keep is not None:
            mask = mask & torch.tensor(keep, dtype=torch.bool)[None, :]
        position_ids = list(range(n)) if position_ids is None else position_ids
        with torch.no_grad():
            model.model(input_ids=torch.tensor([ids[split:]], device=device), position_ids=torch.tensor([position_ids[split:]], device=device),
                        attention_mask=mask[None, None].to(device), past_key_values=copy.deepcopy(cache), use_cache=True)
        return torch.stack([self.saved[layer] for layer in self.layers], dim=1)


SHUFFLE_DEFAULT_LEN = 64     # the shuffled span's length on a turn with no peer span to match
TRACE_LAYERS = (18, 34)      # layers whose per-token surplus direction the trace keeps; deltas are kept at every layer


def shuffled_span(tokenizer, turn: dict, r: dict, length: int, seed: int) -> tuple[int, int] | None:
    """A random run of `length` sanctioned tokens inside the turn's last designer or environment message (its task or
    verdict prompt), drawn with a fixed seed; None when that message is shorter than `length`."""
    import random
    messages, meta = turn["messages"], turn["message_meta"]
    last = max((i for i, m in enumerate(meta) if m["provenance"] in ("designer", "environment") and messages[i]["role"] == "user"), default=None)
    if last is None:
        return None
    upto = tokenizer.apply_chat_template(messages[:last + 1], tools=turn["tools"], tokenize=False, add_generation_prompt=False)
    start_char = upto.rfind("<|im_start|>user\n") + len("<|im_start|>user\n")
    offsets = tokenizer(r["text"], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    first = next(i for i, (a, b) in enumerate(offsets) if b > start_char)
    last_token = next((i for i, (a, b) in enumerate(offsets) if a >= len(upto) - len("<|im_end|>\n")), len(offsets))
    if last_token - first < length:
        return None
    start = random.Random(seed).randrange(first, last_token - length + 1)
    return start, start + length


def replay_turn(model, tokenizer, capture, turn, lens, out_path, shared_prefix=False, compacted=False, mentions=None,
                shuffled=False, trace=False):
    """shuffled adds a reference that masks a random sanctioned span as long as the turn's peer spans (shuffled_span).
    trace keeps, for every token of the reply, the relative surplus per layer against each reference and the surplus
    direction at TRACE_LAYERS, from the same passes."""
    import torch
    r = render_turn(tokenizer, turn, mentions)
    names = list(r["positions"])
    positions = [r["positions"][n] for n in names]
    started = time.perf_counter()
    masked = bool(r["peer_token_spans"])
    shuffle = None
    if shuffled:
        length = (len(r["mask"]) - sum(r["mask"])) or SHUFFLE_DEFAULT_LEN
        seed = int.from_bytes(f"{turn['agent']}|{turn['episode_index']}|{turn['phase']}|{out_path.parent}".encode()[-64:], "little") % 2**32
        shuffle = shuffled_span(tokenizer, turn, r, length, seed)
    keep_shuffled = None
    if shuffle:
        keep_shuffled = [1]*len(r["ids"])
        keep_shuffled[shuffle[0]:shuffle[1]] = [0]*(shuffle[1] - shuffle[0])
    starts = ([min(a for a, _ in r["peer_token_spans"])] if masked else []) + ([shuffle[0]] if shuffle else [])
    split = min(starts) if starts and shared_prefix else 0
    capture.trace_from = r["prompt_tokens"] - split if trace else None
    passes, traces = {}, {}
    if split:
        cache = capture.prefix(model, r["ids"][:split])
        passes["actual"] = capture.run_suffix(model, cache, r["ids"], split, positions)
        traces["actual"] = capture.trace() if trace else None
        if masked:
            passes["reference"] = capture.run_suffix(model, cache, r["ids"], split, positions, r["mask"])
            traces["reference"] = capture.trace() if trace else None
        if shuffle:
            passes["shuffled"] = capture.run_suffix(model, cache, r["ids"], split, positions, keep_shuffled)
            traces["shuffled"] = capture.trace() if trace else None
        if compacted and masked:
            passes["compacted"] = capture.run_suffix(model, cache, r["ids"], split, positions, r["mask"], compacted_positions(r["mask"]))
        del cache
    capture.trace_from = r["prompt_tokens"] if trace else None
    if "actual" not in passes:
        passes["actual"] = capture.run(model, r["ids"], positions, attention_mask_4d(len(r["ids"])))
        traces["actual"] = capture.trace() if trace else None
    if "reference" not in passes:
        passes["reference"] = capture.run(model, r["ids"], positions, attention_mask_4d(len(r["ids"]), r["mask"]) if masked else None)
        traces["reference"] = capture.trace() if trace else None
    if compacted and "compacted" not in passes:
        passes["compacted"] = (capture.run(model, r["ids"], positions, attention_mask_4d(len(r["ids"]), r["mask"]),
                                           compacted_positions(r["mask"])) if masked else passes["reference"])
    capture.trace_from, capture.traced = None, {}
    actual, reference = passes["actual"], passes["reference"]
    arrays = {label: H.numpy() for label, H in passes.items()}
    if trace:
        A = traces["actual"].float()
        arrays["trace_actual_norm"] = torch.linalg.vector_norm(A, dim=-1).cpu().numpy().astype(np.float32)
        trace_layers = [capture.layers.index(l) for l in TRACE_LAYERS if l in capture.layers]
        for label in ("reference", "shuffled"):
            if traces.get(label) is None:
                continue
            D = A - traces[label].float()
            arrays[f"trace_delta_{label}"] = (torch.linalg.vector_norm(D, dim=-1)/torch.linalg.vector_norm(A, dim=-1)).cpu().numpy().astype(np.float32)
            arrays[f"trace_d_{label}"] = D[:, trace_layers].cpu().numpy().astype(np.float16)
        del A, D, traces
    for label, H in passes.items():
        ids = np.zeros((len(names), len(capture.layers), 10), dtype=np.int32)
        logits = np.zeros((len(names), len(capture.layers), 10), dtype=np.float32)
        for li, layer in enumerate(capture.layers):
            top = lens.top(H[:, li], layer)
            ids[:, li], logits[:, li] = top.indices.cpu().numpy(), top.values.cpu().numpy()
        arrays[f"{label}_top_ids"], arrays[f"{label}_top_logits"] = ids, logits
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)        # uncompressed: zlib gains little on float32 residuals and blocks the GPU loop
    return {"agent": turn["agent"], "episode_index": turn["episode_index"], "phase": turn["phase"], "file": out_path.name,
            "tokens": len(r["ids"]), "prompt_tokens": r["prompt_tokens"], "masked_tokens": len(r["mask"]) - sum(r["mask"]),
            "reference_kind": "masked" if masked else "plain_kernel", "shared_prefix_tokens": split,
            "passes": list(passes), "shuffled_span": shuffle, "trace_from": r["prompt_tokens"] if trace else None,
            "trace_layers": list(TRACE_LAYERS) if trace else None,
            "peer_token_spans": r["peer_token_spans"], "positions": dict(zip(names, positions)),
            "position_tokens": {n: tokenizer.decode([r["ids"][p]]) for n, p in zip(names, positions)},
            "position_token_ids": {n: r["ids"][p] for n, p in zip(names, positions)},
            "max_abs_delta_at_layer": float((actual - reference).abs().max()), "seconds": round(time.perf_counter() - started, 2)}


def blind_agents(run_dir: Path, episode: int) -> set[str]:
    config = json.loads((Path(run_dir)/"run.json").read_text())["run_config"]
    return {agent for ep, agent in config["blind_plan"] if ep == episode}


def planted_answers(run_dir: Path, episode: int) -> dict[str, str]:
    """Blind agent -> the withheld answer of its task at `episode`, from the experiment's task manifests. In a leak branch
    this is the planted value; in the others it is what the agent could only write by computing it."""
    run = json.loads((Path(run_dir)/"run.json").read_text())
    out = {}
    for agent, record in run["results"][episode]["agents"].items():
        if record["blind"]:
            manifest = Path(run_dir).parent.parent/"tasks"/f"manifest_{record['task_type']}.json"
            task = next(t for t in json.loads(manifest.read_text()) if t["task_id"] == record["task_id"])
            out[agent] = task["answer_key"][0]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--agents", nargs="*")
    parser.add_argument("--blind-only", action="store_true", help="only the agents blind at --episode")
    parser.add_argument("--phases", nargs="*", default=list(PHASES))
    parser.add_argument("--out-root", type=Path, default=ROOT/"data/replay")
    parser.add_argument("--resume", action="store_true", help="skip run directories that already have an index.json")
    parser.add_argument("--shared-prefix", action="store_true", help="compute the tokens before the first peer span once")
    parser.add_argument("--compacted", action="store_true", help="add the compacted reference pass")
    args = parser.parse_args()
    from detector.jspace import Lens
    model, tokenizer = load_model()
    lens = Lens(MODEL_DIR, LENS_FILE)
    capture = Capture(model, lens.layers)
    for run_dir in args.run_dirs:
        out = args.out_root/Path(*run_dir.parts[-3:])
        if args.resume and (out/"index.json").exists():
            continue
        old = json.loads((out/"index.json").read_text())["turns"] if (out/"index.json").exists() else []
        done = {row["file"] for row in old}
        blind = blind_agents(run_dir, args.episode) if args.blind_only else None
        planted = planted_answers(run_dir, args.episode)
        rows = []
        for turn in load_turns(run_dir):
            if turn["episode_index"] != args.episode or turn["phase"] not in args.phases:
                continue
            if (args.agents and turn["agent"] not in args.agents) or (blind is not None and turn["agent"] not in blind):
                continue
            path = out/f"{turn['agent']}_e{turn['episode_index']}_{turn['phase']}.npz"
            if path.name in done and path.exists():
                continue
            mentions = {"planted": planted[turn["agent"]]} if turn["agent"] in planted else None
            rows.append(replay_turn(model, tokenizer, capture, turn, lens, path, args.shared_prefix, args.compacted, mentions))
            print(run_dir, turn["agent"], turn["phase"], rows[-1]["tokens"], rows[-1]["shared_prefix_tokens"], rows[-1]["seconds"], flush=True)
        if rows or not old:
            out.mkdir(parents=True, exist_ok=True)
            (out/"index.json").write_text(json.dumps({"run": str(run_dir), "model": MODEL_NAME, "layers": capture.layers,
                                                      "turns": old + rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
