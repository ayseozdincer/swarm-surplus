#!/usr/bin/env python3
"""GPU checks run on the pod before any replay result is read. Prints one PASS or FAIL line per check and writes
results/gpu_checks.json. The first version (commit 5fd1a0c) failed three checks; results/gpu_checks_v1.log keeps its
output. It compared passes across attention kernels and used one global norm, so these criteria were fixed before this
version was run:

  mask causal      same kernel on both sides: positions before the first masked token must be exactly equal
  mask same path   the 2D mask with zeros against the equivalent explicit 4D mask must be exactly equal
  mask deletion    masking a span against physically removing its tokens with position_ids kept, same kernel, per token and
                   per layer: median delta under half the median effect of the mask itself
  mask no-op       with nothing masked, an all-ones 2D mask agrees with no mask
  repeat           the same pass run twice
  kernel noise     no pass or fail. The plain path against an explicit causal 4D mask, per token and at the read positions,
                   beside the mask effect at those positions. This is the floor a replay has to clear.
  pursuit          detector.jspace.pursuit against TransformerLens get_sparse_decomposition
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]
from detector import jspace  # noqa: E402
from detector.render import find_turn, load_turns, render_turn  # noqa: E402
from replay import LENS_FILE, MODEL_DIR, load_model  # noqa: E402

RUN = ROOT/"runs/pilot/seed1000"
LAYERS = [0, 15, 26, 30, 36, 40, 45, 50, 56, 62]
READS = ("prompt_last", "first_generated", "first_thought", "after_think")
out = {}


def report(name, ok, **values):
    out[name] = {"ok": bool(ok), **values}
    print(("PASS" if ok else "FAIL"), name, json.dumps(values), flush=True)


def relative(a, b):
    return float((a - b).norm()/b.norm())


def per_token(a, b):
    """Relative delta of each token's residual: rows of a against rows of b."""
    return (a - b).norm(dim=-1)/b.norm(dim=-1)


def spread(x):
    return {"median": float(x.median()), "p95": float(x.quantile(0.95)), "max": float(x.max())}


class Grab:
    def __init__(self, model, layers):
        self.layers, self.h = layers, {}
        self.handles = [model.model.layers[l].register_forward_hook(self._hook(l)) for l in layers]

    def _hook(self, l):
        def hook(_m, _i, o):
            self.h[l] = (o[0] if isinstance(o, tuple) else o)[0].float().cpu()
        return hook

    def run(self, model, ids, position_ids=None, mask=None):
        device = model.device
        n = len(ids)
        pos = torch.arange(n, device=device)[None] if position_ids is None else torch.tensor([position_ids], device=device)
        with torch.no_grad():
            model.model(input_ids=torch.tensor([ids], device=device), position_ids=pos,
                        attention_mask=None if mask is None else mask.to(device), use_cache=False)
        return dict(self.h)


def main():
    model, tok = load_model()
    grab = Grab(model, LAYERS)

    leak = render_turn(tok, find_turn(load_turns(RUN/"leak"), "carol", 2, "solve"))
    ids, keep = leak["ids"], leak["mask"]
    a, b = leak["peer_token_spans"][0]
    n = len(ids)
    print("leak turn tokens", n, "masked span", a, b, flush=True)
    causal = torch.ones(n, n, dtype=torch.bool).tril()[None, None]
    keys = torch.tensor(keep, dtype=torch.bool)
    masked4d = causal & keys[None, None, None, :]

    full = grab.run(model, ids, mask=causal)                                  # the note visible, explicit mask kernel
    ref4d = grab.run(model, ids, mask=masked4d)                               # the note masked, same kernel
    ref2d = grab.run(model, ids, mask=torch.tensor([keep]))                   # the note masked, as scripts/replay.py does
    worst = max(float((full[l][:a] - ref4d[l][:a]).abs().max()) for l in LAYERS)
    report("mask causal", worst == 0.0, max_abs_before_span=worst)
    same = max(float((ref2d[l] - ref4d[l]).abs().max()) for l in LAYERS)
    report("mask same path", same == 0.0, max_abs=same)

    kept = [i for i, m in enumerate(keep) if m]
    m = len(kept)
    removed = grab.run(model, [ids[i] for i in kept], position_ids=kept, mask=torch.ones(m, m, dtype=torch.bool).tril()[None, None])
    index = torch.tensor([i for i in kept if i >= b])
    tail = torch.arange(m - len(index), m)
    rows, ok = {}, True
    for l in LAYERS:
        deletion = per_token(removed[l][tail], ref4d[l][index])
        effect = per_token(ref4d[l][index], full[l][index])
        rows[l] = {"deletion": spread(deletion), "effect": spread(effect)}
        ok = ok and rows[l]["deletion"]["median"] < 0.5*rows[l]["effect"]["median"]
    report("mask deletion", ok, by_layer=rows)

    effect_at = {name: {l: float(per_token(ref4d[l][p:p + 1], full[l][p:p + 1])) for l in LAYERS}
                 for name, p in leak["positions"].items() if name in READS}

    plain = render_turn(tok, find_turn(load_turns(RUN/"none"), "carol", 2, "solve"))
    n = len(plain["ids"])
    unmasked = grab.run(model, plain["ids"])
    all_ones = grab.run(model, plain["ids"], mask=torch.ones(1, n, dtype=torch.long))
    noop = {l: relative(all_ones[l], unmasked[l]) for l in LAYERS}
    report("mask no-op", max(noop.values()) < 1e-2, relative_delta_by_layer=noop)
    again = grab.run(model, plain["ids"])
    repeat = {l: relative(again[l], unmasked[l]) for l in LAYERS}
    report("repeat", max(repeat.values()) < 1e-2, relative_delta_by_layer=repeat)
    causal4d = torch.ones(n, n, dtype=torch.bool).tril()[None, None]
    explicit = grab.run(model, plain["ids"], mask=causal4d)
    noise_all = {l: spread(per_token(explicit[l], unmasked[l])) for l in LAYERS}
    noise_at = {name: {l: float(per_token(explicit[l][p:p + 1], unmasked[l][p:p + 1])) for l in LAYERS}
                for name, p in plain["positions"].items() if name in READS}
    norms = {l: spread(unmasked[l].norm(dim=-1)) for l in LAYERS}
    report("kernel noise", True, per_token_by_layer=noise_all, at_read_positions=noise_at, mask_effect_at_read_positions=effect_at, residual_norm_by_layer=norms)
    for handle in grab.handles:
        handle.remove()
    del model, grab, full, ref4d, ref2d, removed, unmasked, all_ones, again, explicit
    torch.cuda.empty_cache()

    from lens import decompose
    from transformer_lens.tools.analysis import get_sparse_decomposition  # noqa: F401
    lens = jspace.Lens(MODEL_DIR, LENS_FILE)
    rng = np.random.default_rng(0)
    data = np.load(ROOT/"data/replay/pilot/seed1000/leak/carol_e2_solve.npz") if (ROOT/"data/replay/pilot/seed1000/leak/carol_e2_solve.npz").exists() else None
    for layer in (26, 45):
        atoms, norms = lens.atoms(layer)
        rows = torch.tensor(data["actual"][:, lens.layers.index(layer)], device="cuda") if data is not None else \
            torch.randn(8, atoms.shape[1], device="cuda")
        t = time.time()
        support, coefficients = jspace.pursuit(rows, atoms, norms)
        torch.cuda.synchronize()
        fast = time.time() - t
        t = time.time()
        s_ref, c_ref = decompose(rows, atoms, jspace.K)
        slow = time.time() - t
        z, z_ref = jspace.shares(support.cpu().numpy(), coefficients.cpu().numpy()), jspace.shares(s_ref, c_ref)
        overlap = float(np.mean([len(set(x) & set(y))/max(len(set(x) | set(y)), 1) for x, y in zip(z, z_ref)]))
        tv = float(np.mean([0.5*sum(abs(x.get(c, 0) - y.get(c, 0)) for c in set(x) | set(y)) for x, y in zip(z, z_ref)]))
        report(f"pursuit layer {layer}", overlap > 0.98 and tv < 0.02, rows=len(rows), jaccard=overlap, total_variation=tv,
               batched_s=round(fast, 2), reference_s=round(slow, 2))
        del atoms, norms
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"results/gpu_checks.json").write_text(json.dumps(out, indent=1) + "\n")
    print("failed:", [k for k, v in out.items() if not v["ok"]])


if __name__ == "__main__":
    main()
