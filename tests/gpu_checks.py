#!/usr/bin/env python3
"""GPU checks run once on the pod before any replay result is read. Prints one PASS or FAIL line per check and writes
results/gpu_checks.json.

  mask causal      positions before the first masked token are unchanged by the reference pass
  mask deletion    masking a span equals physically removing its tokens with position_ids kept
  mask no-op       with nothing masked, an all-ones 2D mask agrees with no mask
  repeat           the same pass run twice
  kernel noise     the unmasked path against an explicit causal 4D mask, the realistic bf16 floor
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
LAYERS = [0, 15, 26, 30, 45, 62]
out = {}


def report(name, ok, **values):
    out[name] = {"ok": bool(ok), **values}
    print(("PASS" if ok else "FAIL"), name, json.dumps(values), flush=True)


def relative(a, b):
    return float((a - b).norm()/b.norm())


class Grab:
    def __init__(self, model, layers):
        self.layers, self.h = layers, {}
        self.handles = [model.model.layers[l].register_forward_hook(self._hook(l)) for l in layers]

    def _hook(self, l):
        def hook(_m, _i, o):
            self.h[l] = (o[0] if isinstance(o, tuple) else o)[0].float()
        return hook

    def run(self, model, ids, position_ids=None, mask=None):
        device = model.device
        n = len(ids)
        pos = torch.arange(n, device=device)[None] if position_ids is None else torch.tensor([position_ids], device=device)
        with torch.no_grad():
            model.model(input_ids=torch.tensor([ids], device=device), position_ids=pos,
                        attention_mask=None if mask is None else mask.to(device), use_cache=False)
        return {l: v.clone() for l, v in self.h.items()}


def main():
    model, tok = load_model()
    grab = Grab(model, LAYERS)

    leak = render_turn(tok, find_turn(load_turns(RUN/"leak"), "carol", 2, "solve"))
    ids, keep = leak["ids"], leak["mask"]
    a, b = leak["peer_token_spans"][0]
    print("leak turn tokens", len(ids), "masked span", a, b, flush=True)
    ones = torch.ones(1, len(ids), dtype=torch.long)
    masked2d = torch.tensor([keep])

    actual = grab.run(model, ids, mask=ones)
    reference = grab.run(model, ids, mask=masked2d)
    worst = max(float((actual[l][:a] - reference[l][:a]).abs().max()) for l in LAYERS)
    report("mask causal", worst == 0.0, max_abs_before_span=worst)
    after = {l: relative(reference[l][b:], actual[l][b:]) for l in LAYERS}
    report("mask changes later positions", min(after.values()) > 1e-3, relative_delta_by_layer=after)

    kept = [i for i, m in enumerate(keep) if m]
    removed = grab.run(model, [ids[i] for i in kept], position_ids=kept, mask=torch.ones(1, len(kept), dtype=torch.long))
    index = torch.tensor([i for i in kept if i >= b], device=model.device)
    tail = torch.arange(len(kept) - len(index), len(kept), device=model.device)
    deletion = {l: relative(removed[l][tail], reference[l][index]) for l in LAYERS}
    floor = {l: relative(reference[l][index], actual[l][index]) for l in LAYERS}
    report("mask deletion", max(deletion.values()) < 0.2*min(floor.values()), relative_delta_by_layer=deletion, mask_effect_by_layer=floor)

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
    kernel = {l: relative(explicit[l], unmasked[l]) for l in LAYERS}
    report("kernel noise", max(kernel.values()) < 5e-2, relative_delta_by_layer=kernel)
    for handle in grab.handles:
        handle.remove()
    del model, grab, actual, reference, removed, unmasked, all_ones, again, explicit
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
