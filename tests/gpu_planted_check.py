#!/usr/bin/env python3
"""Phase 2c check on the real model, first thing on the replay card, a few passes on one leak turn. Run with SWARM_MODEL set.

  python tests/gpu_planted_check.py runs/planted/seed8001/leak results/lane_e/planted_check.json

Per lens layer, at the 512 tokens after the line
  mask_effect     relative difference, actual against the derived reference
  kernel_noise    relative difference, actual (explicit 4D mask) against declared (plain kernel), what the declared mask leaves
  swap_leak       relative difference, derived reference against the same pass with the line's token ids replaced. The masked
                  tokens must be invisible to everything else, so this is zero at bf16 precision
  shared_prefix   relative difference, the derived reference from a cached prefix against the full pass. Not always zero. On 14B
                  turns of 8 to 9k tokens the two paths differ like two kernels (about 1 percent), and at 12 to 16k tokens they are
                  bit-identical, so the attention kernel switches with sequence size. Both passes a comparison uses run on one path
The check fails if the masked tokens are visible at all, or if the median path difference or kernel noise is not under half the mask effect.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts"), str(ROOT/"scripts/pod"), str(ROOT/"third_party/maj-lens")]
from detector.render import render_turn  # noqa: E402
import replay  # noqa: E402
import replay_planted  # noqa: E402


def main() -> int:
    run_dir = Path(sys.argv[1])
    model, tok = replay.load_model()
    layers = list(range(model.config.num_hidden_layers - 1))
    capture = replay.Capture(model, layers)
    _, turns = replay_planted.target_turns(run_dir)
    r = render_turn(tok, turns[0], mask_mode="derived")
    ids, mask = r["ids"], r["mask"]
    first, last = r["derived_token_spans"][0]
    n = len(ids)
    after = list(range(last, min(n, last + 512)))
    swapped = list(ids)
    swapped[first:last] = [(i + 7) % len(tok) for i in ids[first:last]]
    actual = capture.run(model, ids, after, replay.attention_mask_4d(n))
    derived = capture.run(model, ids, after, replay.attention_mask_4d(n, mask))
    plain = capture.run(model, ids, after, None)
    swap = capture.run(model, swapped, after, replay.attention_mask_4d(n, mask))
    cache = capture.prefix(model, ids[:first])
    shared = capture.run_suffix(model, cache, ids, first, after, mask)
    rel = lambda a, b: (torch.linalg.vector_norm(a - b, dim=-1)/torch.linalg.vector_norm(b, dim=-1)).numpy()  # noqa: E731
    out = {"tokens": n, "line_tokens": last - first, "line": tok.decode(ids[first:last]), "by_layer": []}
    effect, noise, leak, prefix = rel(actual, derived), rel(actual, plain), rel(swap, derived), rel(shared, derived)
    for li, layer in enumerate(layers):
        out["by_layer"].append({"layer": layer, "mask_effect": float(np.median(effect[:, li])), "kernel_noise": float(np.median(noise[:, li])),
                                "swap_leak_max": float(leak[:, li].max()), "shared_prefix_max": float(prefix[:, li].max()),
                                "shared_prefix_median": float(np.median(prefix[:, li]))})
    band = out["by_layer"][16:]
    out["worst_swap_leak"] = max(x["swap_leak_max"] for x in out["by_layer"])
    out["worst_shared_prefix"] = max(x["shared_prefix_max"] for x in out["by_layer"])
    out["median_shared_prefix_L16_38"] = float(np.median([x["shared_prefix_median"] for x in band]))
    out["median_mask_effect_L16_38"] = float(np.median([x["mask_effect"] for x in band]))
    out["median_kernel_noise_L16_38"] = float(np.median([x["kernel_noise"] for x in band]))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "by_layer"}, indent=1))
    effect = out["median_mask_effect_L16_38"]
    ok = out["worst_swap_leak"] < 0.01*effect and out["median_shared_prefix_L16_38"] < 0.5*effect and out["median_kernel_noise_L16_38"] < 0.5*effect
    print("planted check", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
