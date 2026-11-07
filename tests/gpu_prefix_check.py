#!/usr/bin/env python3
"""Lane D check of replay --shared-prefix: on one masked turn, each pass run from a cached prefix against the same pass run
over the whole sequence, both on the explicit 4D-mask kernel. Relative residual difference per layer at the read
positions, beside the mask effect (actual against reference). Run with SWARM_MODEL set; MODEL_DIR may be a small Qwen3 for a
logic check while the card is busy.

  python tests/gpu_prefix_check.py RUN_DIR AGENT EPISODE PHASE [MODEL_DIR]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts"), str(ROOT/"third_party/maj-lens")]
from detector.render import compacted_positions, find_turn, load_turns, render_turn  # noqa: E402
import replay  # noqa: E402


def main() -> int:
    run_dir, agent, episode, phase = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
    if len(sys.argv) > 5:
        replay.MODEL_DIR = Path(sys.argv[5])
    model, tok = replay.load_model()
    layers = list(range(model.config.num_hidden_layers - 1))
    capture = replay.Capture(model, layers)
    r = render_turn(tok, find_turn(load_turns(run_dir), agent, episode, phase))
    positions = list(r["positions"].values())
    split = min(a for a, _ in r["peer_token_spans"])
    n = len(r["ids"])
    full = {"actual": capture.run(model, r["ids"], positions, replay.attention_mask_4d(n)),
            "reference": capture.run(model, r["ids"], positions, replay.attention_mask_4d(n, r["mask"])),
            "compacted": capture.run(model, r["ids"], positions, replay.attention_mask_4d(n, r["mask"]), compacted_positions(r["mask"]))}
    cache = capture.prefix(model, r["ids"][:split])
    shared = {"actual": capture.run_suffix(model, cache, r["ids"], split, positions),
              "reference": capture.run_suffix(model, cache, r["ids"], split, positions, r["mask"]),
              "compacted": capture.run_suffix(model, cache, r["ids"], split, positions, r["mask"], compacted_positions(r["mask"]))}
    rel = lambda a, b: (torch.linalg.vector_norm(a - b, dim=-1)/torch.linalg.vector_norm(b, dim=-1)).numpy()  # noqa: E731
    out = {"tokens": n, "split": split, "masked": n - sum(r["mask"]), "positions": len(positions), "by_layer": []}
    effect = rel(full["actual"], full["reference"])
    for li, layer in enumerate(layers):
        row = {"layer": layer, "mask_effect_median": float(np.median(effect[:, li]))}
        for p in full:
            d = rel(shared[p], full[p])[:, li]
            row[f"{p}_shared_vs_full_median"], row[f"{p}_shared_vs_full_max"] = float(np.median(d)), float(d.max())
        out["by_layer"].append(row)
    worst = max(r_["actual_shared_vs_full_max"] for r_ in out["by_layer"])
    print(json.dumps({k: v for k, v in out.items() if k != "by_layer"}), "worst actual shared vs full", worst)
    for row in out["by_layer"][::max(1, len(layers)//8)]:
        print({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()})
    if len(sys.argv) > 6:
        Path(sys.argv[6]).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
