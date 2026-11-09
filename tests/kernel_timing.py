#!/usr/bin/env python3
"""Lane D: time one teacher-forced pass of Qwen3-14B over N tokens, no mask against the explicit 4D boolean mask, with the
SDPA backend left to choose and with each backend forced, to see which kernel the replay's masked passes run on."""
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]
import replay  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
model, tok = replay.load_model()
ids = torch.randint(1000, 100000, (1, n), device="cuda")
pos = torch.arange(n, device="cuda")[None]
mask = replay.attention_mask_4d(n).cuda()


def timed(label, **kw):
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        model.model(input_ids=ids, position_ids=pos, use_cache=False, **kw)
    torch.cuda.synchronize()
    print(f"{label:<40} {time.time() - t:6.2f}s  peak {torch.cuda.max_memory_allocated()/2**30:5.1f} GiB", flush=True)
    torch.cuda.reset_peak_memory_stats()


timed("warmup, no mask")
timed("no mask")
timed("4D bool mask, default", attention_mask=mask)
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
for backend in (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH):
    try:
        with sdpa_kernel(backend):
            timed(f"4D bool mask, {backend.name}", attention_mask=mask)
    except Exception as e:  # noqa: BLE001
        print(f"4D bool mask, {backend.name}: {type(e).__name__} {str(e)[:120]}", flush=True)
