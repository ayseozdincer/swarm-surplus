#!/usr/bin/env python3
"""W2 done check. Load Qwen3-32B in bf16 and the released lens, print the top-10 lens readout at the last token of a
short prompt for 5 layers, once through the J-space repo's Readout and once through detector.jspace.Lens."""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens")]
from detector.jspace import Lens  # noqa: E402

ART = Path("/workspace/artifacts")
MODEL = ART/"qwen3_32b"
LENS = ART/"lens_candidate/qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt"
LAYERS = (10, 26, 30, 45, 60)
PROMPT = "The capital of France is"

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
t = time.time()
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()
print(f"model loaded in {time.time()-t:.0f}s, {torch.cuda.memory_allocated()/2**30:.1f} GiB, layers {model.config.num_hidden_layers}, attn {model.config._attn_implementation}", flush=True)
t = time.time()
lens = Lens(MODEL, LENS)
print(f"lens loaded in {time.time()-t:.0f}s, {len(lens.layers)} layers {lens.layers[0]}..{lens.layers[-1]}, J {tuple(lens.J[0].shape)} {lens.J[0].dtype}, vocab {lens.vocab}", flush=True)

ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
saved = {}
handles = [model.model.layers[l].register_forward_hook(lambda m, i, o, l=l: saved.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0, -1].float())) for l in LAYERS]
with torch.no_grad():
    out = model.model(ids, use_cache=False)
    final = model.lm_head(out.last_hidden_state[:, -1])[0].float()
for h in handles:
    h.remove()
print("model next-token top-5:", [tok.decode([i]) for i in final.topk(5).indices.tolist()])
for l in LAYERS:
    top = lens.top(saved[l][None], l)
    print(f"L{l:2d} top-10:", [tok.decode([i]) for i in top.indices[0].tolist()], flush=True)

# The hand-written norm against the model's own module, on the same input.
H = saved[LAYERS[2]][None]
x = H @ lens.J[LAYERS[2]].cuda().float().T
theirs = model.model.norm(x).float() @ lens.head.T
mine = lens.logits(H, LAYERS[2])
print("max |lens_logits - model.norm path|:", float((theirs - mine).abs().max()), "scale", float(mine.abs().max()))
