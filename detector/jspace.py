"""The J-lens readout and the sparse J-space decomposition, batched for replay.

lens logits  W_U norm(J_l h)           third_party/maj-lens/lens.py Readout.logits, the top-10 of which a model "holds"
atoms        v_t = W_U[t] J_l           lens.py dictionary
pursuit      TransformerLens get_sparse_decomposition(..., algorithm="gradient_pursuit"), the same steps on a batch of
             rows, so one pass over the 151k atoms serves every row. tests/gpu_checks.py compares it with the original.
"""
from __future__ import annotations

import math

import numpy as np
import torch

K = 25
TOPK = 10
BACKTRACK = 20
CORRELATION_TOL = math.sqrt(torch.finfo(torch.float32).eps)
ACTIVE_TOL = math.sqrt(torch.finfo(torch.float64).eps)


def lens_logits(H: torch.Tensor, J: torch.Tensor, norm_weight: torch.Tensor, head: torch.Tensor, eps: float) -> torch.Tensor:
    """W_U norm(J h) for rows of H; Qwen3's final RMSNorm, in float32."""
    x = H.float() @ J.float().T
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm_weight.float()
    return x @ head.float().T


def _gradient_step(A, target, c, live):
    """One projected-gradient update on the rows in `live`. A [B,s,d] selected atoms, c [B,s] coefficients."""
    residual = target - torch.einsum("bs,bsd->bd", c, A)
    direction = torch.einsum("bsd,bd->bs", A, residual)
    projected = torch.einsum("bs,bsd->bd", direction, A)
    denominator = (projected*projected).sum(1)
    step = (projected*residual).sum(1)/denominator.clamp_min(torch.finfo(torch.float32).tiny)
    current = (residual*residual).sum(1)
    out = c.clamp_min(0.0)
    settled = denominator <= 0
    for _ in range(BACKTRACK + 1):
        candidate = (c + step[:, None]*direction).clamp_min(0.0)
        r = target - torch.einsum("bs,bsd->bd", candidate, A)
        accept = (~settled) & ((r*r).sum(1) <= current)
        out = torch.where(accept[:, None], candidate, out)
        settled = settled | accept
        if bool(settled.all()):
            break
        step = step*0.5
    return torch.where(live[:, None], out, c)


@torch.no_grad()
def pursuit(X: torch.Tensor, atoms: torch.Tensor, atom_norms: torch.Tensor, k: int = K):
    """support [B,k] token ids (-1 unused) and coefficients [B,k] for each row of X, on X's device."""
    B = len(X)
    target = X.float()
    tol = CORRELATION_TOL*target.norm(dim=1)
    selected = torch.full((B, k), -1, dtype=torch.long, device=X.device)
    c = torch.zeros(B, 0, device=X.device)
    residual = target.clone()
    live = torch.ones(B, dtype=torch.bool, device=X.device)
    rows = torch.arange(B, device=X.device)
    for step in range(k):
        correlation = (residual @ atoms.T)/atom_norms
        if step:
            taken = selected[:, :step]
            correlation.scatter_(1, taken.clamp_min(0), float("-inf"))
        best, candidate = correlation.max(dim=1)
        live = live & (best > tol)
        if not bool(live.any()):
            break
        selected[rows[live], step] = candidate[live]
        A = atoms[selected[:, :step + 1].clamp_min(0)]*(selected[:, :step + 1] >= 0)[:, :, None]
        c = _gradient_step(A, target, torch.cat([c, c.new_zeros(B, 1)], dim=1), live)
        residual = target - torch.einsum("bs,bsd->bd", c, A)
    used = c.shape[1]
    A = atoms[selected[:, :used].clamp_min(0)]
    contribution = c*A.norm(dim=2)
    active = (contribution > ACTIVE_TOL*target.norm(dim=1, keepdim=True)) & (selected[:, :used] >= 0)
    support = torch.full((B, k), -1, dtype=torch.long, device=X.device)
    coefficients = torch.zeros(B, k, device=X.device)
    for b in range(B):
        kept = active[b].nonzero().flatten()
        support[b, :len(kept)] = selected[b, kept]
        coefficients[b, :len(kept)] = c[b, kept]
    return support, coefficients


def shares(support: np.ndarray, coefficients: np.ndarray) -> list[dict[int, float]]:
    """Concept shares z_c of each row, coefficient over the row's sum (detect.py shares)."""
    coef = coefficients.astype(np.float64)
    total = coef.sum(axis=1, keepdims=True)
    frac = np.divide(coef, total, out=np.zeros_like(coef), where=total > 0)
    return [{int(t): float(v) for t, v in zip(s, f) if t >= 0} for s, f in zip(support, frac)]


def surplus(actual: dict[int, float], reference: dict[int, float]) -> dict[int, float]:
    """e_c = [z_c - z~_c]+ for one agent, position and layer."""
    return {c: v - reference.get(c, 0.0) for c, v in actual.items() if v > reference.get(c, 0.0)}


class Lens:
    """The released Qwen3-32B lens and output head, on `device`, without the model. A layer's atoms are built on
    demand. J is read with the J-space repo's own loader so the file format stays theirs."""

    def __init__(self, model_dir, lens_file, device="cuda"):
        import json
        from pathlib import Path
        from safetensors import safe_open
        from transformers import AutoConfig
        from utils.release import load_released_lens
        model_dir = Path(model_dir)
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.eps = config.rms_norm_eps
        self.device = device
        lens, _ = load_released_lens(Path(lens_file))
        self.J = lens.jacobians
        self.layers = sorted(self.J)
        index = json.loads((model_dir/"model.safetensors.index.json").read_text())["weight_map"]
        tensors = {}
        for key in ("lm_head.weight", "model.norm.weight"):
            with safe_open(model_dir/index[key], framework="pt", device="cpu") as source:
                tensors[key] = source.get_tensor(key)
        from transformers import AutoTokenizer
        self.vocab = len(AutoTokenizer.from_pretrained(model_dir, local_files_only=True))
        self.head = tensors["lm_head.weight"][:self.vocab].to(device).float()
        self.norm_weight = tensors["model.norm.weight"].to(device).float()

    def top(self, H: torch.Tensor, layer: int, k: int = TOPK):
        """top-k token ids and logits of the lens readout at rows of H."""
        logits = lens_logits(H.to(self.device), self.J[layer].to(self.device), self.norm_weight, self.head, self.eps)
        return logits.topk(k, dim=-1)

    def logits(self, H: torch.Tensor, layer: int) -> torch.Tensor:
        return lens_logits(H.to(self.device), self.J[layer].to(self.device), self.norm_weight, self.head, self.eps)

    def atoms(self, layer: int):
        a = self.head @ self.J[layer].to(self.device).float()
        return a, torch.linalg.vector_norm(a, dim=1)
