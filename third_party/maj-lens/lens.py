"""The Jacobian lens (Anthropic's J-lens; released lenses from Neuronpedia, loaded by utils/release.py).

For a layer-l residual h_l, the lens readout is W_U norm(J_l h_l): the output head applied, through the model's final
RMSNorm, to the released Jacobian J_l times h_l (jlens/hf.py `unembed`). Its top-10 tokens are what a model "holds"
at that position. The atoms v_t = W_U[t] J_l, one per vocabulary token t, form the dictionary of the sparse
nonnegative J-space decomposition used by J-PACT.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import torch

from config import K, TOPK


def residual(hidden, stream_reduce=None):
    """The residual the lens reads. DeepSeek-V4-Flash keeps four residual streams per layer; its released lens
    reads their mean."""
    if stream_reduce == 'mean':
        if hidden.ndim != 4 or hidden.shape[-2] != 4:
            raise ValueError('expected four DeepSeek post-block residual streams')
        hidden = hidden.mean(dim=-2)
    elif stream_reduce is not None:
        raise ValueError('unknown residual stream reduction')
    if hidden.ndim != 3:
        raise ValueError('expected batch/sequence/hidden residuals')
    return hidden


def end_of_response(hidden, stream_reduce=None):
    """The observation J-PACT and the probe score: the residual at the last token of a public response."""
    return residual(hidden, stream_reduce)[0, -1]


def dictionary(readout, layer):
    """Atoms v_t = W_U[t] J_l for every vocabulary token t, as rows."""
    return readout.head[:readout.vocab_size].float() @ readout.jacobians[layer].float()


def decompose(X, atoms, k=K):
    """Sparse nonnegative J-space decomposition of each row of X: TransformerLens's gradient pursuit, one row at a
    time, k atoms. Returns support [rows, k] (token ids, -1 unused) and coefficients [rows, k]."""
    from transformer_lens.tools.analysis import get_sparse_decomposition
    support = np.full((len(X), k), -1, dtype=np.int32)
    coordinates = np.zeros((len(X), k), dtype=np.float32)
    for i in range(len(X)):
        result = get_sparse_decomposition(X[i], atoms, k, algorithm='gradient_pursuit')
        active = len(result.support)
        support[i, :active] = result.support.numpy()
        coordinates[i, :active] = result.coordinates.detach().cpu().numpy()
    return support, coordinates


def decompose_cached(path, X, atoms, k=K):
    """decompose(), stored at `path` and read from there when present."""
    if path.exists():
        z = np.load(path)
        if len(z['support']) != len(X):
            raise ValueError(f'{path}: {len(z["support"])} rows cached, {len(X)} expected')
        return z['support'], z['coordinates']
    support, coordinates = decompose(X, atoms, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, support=support, coordinates=coordinates)
    return support, coordinates


def ontology_directions(head, jacobian, target_ids, control_ids):
    """Unit directions (W_U[w] - mean_c W_U[c]) J_l for each ontology word w against the control words c."""
    direction = (head[target_ids] - head[control_ids].mean(0)) @ jacobian
    return direction/direction.norm(dim=1, keepdim=True)


class Readout:
    """Forward hooks on chosen layers of a loaded model; after a forward pass, `top` gives the J-lens top-10 at
    chosen positions."""

    def __init__(self, model, spec, layers):
        self.layers = sorted(layers)
        r = spec.readout(layers=set(self.layers), device='cuda')
        self.head = r.head[:r.vocab_size].float()
        self.J = {l: r.jacobians[l].float().cuda() for l in self.layers}
        self.norm = model.model.norm                                    # the final RMSNorm before the output head
        self.cap = {}
        self.handles = [model.model.layers[l].register_forward_hook(partial(self.hook, l)) for l in self.layers]

    def hook(self, l, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        self.cap[l] = h.mean(dim=-2)[0] if h.ndim == 4 else h[0]      # DeepSeek: mean of its four streams

    def logits(self, l, positions):
        H = self.cap[l][torch.as_tensor(positions, device=self.cap[l].device)].float().to(self.J[l].device)
        x = self.norm((H @ self.J[l].T).to(self.norm.weight.device))
        return x.float().to(self.head.device) @ self.head.T

    def top(self, positions, layers=None):
        return {str(l): self.logits(l, positions).topk(TOPK, dim=-1).indices.tolist() for l in (layers or self.layers)}

    def close(self):
        for h in self.handles:
            h.remove()
        self.J.clear(); self.cap.clear()
        torch.cuda.empty_cache()
