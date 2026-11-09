#!/usr/bin/env python3
"""The J-space repo's own detect.sweep with one change, lens.decompose replaced by the batched pursuit. The original
decomposes one row at a time, about 135k rows for Qwen3-32B over all 63 layers. tests/gpu_checks.py compares the two.
Run from third_party/maj-lens with PYTHONPATH=.:/workspace/swarm"""
import sys

import numpy as np
import torch

import detect
import lens
from detector import jspace

BATCH = 384


def decompose(X, atoms, k=jspace.K):
    norms = torch.linalg.vector_norm(atoms, dim=1)
    support = np.full((len(X), k), -1, dtype=np.int32)
    coordinates = np.zeros((len(X), k), dtype=np.float32)
    for start in range(0, len(X), BATCH):
        s, c = jspace.pursuit(X[start:start + BATCH], atoms, norms, k)
        support[start:start + len(s)] = s.cpu().numpy()
        coordinates[start:start + len(c)] = c.cpu().numpy()
    return support, coordinates


lens.decompose = decompose
detect.sweep(sys.argv[1])
