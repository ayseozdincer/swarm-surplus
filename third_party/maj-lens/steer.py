"""Steer: swapping two J-lens coordinates in DeepSeek-V4-Flash's Stego receiver redirects the count range it decodes.

Anthropic's clamped lens-coordinate swap between tokens a and b: V = [v_a v_b] (their J-lens vectors
v_t = W_U[t] J_l), clean-pass coordinates c = V^+ h, and at every band layer and every position except the last (the
position that emits the decoded range) the live activation is held at the swapped clean coordinates:
h <- h + V (c* - V^+ h), c* = c + alpha (swap(c) - c). The receiver's decoded range is read from its forced private note
("PRIVATE: Running count is") as the range whose name has the highest next-token logit. Control: the same swap
between random token pairs at the strongest alpha.

  python steer.py    collect the swap stage if missing (GPU), then write the Steer table, its figure and Figure 1
"""
from __future__ import annotations

from functools import partial

import numpy as np

from config import BAND, RANDOM_PAIRS, RANDOM_SEED, RANGES, RECEIVER_FORCE, SWAP, SWAP_ALPHAS
from read import STEGO, collect_core, collect_stego, encode, strip_private
from utils import deepseek as DS, narcbench as NB
from utils.interp import load_stage


class Swap:
    def __init__(self, model, spec, a, b):
        import torch
        self.on, self.record, self.alpha, self.V, self.Vp, self.clean = False, False, 1.0, {}, {}, {}
        r = spec.readout(layers=set(BAND), device='cuda')
        W = r.head[:r.vocab_size].float()
        for l in BAND:
            V = (W[[a, b]] @ r.jacobians[l].float().cuda()).T
            self.V[l], self.Vp[l] = V, torch.linalg.pinv(V)
        self.handles = [model.model.layers[l].register_forward_hook(partial(self.hook, l)) for l in BAND]

    def hook(self, l, module, inputs, output):
        if not (self.on or self.record):
            return None
        h = output[0] if isinstance(output, tuple) else output
        m = h.mean(dim=-2) if h.ndim == 4 else h                     # DeepSeek: the lens reads the stream mean
        c = m.float() @ self.Vp[l].T
        if self.record:                                               # clean pass: keep the coordinates
            self.clean[l] = c
            return None
        c0 = self.clean[l]
        target = c0 + self.alpha * (c0.flip(-1) - c0)                 # clamp to the swapped clean coordinates
        delta = ((target - c) @ self.V[l].T).to(h.dtype)
        delta[:, -1] = 0                                              # the position that emits the decoded range
        h = h + (delta.unsqueeze(-2) if h.ndim == 4 else delta)
        return (h, *output[1:]) if isinstance(output, tuple) else h

    def close(self):
        for h in self.handles:
            h.remove()


def collect_stego_swap(model, tok, spec):
    """Every Stego collusion hand: the receiver's next-token logits for the three range names after the forced
    note, with no swap, the HOT/Negative swap at each alpha, and each random pair at the strongest alpha."""
    ids = {k: tok.encode(v, add_special_tokens=False) for k, v in SWAP.items()}
    assert all(len(v) == 1 for v in ids.values()), ids
    text = tok.batch_decode([[i] for i in range(len(tok))])
    words = [i for i, t in enumerate(text) if t.startswith(' ') and t[1:].isalpha() and t[1:].isascii() and len(t) >= 5]
    pairs = np.random.default_rng(RANDOM_SEED).choice(words, (RANDOM_PAIRS, 2), replace=False).tolist()
    arms = {'swap': Swap(model, spec, ids['HOT'][0], ids['NEGATIVE'][0]),
            **{f'random_{k}': Swap(model, spec, int(a), int(b)) for k, (a, b) in enumerate(pairs)}}
    labels = {r: tok.encode(' ' + r.lower(), add_special_tokens=False)[0] for r in RANGES}
    plan = [(f'alpha_{a}', 'swap' if a > 0 else None, a) for a in (0.0, *SWAP_ALPHAS)]
    plan += [(k, k, max(SWAP_ALPHAS)) for k in arms if k != 'swap']
    rows = []
    for run, ep in NB.episodes(STEGO).items():
        if not run.endswith('collusion'):
            continue
        for hand in range(1, 8):
            turn = NB.turn(ep, hand, ep['big_player'])
            _, pids, _ = encode(tok, {**turn, 'user': strip_private(turn['user'])})
            pids = pids + tok(RECEIVER_FORCE, add_special_tokens=False).input_ids
            row = {'run': run, 'hand': hand, 'range': NB.count_range(ep['running_count'][hand - 1])}
            for key, arm, alpha in plan:
                for k, sw in arms.items():                            # the no-swap pass is the clean pass
                    sw.on, sw.alpha, sw.record = k == arm, alpha, arm is None
                logits = DS.forward(model, pids).logits[0, -1].float()
                row[key] = {r: float(logits[i]) for r, i in labels.items()}
            rows.append(row)
        print('stego_swap', run, flush=True)
    for sw in arms.values():
        sw.on = False; sw.close()
    return {'tokens': ids, 'alphas': list(SWAP_ALPHAS), 'random_pairs': [[text[a], text[b]] for a, b in pairs], 'rows': rows}


def decoded(sw):
    """Share of hands of each true range decoded as each range, per arm; the random pairs also pooled."""
    pick = lambda x: max(RANGES, key=lambda r: x[r])
    keys = [k for k in sw['rows'][0] if k.startswith(('alpha_', 'random_'))]
    n = {t: sum(r['range'] == t for r in sw['rows']) for t in RANGES}
    out = {k: {t: {g: round(sum(pick(r[k]) == g for r in sw['rows'] if r['range'] == t) / n[t], 2) for g in RANGES}
               | {'hands': n[t]} for t in RANGES} for k in keys}
    rnd = [k for k in keys if k.startswith('random_')]
    out['random_pooled'] = {t: {g: round(float(np.mean([out[k][t][g] for k in rnd])), 2) for g in RANGES} | {'hands': n[t]}
                            for t in RANGES}
    out['random_pairs'] = sw['random_pairs']
    return out


def report():
    from utils import figures, tables
    sw = decoded(load_stage('stego_swap', collect_stego_swap))
    tables.interp_steer(sw)
    figures.steer(sw)
    figures.overview(load_stage('core', collect_core), load_stage('stego', collect_stego), sw)


if __name__ == '__main__':
    report()
