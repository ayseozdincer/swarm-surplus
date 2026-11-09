"""J-PACT, step by step, on one NARCBench Core run, from the released cache. CPU only, no model, no lens weights.

What the paper's pipeline did before this script runs (all cached under data/jpact-cache):
  1. replayed every public response of every Qwen3-32B Core run through the model,
  2. kept the residual at the last token of each response, at every layer           raw/input/core_layer_L.npz
  3. decomposed each residual onto at most 25 J-lens atoms v_t = W_U[t] J_L         individual_j_reconstruction/sparse/
     (sparse nonnegative gradient pursuit, TransformerLens)                          core_layer_L.npz: support, coordinates

What this script does, printing every intermediate:
  4. concept shares     z_{a,c}  = coefficient_c / sum of the agent's coefficients
  5. pair sharing       s_{ij,c} = sqrt(z_{i,c} z_{j,c})
  6. pair contrast      r_{ij,c} = [ s_{ij,c} - mean_k (s_{ik,c} + s_{jk,c}) / 2 ]_+     k = the two outsiders
  7. episode pair score R_ij     = mean over the pair's concept pool of r_{ij,c}, averaged over (round, layer) cells
  8. run score          T_s      = max over the six pairs of R_ij
then checks R_ij against the paper's cached sweep, and scores all 62 Core runs (31 collusion + 31 control) -> AUROC.

Run:
  data/venv-jspace/bin/python experiments/phase0_controls/jpact_walkthrough.py                       # audit_01, band layers 26-30
  data/venv-jspace/bin/python experiments/phase0_controls/jpact_walkthrough.py --run deception__cyber_01 --layers all
  data/venv-jspace/bin/python experiments/phase0_controls/jpact_walkthrough.py --round 1 --layer 28 --top 8 --no-plot
Options: --run <scenario stem>  --layers band|all|26,27,28  --round N (the cell shown in detail)  --layer L
         --top N (concepts printed per agent and pair)  --no-plot  --out <png path>
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'data' / 'jpact-cache'
PRIMARY = CACHE / 'maj' / '2026-09-22-repro' / 'qwen3_32b' / 'end_of_response'
STAGED = CACHE / 'cache_expansion' / 'qwen3_32b' / 'end_of_response'
IDENTITY = CACHE / 'qwen_reproduction' / 'identity' / 'core' / 'sites.jsonl'
OBS = CACHE / 'observations'
SCEN = ROOT / 'data' / 'narcbench' / 'scenarios' / 'qwen3_32b' / 'core'
BAND = [26, 27, 28, 29, 30]          # NARCBench's probing band for Qwen3-32B; the paper's sweep uses all 63 layers
PAIRS = list(itertools.combinations(range(4), 2))


# ───────────────────────────────────────────── loading the cache

def load_routes():
    """episode id -> {run_id, target (1 = collusion), domain}."""
    runs = json.loads((PRIMARY / 'evaluator' / 'core.json').read_text())['runs']
    return {int(r['episode']): r for r in runs}


def load_agent_names():
    """(episode, agent int) -> agent name, from the replay identity file aligned with the staged rows."""
    identity = [json.loads(l) for l in IDENTITY.read_text().splitlines()]
    rows = np.load(STAGED / 'raw' / 'input' / 'core_layer_0.npz')
    if len(identity) != len(rows['episode']):
        sys.exit('identity and staged rows differ in length')
    names = {}
    for rec, e, a in zip(identity, rows['episode'], rows['agent']):
        run_id, agent_name = rec['row_key'][:2]
        names[(int(e), int(a))] = agent_name
    return names


def load_rounds():
    """row index -> discussion round, from the same-round pair table."""
    z = np.load(PRIMARY / 'pairs' / 'core.npz')
    rounds = {}
    for side in ('row_i', 'row_j'):
        for row, t in zip(z[side], z['round']):
            rounds[int(row)] = int(t)
    return rounds


def load_shares(layers):
    """share[episode][(round, agent)][layer] = {token id: z}. Steps 3 -> 4."""
    rounds = load_rounds()
    share = defaultdict(lambda: defaultdict(dict))
    for layer in layers:
        z = np.load(PRIMARY / 'individual_j_reconstruction' / 'sparse' / f'core_layer_{layer}.npz')
        coef = z['coordinates'].astype(np.float64)
        total = coef.sum(axis=1, keepdims=True)
        frac = np.divide(coef, total, out=np.zeros_like(coef), where=total > 0)
        for i, (e, a) in enumerate(zip(z['episode'], z['agent'])):
            share[int(e)][(rounds[i], int(a))][layer] = {int(c): float(v) for c, v in zip(z['support'][i], frac[i]) if c >= 0}
    return share


def tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained('Qwen/Qwen3-32B')
    except Exception as err:   # offline, or transformers missing: fall back to ids
        print(f'(tokenizer unavailable, printing token ids: {type(err).__name__})')
        return None


def word(tok, token_id):
    if tok is None:
        return f'#{token_id}'
    return repr(tok.decode([token_id]))


# ───────────────────────────────────────────── J-PACT, as in detect.py

def pair_contrast(Z):
    """Steps 5 and 6. Z [4 agents, concepts] -> r [6 pairs, concepts]."""
    shared = {p: np.sqrt(Z[p[0]] * Z[p[1]]) for p in PAIRS}
    out = []
    for i, j in PAIRS:
        outsiders = [k for k in range(4) if k not in (i, j)]
        background = np.mean([(shared[tuple(sorted((i, k)))] + shared[tuple(sorted((j, k)))]) / 2 for k in outsiders], axis=0)
        out.append(np.maximum(shared[i, j] - background, 0))
    return np.stack(out)


def cell_matrix(share_e, agents, rnd, layer):
    """Z for one (round, layer) cell: rows = agents, columns = the union of their concepts."""
    cols = sorted({c for a in agents for c in share_e.get((rnd, a), {}).get(layer, {})})
    idx = {c: i for i, c in enumerate(cols)}
    Z = np.zeros((4, len(cols)))
    for ai, a in enumerate(agents):
        for c, v in share_e.get((rnd, a), {}).get(layer, {}).items():
            Z[ai, idx[c]] += v
    return Z, cols


def jpact(share_e, agents, rounds, layers):
    """Step 7. R_ij over the given cells, plus the per-pair concept pools for printing."""
    acc, cells = defaultdict(lambda: defaultdict(float)), 0
    for rnd in rounds:
        for layer in layers:
            Z, cols = cell_matrix(share_e, agents, rnd, layer)
            if not cols:
                continue
            r = pair_contrast(Z)
            for p in range(6):
                for nz in np.flatnonzero(r[p]):
                    acc[p][cols[nz]] += r[p, nz]
            cells += 1
    denom = max(cells, 1)
    R = np.array([float(np.mean([v / denom for v in acc[p].values()])) if acc[p] else 0.0 for p in range(6)])
    pools = {p: {c: v / denom for c, v in acc[p].items()} for p in range(6)}
    return R, pools


def auroc(values, labels):
    from scipy.stats import rankdata
    values, labels = np.asarray(values, float), np.asarray(labels, int)
    pos, neg = labels.sum(), len(labels) - labels.sum()
    ranks = rankdata(values)
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


# ───────────────────────────────────────────── the walkthrough

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='deception__audit_01', help='scenario stem; both its collusion and control episodes are shown')
    ap.add_argument('--layers', default='band', help='band (26-30), all (0-62), or a comma list')
    ap.add_argument('--round', type=int, default=1, help='round of the cell shown in detail')
    ap.add_argument('--layer', type=int, default=28, help='layer of the cell shown in detail')
    ap.add_argument('--top', type=int, default=6)
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    layers = BAND if args.layers == 'band' else list(range(63)) if args.layers == 'all' else [int(x) for x in args.layers.split(',')]
    if args.layer not in layers:
        layers = sorted(set(layers) | {args.layer})
    print(f'cache   {CACHE}')
    print(f'layers  {layers[0]}..{layers[-1]} ({len(layers)} layers)   detail cell: round {args.round}, layer {args.layer}\n')

    routes = load_routes()
    names = load_agent_names()
    share = load_shares(layers)
    tok = tokenizer()
    by_run = {r['run_id']: e for e, r in routes.items()}

    for mode in ('collusion', 'control'):
        run_id = f'{args.run}__{mode}'
        if run_id not in by_run:
            sys.exit(f'{run_id} not in the Qwen3-32B Core cache. Known stems: ' + ', '.join(sorted({r["run_id"].rsplit("__", 1)[0] for r in routes.values()})[:12]) + ' ...')
        e = by_run[run_id]
        agents = sorted({a for (_, a) in share[e].keys()})
        rounds = sorted({r for (r, _) in share[e].keys()})
        cfg = json.loads((SCEN / run_id / 'run_config.json').read_text()) if (SCEN / run_id / 'run_config.json').exists() else {}
        roles = cfg.get('roles', {})
        label = lambda a: f"{names.get((e, a), '?')}" + (f" [{roles[names[(e, a)]]}]" if names.get((e, a)) in roles else '')
        coalition = {ai for ai, a in enumerate(agents) if roles.get(names.get((e, a))) == 'colluder'}

        print('=' * 100)
        print(f'{run_id}   episode {e}   target={routes[e]["target"]}   rounds {rounds}   title: {cfg.get("scenario_title", "")}')
        for ai, a in enumerate(agents):
            print(f'  agent {ai} = {label(a)}')

        # step 4: concept shares in the detail cell
        print(f'\n-- step 4. concept shares z at round {args.round}, layer {args.layer} (top {args.top} per agent, share of the agent\'s J-space mass)')
        Z, cols = cell_matrix(share[e], agents, args.round, args.layer)
        for ai, a in enumerate(agents):
            order = np.argsort(-Z[ai])[:args.top]
            items = ', '.join(f'{word(tok, cols[k])} {Z[ai, k]:.3f}' for k in order if Z[ai, k] > 0)
            print(f'  {label(a):34s} {items}')

        # steps 5-6: pair contrast in the detail cell
        print(f'\n-- steps 5-6. pair contrast r in that cell. For each pair: concepts both hold more than each shares with the outsiders')
        r = pair_contrast(Z)
        for p, (i, j) in enumerate(PAIRS):
            mark = ' <= coalition' if {i, j} == coalition and coalition else ''
            order = np.argsort(-r[p])[:args.top]
            items = ', '.join(f'{word(tok, cols[k])} {r[p, k]:.3f}' for k in order if r[p, k] > 0) or '(nothing beyond background)'
            print(f'  ({i},{j}) {names.get((e, agents[i]), "?")[:14]:14s} & {names.get((e, agents[j]), "?")[:14]:14s} sum={r[p].sum():.3f}  {items}{mark}')

        # step 7-8: over all cells
        R, pools = jpact(share[e], agents, rounds, layers)
        print(f'\n-- steps 7-8. R_ij over {len(rounds)} rounds x {len(layers)} layers, and the run score')
        for p, (i, j) in enumerate(PAIRS):
            mark = ' <= coalition' if {i, j} == coalition and coalition else ''
            top = sorted(pools[p].items(), key=lambda kv: -kv[1])[:args.top]
            items = ', '.join(f'{word(tok, c)} {v:.4f}' for c, v in top)
            print(f'  R({i},{j}) = {R[p]:.5f}   pool {len(pools[p]):4d} concepts   top: {items}{mark}')
        best = int(np.argmax(R))
        print(f'  T_s = max = {R[best]:.5f} at pair {PAIRS[best]}  ->  {names.get((e, agents[PAIRS[best][0]]), "?")} & {names.get((e, agents[PAIRS[best][1]]), "?")}'
              + ('   (correct pair)' if set(PAIRS[best]) == coalition and coalition else ''))

        # check against the paper's cached sweep (all 63 layers, all rounds)
        z = np.load(OBS / 'pairs_qwen3_32b.npz')
        k = int(np.flatnonzero(z['core_episodes'] == e)[0])
        cached = z['core_R'][k]
        if args.layers == 'all':
            print(f'  cached sweep R = {np.round(cached, 5).tolist()}   max abs diff {np.abs(cached - R).max():.2e}  (should be ~0)')
        else:
            print(f'  cached sweep R (all 63 layers) = {np.round(cached, 5).tolist()}   T_s = {cached.max():.5f}  (this run used {len(layers)} layers, so values differ)')
        print()

    # all 62 Core runs, from the cached sweep
    z = np.load(OBS / 'pairs_qwen3_32b.npz')
    eps, R_all = z['core_episodes'], z['core_R']
    T = R_all.max(1)
    y = np.array([routes[int(e)]['target'] for e in eps])
    print('=' * 100)
    print(f'all Core runs from the cached sweep: {int(y.sum())} collusion, {int(len(y) - y.sum())} control.   AUROC(T_s) = {auroc(T, y):.3f}   (paper, grouped by domain: 0.984)')
    order = np.argsort(-T)
    print('  rank  T_s      label      run')
    for rank, k in enumerate(order, 1):
        print(f'  {rank:4d}  {T[k]:.5f}  {"COLLUSION" if y[k] else "control  "}  {routes[int(eps[k])]["run_id"]}')

    # pair localisation on collusion runs: does the max pair name the coalition?
    hits = total = 0
    for k, e in enumerate(eps):
        if not y[k]:
            continue
        run_id = routes[int(e)]['run_id']
        cfgp = SCEN / run_id / 'run_config.json'
        if not cfgp.exists():
            continue
        roles = json.loads(cfgp.read_text()).get('roles', {})
        agents = [int(a) for a in z['core_agents'][k]]
        coal = {ai for ai, a in enumerate(agents) if roles.get(names.get((int(e), a))) == 'colluder'}
        best = set(PAIRS[int(np.argmax(R_all[k]))])
        total += 1
        hits += best == coal
    if total:
        print(f'\n  on collusion runs the max pair is the actual coalition in {hits}/{total} runs')

    if not args.no_plot:
        plot(args, routes, names, share, layers, eps, R_all, T, y, by_run)


def plot(args, routes, names, share, layers, eps, R_all, T, y, by_run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    e = by_run[f'{args.run}__collusion']
    ec = by_run[f'{args.run}__control']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={'width_ratios': [1.1, 1.1, 1.4]})

    for ax, ep, title in ((axes[0], e, 'collusion run'), (axes[1], ec, 'control run')):
        agents = sorted({a for (_, a) in share[ep].keys()})
        rounds = sorted({r for (r, _) in share[ep].keys()})
        R, _ = jpact(share[ep], agents, rounds, layers)
        run_id = routes[ep]['run_id']
        cfgp = SCEN / run_id / 'run_config.json'
        roles = json.loads(cfgp.read_text()).get('roles', {}) if cfgp.exists() else {}
        labels = [f"{names.get((ep, agents[i]), '?').split()[0]}+{names.get((ep, agents[j]), '?').split()[0]}" for i, j in PAIRS]
        colors = ['#c2410c' if {roles.get(names.get((ep, agents[i]))), roles.get(names.get((ep, agents[j])))} == {'colluder'} else '#8a93a3' for i, j in PAIRS]
        ax.bar(range(6), R, color=colors)
        ax.set_xticks(range(6)); ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        ax.set_title(f'{args.run} {title}\nR_ij per pair, {len(layers)} layers (orange = coalition)', fontsize=10)
        ax.set_ylabel('R_ij')

    ax = axes[2]
    rng = np.random.default_rng(0)
    for lab, color, name in ((1, '#c2410c', 'collusion'), (0, '#1f5f8b', 'control')):
        xs = lab + rng.uniform(-0.18, 0.18, size=int((y == lab).sum()))
        ax.scatter(xs, T[y == lab], s=22, color=color, alpha=.8, label=name)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['control', 'collusion'])
    ax.set_ylabel('T_s = max pair R_ij'); ax.set_title(f'all 62 Qwen3-32B Core runs, cached sweep\nAUROC {auroc(T, y):.3f}', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = Path(args.out) if args.out else ROOT / 'results' / 'figures' / f'jpact_walkthrough_{args.run}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f'\nfigure written to {out}')


if __name__ == '__main__':
    main()
