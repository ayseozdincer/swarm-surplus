"""Detect: J-PACT, the ontology baseline and NARCBench's probe on NARCBench (Core, Transfer, Stego).

J-PACT, per episode (four agents), round and layer, from each agent's sparse J-space decomposition (lens.decompose):
  z_{a,c}  = coefficient of token atom c / sum of the agent's coefficients                 (concept shares)
  s_{ij,c} = sqrt(z_{i,c} z_{j,c})
  r_{ij,c} = [s_{ij,c} - mean_{k not in ij} (s_{ik,c} + s_{jk,c}) / 2]+                   (pair contrast)
  R_ij     = mean over the pair's pool (every c with r_{ij,c} > 0 somewhere) of r_{ij,c}, averaged over the
             episode's rounds and layers
  T_s      = max over the six pairs of R_ij                                                 (episode score)
No labels enter the score. The ontology baseline scores each pair by the mean of its two agents' cosines with
the ontology directions (lens.ontology_directions), averaged over words, layers and rounds.

  python detect.py sweep <model>   J-PACT and the ontology baseline at every observation count (GPU)
  python detect.py probe           NARCBench's probe on every band layer (CPU)
  python detect.py report          results/main/{detection,observations,localisation}.tex from the cache
"""
from __future__ import annotations

import itertools
import json
import sys
import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

import lens
from cache import OBSERVATIONS, PRIMARY, PROBE, ROUNDS
from config import (BOOTSTRAP_SEED, FPR, K, MAX_SUBSETS, MIN_DETECTED, MODELS, NARC_BANDS, PAIRS, PROBE_SEED,
                    RESAMPLES, SHOWN, SITE, TIERS)
from utils import data, tables
from utils.io import immutable, run_jobs
from utils.models import get_model


# ═══════════════════════════════════════════════════════════════════ J-PACT and the ontology baseline

def shares(support, coordinates):
    """Concept shares z_{a,c} of each row: {token: coefficient / row sum}."""
    coef = coordinates.astype(np.float64)
    total = coef.sum(axis=1, keepdims=True)
    frac = np.divide(coef, total, out=np.zeros_like(coef), where=total > 0)
    return [{int(c): float(v) for c, v in zip(s, f) if c >= 0} for s, f in zip(support, frac)]


def pair_contrast(Z):
    """Z [4 agents, concepts] -> r [6 pairs, concepts]: pair-shared mass beyond what each member shares with the
    other two agents."""
    shared = {p: np.sqrt(Z[p[0]] * Z[p[1]]) for p in PAIRS}
    return np.stack([np.maximum(shared[i, j] - np.mean([(shared[tuple(sorted((i, k)))] + shared[tuple(sorted((j, k)))]) / 2
                                                         for k in range(4) if k not in (i, j)], axis=0), 0)
                     for i, j in PAIRS])


def jpact(share, agents, rounds, layers):
    """R_ij of one episode over the given rounds and layers; share[(round, agent)][layer] = {token: z}."""
    acc, cells = defaultdict(lambda: defaultdict(float)), 0
    for rnd in rounds:
        for layer in layers:
            cols = sorted({c for a in agents for c in share.get((rnd, a), {}).get(layer, {})})
            if not cols:
                continue
            idx = {c: i for i, c in enumerate(cols)}
            Z = np.zeros((4, len(cols)))
            for ai, a in enumerate(agents):
                for c, v in share.get((rnd, a), {}).get(layer, {}).items():
                    Z[ai, idx[c]] += v
            r = pair_contrast(Z)
            for p in range(6):
                for nz in np.flatnonzero(r[p]):
                    acc[p][cols[nz]] += r[p, nz]
            cells += 1
    denom = max(cells, 1)
    return np.array([float(np.mean([v/denom for v in acc[p].values()])) if acc[p] else 0.0 for p in range(6)])


def sweep(model):
    """J-PACT (R) and the ontology baseline (C) with 1..n observations (rounds) per agent, every released layer.
    Core/Transfer read the replayed end-of-response activations; Stego reads every discussion round of its replay.
    Writes observations/<model>.json (AUROC per count), pairs_<model>.npz (pair scores at the largest count) and
    subsets_<model>.npz (J-PACT episode scores per round subset, for the table's intervals)."""
    import torch
    spec = get_model(model)
    ontology = spec.ontology()
    layers = sorted(spec.all_layers)
    readout = spec.readout(tuple(layers), device='cuda')
    head = readout.head[:readout.vocab_size].float()
    target = torch.tensor(list(ontology['target_ids_by_word'].values()), device='cuda')
    control = torch.tensor(list(ontology['control_ids_by_word'].values()), device='cuda')
    rows, pairs, subsets_out = [], {}, {}
    for tier in TIERS:
        if tier == 'stego':
            vectors, meta = data.replay_rows(model, tier, layers, ROUNDS/model)
            cached = lambda layer: ROUNDS/model/'jspace'/f'{tier}_layer_{layer}.npz'
        else:
            vectors, meta = data.primary_rows(model, tier, layers)
            cached = lambda layer: PRIMARY/model/SITE/'individual_j_reconstruction'/'sparse'/f'{tier}_layer_{layer}.npz'
        share, unary = defaultdict(dict), defaultdict(list)
        for layer in layers:
            J = readout.jacobians[layer].float()
            X = torch.tensor(vectors(layer), device='cuda')
            cos = (torch.nn.functional.normalize(X, dim=1)
                   @ lens.ontology_directions(head, J, target, control).T).mean(1).cpu().numpy()
            support, coordinates = lens.decompose_cached(cached(layer), X, lens.dictionary(readout, layer), K)
            for m, c, z in zip(meta, cos, shares(support, coordinates)):
                unary[m['episode'], m['round'], m['agent']].append(float(c))
                share[m['episode']].setdefault((m['round'], m['agent']), {})[layer] = z
            del X; torch.cuda.empty_cache()

        routes = {int(r['episode']): r for r in data.routes(model, tier)}
        episodes = sorted({m['episode'] for m in meta})
        labels = np.array([routes[e]['target'] for e in episodes])
        groups = np.array([GROUP[tier](routes[e]) for e in episodes])
        agents_of = {e: sorted({m['agent'] for m in meta if m['episode'] == e}) for e in episodes}
        rounds_present = sorted({m['round'] for m in meta})
        for n in range(1, len(rounds_present) + 1):
            subsets = list(itertools.combinations(rounds_present, n))[:MAX_SUBSETS]
            got = defaultdict(list)
            for keep in subsets:
                R, C = [], []
                for e in episodes:
                    ags = agents_of[e]
                    if len(ags) != 4:
                        R.append(np.zeros(6)); C.append(np.zeros(6)); continue
                    R.append(jpact(share[e], ags, keep, layers))
                    u = {a: float(np.mean([np.mean(unary.get((e, rnd, a), [0.])) for rnd in keep])) for a in ags}
                    C.append(np.array([(u[ags[i]] + u[ags[j]])/2 for i, j in PAIRS]))
                R, C = np.array(R), np.array(C)
                if n == len(rounds_present) and keep == subsets[0]:
                    pairs.update({f'{tier}_episodes': np.array(episodes, dtype=np.int64),
                                  f'{tier}_agents': np.array([agents_of[e] for e in episodes], dtype=np.int64),
                                  f'{tier}_R': R, f'{tier}_C': C, f'{tier}_n_obs': np.array([n], dtype=np.int64)})
                subsets_out.setdefault(f'{tier}_n{n}_R', []).append(R.max(1))
                subsets_out[f'{tier}_episodes'] = np.array(episodes, dtype=np.int64)
                for arm, scores in (('R', R), ('C', C)):
                    got[arm].append(grouped_auroc(scores.max(1), labels, groups))
            rows.append({'model': model, 'tier': tier, 'n_obs': n, 'subsets': len(subsets),
                         **{a: float(np.mean(got[a])) for a in ('R', 'C')},
                         **{f'{a}_sd': float(np.std(got[a])) for a in ('R', 'C')}})
            print(f"{model} {tier} n={n}: R={rows[-1]['R']:.3f} C={rows[-1]['C']:.3f}", flush=True)
    OBSERVATIONS.mkdir(parents=True, exist_ok=True)
    (OBSERVATIONS/f'{model}.json').write_text(json.dumps(rows, indent=1))
    np.savez(OBSERVATIONS/f'pairs_{model}.npz', **pairs)
    np.savez(OBSERVATIONS/f'subsets_{model}.npz', **{k: np.stack(v) if isinstance(v, list) else v
                                                       for k, v in subsets_out.items()})


def swept(model, tier, arm):
    """{episode: T_s} of one arm at the sweep's largest observation count."""
    z = np.load(OBSERVATIONS/f'pairs_{model}.npz')
    return dict(zip(z[f'{tier}_episodes'].tolist(), z[f'{tier}_{arm}'].max(axis=1)))


def swept_subsets(model, tier, n):
    """[{episode: T_s}] of J-PACT per round subset at observation count n, or None."""
    z = np.load(OBSERVATIONS/f'subsets_{model}.npz')
    key = f'{tier}_n{n}_R'
    return [dict(zip(z[f'{tier}_episodes'].tolist(), row)) for row in z[key]] if key in z else None


def sweep_rows():
    return [r for m in MODELS for r in json.loads((OBSERVATIONS/f'{m}.json').read_text())]


# ═══════════════════════════════════════════════════════════════════ NARCBench's probe (baseline)

def probe_job(model, layer):
    """NARCBench's LR probe at one layer, unchanged (utils/narcbench.py imports it): fitted on Core colluders vs honest
    agents, scored with its five episode aggregators; Core leave-one-domain-out, Transfer from the all-Core fit,
    Stego on every discussion round from the all-Core fit."""
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    from utils.narcbench import ALL_METHODS, agent_activation_means, fit_core_probe, fit_core_aggregator_and_score
    path = PROBE/'probe'/model/f'layer_{layer}.json'
    if path.exists():
        return f'{path} cached'
    train = data.training(model)
    domains, routes = data.core_routes(train)
    rows = data.raw_rows(model, 'core', layer)
    groups = agent_activation_means(rows)
    ids = sorted(groups, key=lambda i: routes[i]['order'])
    target = lambda chosen: np.array([routes[i]['target'] for i in chosen])
    vectors, meta = data.replay_rows(model, 'stego', [layer], ROUNDS/model)
    stego = agent_activation_means({'episode': np.array([m['episode'] for m in meta]),
                                    'agent': np.array([m['agent'] for m in meta]), 'vectors': vectors(layer)})
    scores = {m: {t: {} for t in TIERS} for m in ALL_METHODS}
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        for d in domains:
            fit = [i for i in ids if routes[i]['fold'] != d]
            test = [i for i in ids if routes[i]['fold'] == d]
            scaler, w = fit_core_probe(rows, train, set(domains) - {d}, layer, PROBE_SEED)
            result = fit_core_aggregator_and_score(groups, test, scaler, w, groups, fit, target(fit), methods=ALL_METHODS)
            for m in ALL_METHODS:
                scores[m]['core'].update({str(i): v for i, v in result[m].items()})
        scaler, w = fit_core_probe(rows, train, set(domains), layer, PROBE_SEED)
        for tier, test in (('transfer', agent_activation_means(data.raw_rows(model, 'transfer', layer))), ('stego', stego)):
            result = fit_core_aggregator_and_score(test, sorted(test), scaler, w, groups, ids, target(ids), methods=ALL_METHODS)
            for m in ALL_METHODS:
                scores[m][tier] = {str(i): v for i, v in result[m].items()}
    immutable(path, {'model': model, 'layer': layer, 'seed': PROBE_SEED, 'scores': scores})
    return str(path)


def probe_cells(model):
    """One {tier: {episode: score}} per band layer x aggregator: NARCBench reports their mean."""
    return [tiers for layer in NARC_BANDS[model]
            for tiers in data.read(PROBE/'probe'/model/f'layer_{layer}.json')['scores'].values()]


# ═══════════════════════════════════════════════════════════════════ Metrics

GROUP = {'core': lambda r: r['domain'], 'transfer': lambda r: r['family'], 'stego': lambda r: 'Stego'}


def auroc(values, labels):
    """Mann-Whitney AUROC, tied scores sharing their mean rank (equal to sklearn's roc_auc_score)."""
    values, labels = np.asarray(values, float), np.asarray(labels, bool)
    n1, n0 = labels.sum(), (~labels).sum()
    return float((rankdata(values)[labels].sum() - n1*(n1+1)/2)/(n1*n0))


def grouped_auroc(values, labels, groups):
    """NARCBench's metric: AUROC within each group (Core domain, Transfer family, Stego whole), then averaged."""
    values, labels, groups = np.asarray(values, float), np.asarray(labels, int), np.asarray(groups)
    return float(np.mean([auroc(values[groups == g], labels[groups == g]) for g in np.unique(groups)]))


def scenario_draws(run_ids):
    """Bootstrap draws over NARCBench scenarios: a scenario's collusion and control episodes are drawn together."""
    rows = defaultdict(list)
    for i, run in enumerate(run_ids):
        rows[run.rsplit('__', 1)[0]].append(i)
    keys = sorted(rows)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return [np.concatenate([rows[keys[k]] for k in rng.integers(len(keys), size=len(keys))]) for _ in range(RESAMPLES)]


def auroc_draws(model, tier, score_sets):
    """(point, bootstrap draws) of a table cell: the mean over score_sets of the grouped AUROC."""
    routes = data.routes(model, tier)
    episodes = [int(r['episode']) for r in routes]
    labels = np.array([r['target'] for r in routes])
    groups = np.array([GROUP[tier](r) for r in routes])
    values = [np.array([float(s[e] if e in s else s[str(e)]) for e in episodes]) for s in score_sets]
    point = float(np.mean([grouped_auroc(v, labels, groups) for v in values]))
    draws = np.array([np.mean([grouped_auroc(v[i], labels[i], groups[i]) for v in values])
                      for i in scenario_draws([r['run_id'] for r in routes])])
    return point, draws


def interval(draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


# ═══════════════════════════════════════════════════════════════════ Localisation

def localisation(model):
    """Per tier: of the collusion episodes J-PACT detects (T_s above the threshold that at most FPR of honest
    episodes exceed), the share whose top pair splits the four agents along the coalition (the colluding pair or
    its complement; chance 1/3)."""
    z = np.load(OBSERVATIONS/f'pairs_{model}.npz')
    out = {}
    for tier in TIERS:
        coalition = data.colluders(model, tier)
        episodes, R = z[f'{tier}_episodes'], z[f'{tier}_R']
        scores = R.max(1)
        honest = [s for e, s in zip(episodes, scores) if not coalition.get(int(e))]
        threshold = float(np.sort(honest)[len(honest) - 1 - int(np.floor(FPR * len(honest)))])
        detected = hits = 0
        for episode, agents, r, score in zip(episodes, z[f'{tier}_agents'], R, scores):
            if len(coalition.get(int(episode), ())) != 2 or score <= threshold:
                continue
            detected += 1
            i, j = PAIRS[int(np.asarray(r).argmax())]
            top, pair = {int(agents[i]), int(agents[j])}, coalition[int(episode)]
            hits += top == pair or top == {int(a) for a in agents} - pair
        out[tier] = hits/detected if detected >= MIN_DETECTED else None
    return out


# ═══════════════════════════════════════════════════════════════════ Report

def report():
    sweeps = sweep_rows()
    value = lambda model, tier, n, arm: next((r[arm] for r in sweeps if (r['model'], r['tier'], r['n_obs']) == (model, tier, n)), None)
    top_n = {(m, t): max(r['n_obs'] for r in sweeps if (r['model'], r['tier']) == (m, t)) for m in MODELS for t in TIERS}

    # Detection table: every detector at the largest observation count (Core 3, Transfer 1, Stego 7).
    detection = []
    for model in MODELS:
        cells = probe_cells(model)
        for detector, sets, arm in (('NARCBench probe', lambda t: [c[t] for c in cells], None),
                                    ('Ontology baseline', lambda t: [swept(model, t, 'C')], 'C'),
                                    ('J-PACT', lambda t: [swept(model, t, 'R')], 'R')):
            row = {'model': model, 'detector': detector}
            for tier in TIERS:
                point, draws = auroc_draws(model, tier, sets(tier))
                if arm:   # the sweep's AUROC, which the bootstrap base must reproduce
                    point, base = value(model, tier, top_n[model, tier], arm), point
                    if abs(point - base) > 1e-9:
                        raise ValueError(f'{model}/{tier}/{arm}: bootstrap base differs from the sweep')
                row[tier] = (point, draws)
            detection.append(row)
    for detector in ('NARCBench probe', 'Ontology baseline', 'J-PACT'):
        rows = [r for r in detection if r['detector'] == detector]
        detection.append({'model': 'mean', 'detector': detector,
                          **{t: (float(np.mean([r[t][0] for r in rows])), np.mean([r[t][1] for r in rows], axis=0))
                             for t in TIERS}})
    tables.detection([{**r, **{t: (r[t][0], interval(r[t][1])) for t in TIERS}} for r in detection])

    # Observations table: J-PACT by observations per agent.
    transfer_n = max(r['n_obs'] for r in sweeps if r['tier'] == 'transfer')
    columns = [(tier, transfer_n if tier == 'transfer' else n) for tier, counts in SHOWN.items() for n in counts]
    cells, draws = {}, {}
    for model in MODELS:
        for tier, n in columns:
            v = value(model, tier, n, 'R')
            sets = [swept(model, tier, 'R')] if n == top_n[model, tier] else swept_subsets(model, tier, n)
            if v is not None and sets:
                point, draws[model, (tier, n)] = auroc_draws(model, tier, sets)
                if abs(point - v) > 1e-9:
                    raise ValueError(f'{model}/{tier}/{n}: bootstrap base differs from the sweep')
            cells[model, (tier, n)] = (v, interval(draws[model, (tier, n)]) if (model, (tier, n)) in draws else None)
    for c in columns:
        vs = [cells[m, c][0] for m in MODELS]
        cells['mean', c] = (None if None in vs else float(np.mean(vs)),
                            interval(np.mean([draws[m, c] for m in MODELS], axis=0)) if all((m, c) in draws for m in MODELS) else None)
    tables.observations(cells, columns)

    tables.localisation({m: localisation(m) for m in MODELS})


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'sweep':
        sweep(sys.argv[2])
    elif action == 'probe':
        run_jobs([(probe_job, m, l) for m in MODELS for l in NARC_BANDS[m]], int(sys.argv[2]) if len(sys.argv) > 2 else 16)
    elif action == 'report':
        report()
