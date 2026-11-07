"""Reading the cached benchmark rows: activations, their episode/agent/round identities, and the labels.

Row files hold numbers only (episode and agent as opaque integers); run ids, agent names, roles and labels are
read from separate files (`evaluator/`, `training/`, the replay identity files) only where a function names them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cache import NARCBENCH, PRIMARY, STAGED, sha256
from config import SITE
from utils.io import numeric, read
from utils.models import get_model

def verified(path, digest):
    if sha256(path) != digest:
        raise ValueError(f'input hash differs from producer manifest: {path}')
    return path


def cell(model, tier, layer):
    return read(PRIMARY/model/SITE/'representations.json')['cells'][f'{tier}/{layer}']


def raw_rows(model, tier, layer):
    """Every public response's end-of-response activation at one layer, hash-checked against its manifest."""
    path = PRIMARY/model/SITE/'raw'/'input'/f'{tier}_layer_{layer}.npz'
    expected = [d for p, d in cell(model, tier, layer)['output_sha256'].items() if Path(p).resolve() == path.resolve()]
    if len(expected) != 1 or expected[0] != sha256(path):
        raise ValueError(f'cache hash differs from manifest: {path}')
    return numeric(path)


def routes(model, tier):
    """Evaluation labels: per episode its run id, target (1 = collusion), Core domain or Transfer family."""
    return read(PRIMARY/model/SITE/'evaluator'/f'{tier}.json')['runs']


def training(model):
    """Core training targets for the NARCBench probe (agent level) and its folds (run level)."""
    return read(PRIMARY/model/SITE/'training'/'individual.json')


def core_routes(train):
    """Core domains and per-episode fold/target/order."""
    rows = train['run_targets']
    domains = sorted({r['fold'] for r in rows})
    if len(domains) < 3 or len({r['episode'] for r in rows}) != len(rows):
        raise ValueError('invalid Core routes')
    return domains, {r['episode']: r for r in rows}


def row_rounds(model, tier):
    """Discussion round of every response row, from the hash-bound same-round pair table."""
    path = verified(PRIMARY/model/SITE/'pairs'/f'{tier}.npz', cell(model, tier, 0)['pair_sha256'])
    pairs = np.load(path)
    rounds = {}
    for side in ('row_i', 'row_j'):
        for row, t in zip(pairs[side], pairs['round']):
            if rounds.setdefault(int(row), int(t)) != int(t):
                raise ValueError('row assigned to two rounds')
    return rounds


def _identity(model, tier):
    """(replay identity rows, numeric raw rows) of one tier, aligned row by row."""
    raw = read(STAGED/model/SITE/'raw'/'manifest.json')['cells'][f'{tier}/0']
    identity_path, = [Path(p) for p in raw['source_sha256'] if '/identity/' in p and p.endswith('sites.jsonl')]
    identity = [json.loads(line) for line in
                verified(identity_path, raw['source_sha256'][str(identity_path)]).read_text().splitlines()]
    rows = numeric(STAGED/model/SITE/'raw'/'input'/f'{tier}_layer_0.npz')
    if len(identity) != len(rows['episode']):
        raise ValueError('identity and raw rows differ')
    return identity, rows


def agent_ids(model, tier):
    """{(run_id, agent_name): agent integer}."""
    identity, rows = _identity(model, tier)
    return {tuple(r['row_key'][:2]): int(a) for r, a in zip(identity, rows['agent'])}


def colluders(model, tier):
    """Evaluation only: {episode: the two colluders' agent integers}, from NARCBench's release metadata."""
    identity, rows = _identity(model, tier)
    to_ids = {tuple(r['row_key'][:2]): (int(e), int(a)) for r, e, a in zip(identity, rows['episode'], rows['agent'])}
    meta = read(NARCBENCH/'activations'/get_model(model).cohort/tier/'metadata_gen.json')
    out = {}
    for r in meta:
        if r['role'] == 'colluder' and (r['run_id'], r['agent_name']) in to_ids:
            e, a = to_ids[r['run_id'], r['agent_name']]
            out.setdefault(e, set()).add(a)
    return out


def primary_rows(model, tier, layers):
    """Core/Transfer: (vectors(layer), meta) for every public response, float32; meta carries episode, agent, round."""
    rounds = row_rounds(model, tier)
    head = np.load(PRIMARY/model/SITE/'raw'/'input'/f'{tier}_layer_{layers[0]}.npz')
    meta = [{'row': i, 'episode': int(e), 'agent': int(a), 'round': int(rounds[i])}
            for i, (e, a) in enumerate(zip(head['episode'], head['agent']))]

    def vectors(layer):
        v = np.asarray(np.load(PRIMARY/model/SITE/'raw'/'input'/f'{tier}_layer_{layer}.npz')['vectors'], dtype=np.float32)
        return v[:, 0] if v.ndim == 3 else v
    return vectors, meta


def replay_rows(model, tier, layers, root):
    """Stego: (vectors(layer), meta) for every discussion round of the replay under `root` (end-of-response site)."""
    run_episode = {r['run_id']: int(r['episode']) for r in routes(model, tier)}
    ids = sorted((json.loads(l) for l in (root/'identity'/tier/'sites.jsonl').read_text().splitlines()),
                 key=lambda r: r['row_index'])
    to_id = agent_ids(model, tier)
    meta = []
    for i, r in enumerate(ids):
        run, agent, _phase, _channel, rnd = r['row_key']
        meta.append({'row': i, 'file': r['row_index'], 'episode': run_episode[run],
                     'agent': to_id[run, agent], 'round': int(rnd)})

    def vectors(layer):
        return np.stack([np.load(root/'public_sites'/tier/f"row_{m['file']:05d}.npz")[f'end_layer_{layer}'][0]
                         for m in meta]).astype(np.float32)
    return vectors, meta
