"""Check cache provenance before an experiment reads or reuses it.

Only ``utils.build.setup`` opens released NARCBench metadata. Scorers receive the
numeric ``input/`` cache; evaluation labels stay in ``evaluator/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cache import sha256

TIERS = ("core", "transfer", "stego")


def validate_pairs(pairs: dict) -> None:
    """Label-free canonical six-pair coverage at each episode/round."""
    from itertools import combinations
    required = {"row_i", "row_j", "episode", "round", "agent_i", "agent_j", "pair"}
    if set(pairs) != required:
        raise ValueError("unexpected pair table fields")
    n = len(pairs["episode"])
    if not n or any(np.asarray(v).shape != (n,) or np.asarray(v).dtype.kind not in "iu"
                    for v in pairs.values()):
        raise ValueError("invalid numeric pair identities")
    groups = {}
    for index, (episode, round_id) in enumerate(zip(pairs["episode"], pairs["round"], strict=True)):
        groups.setdefault((int(episode), int(round_id)), []).append(index)
    owners, membership = {}, {}
    for (episode, round_id), indices in groups.items():
        agents = sorted(set(pairs["agent_i"][indices]) | set(pairs["agent_j"][indices]))
        actual = list(zip(pairs["agent_i"][indices], pairs["agent_j"][indices]))
        if len(agents) != 4 or actual != list(combinations(agents, 2)):
            raise ValueError("pair table needs exactly six canonical pairs per episode/round")
        if list(pairs["pair"][indices]) != list(range(6)):
            raise ValueError("pair IDs must be canonical ranks 0..5")
        if membership.setdefault(episode, agents) != agents:
            raise ValueError("agent membership changes across rounds")
        source_rows = {}
        for index in indices:
            for suffix in ("i", "j"):
                agent, row = int(pairs[f"agent_{suffix}"][index]), int(pairs[f"row_{suffix}"][index])
                if row < 0 or source_rows.setdefault(agent, row) != row:
                    raise ValueError("inconsistent source row within a round")
                owner = (episode, round_id, agent)
                if owners.setdefault(row, owner) != owner:
                    raise ValueError("source row reused across episode/round/agent identities")
        if len(set(source_rows.values())) != 4:
            raise ValueError("agents share a source row")


def reused_raw_tier(source: Path, output: Path, tier: str,
                    layers: tuple[int, ...]) -> dict | None:
    """Return cache counts if a complete raw tier matches its source."""
    numeric_files = [output / "input" / f"{tier}_layer_{layer}.npz" for layer in layers]
    files = [*numeric_files, output / "evaluator" / f"{tier}.json"]
    if tier == "core":
        files.append(output / "input" / "core_training.json")
    provenance = output / "provenance" / f"{tier}.json"
    if not any(path.exists() for path in [*files, provenance]):
        return None
    if not all(path.exists() for path in [*files, provenance]):
        raise FileExistsError(f"incomplete existing {tier} cache")

    recorded = json.loads(provenance.read_text())
    if recorded["layers"] != list(layers):
        raise ValueError(f"existing {tier} cache uses different layers")
    for path in files:
        if recorded["cache_sha256"][path.name] != sha256(path):
            raise ValueError(f"existing {tier} cache provenance mismatch: {path}")
    for filename, key in (("metadata_gen.json", "source_metadata_sha256"),
                          ("activations_gen.npz", "source_activations_sha256")):
        path = source / filename
        if path.exists() and recorded[key] != sha256(path):
            raise ValueError(f"existing {tier} cache provenance mismatch: {path}")
    with np.load(numeric_files[0], allow_pickle=False) as saved:
        responses = len(saved["response"])
        episodes = len(set(map(int, saved["episode"])))
    return {"tier": tier, "public_responses": responses, "episodes": episodes,
            "layers": list(layers), "reused": True}


def _check_episode_cache(cache, episodes, vector_file):
    if set(episodes) != set(TIERS):
        raise ValueError('NARCBench episode request omitted a benchmark tier')
    for tier, expected in episodes.items():
        routes = json.loads((cache/'evaluator'/f'{tier}.json').read_text())['runs']
        with np.load(vector_file(tier), allow_pickle=False) as saved:
            responses = len(saved['response'])
        if ({row['run_id'] for row in routes} != expected['runs']
                or responses != expected['responses']):
            raise ValueError(f'cached {tier} activations do not match NARCBench episodes')


def raw_cache_for_site(model, site, layers, override=None, *, episodes=None):
    """Resolve a numeric probe cache and reject a mislabeled cache override."""
    default = model.root/('boundary/raw' if site == 'response_start' else 'raw')
    cache = override or default
    for tier in TIERS:
        path = cache/'provenance'/f'{tier}.json'
        if not path.exists():
            raise FileNotFoundError(f'prepare the {site} numeric cache first: {path}')
        recorded = json.loads(path.read_text())
        if not set(layers) <= set(recorded['layers']):
            raise ValueError(f'{cache} lacks requested {tier} layers {sorted(set(layers)-set(recorded["layers"]))}')
        if cache.resolve() != default.resolve():
            expected = default/'provenance'/f'{tier}.json'
            if not expected.exists():
                raise FileNotFoundError(f'cannot verify custom cache site without {expected}')
            reference = json.loads(expected.read_text())
            for field in ('source_metadata_sha256', 'source_activations_sha256'):
                if recorded[field] != reference[field]:
                    raise ValueError(f'custom cache does not match the {site} source: {cache}')
        required = [cache/'evaluator'/f'{tier}.json',
                    *(cache/'input'/f'{tier}_layer_{layer}.npz' for layer in layers)]
        if tier == 'core':
            required.append(cache/'input'/'core_training.json')
        if any(not path.exists() for path in required):
            raise FileNotFoundError(f'incomplete {site} numeric cache: {cache}/{tier}')
    if episodes is not None:
        _check_episode_cache(cache, episodes,
                             lambda tier: cache/'input'/f'{tier}_layer_{layers[0]}.npz')
    return cache


