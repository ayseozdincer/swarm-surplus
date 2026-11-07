"""PRIMARY: per model, the end-of-response activations with their discussion rounds and labels."""
from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path
import numpy as np

from cache import sha256
from utils.build.caches import validate_pairs
from utils.io import numeric, save_json

TIERS = ("core", "transfer", "stego")
SITES = ("end_of_response",)


def aligned_rounds(rows: dict, identities: list[dict], tier: str) -> np.ndarray:
    """Validate bijections using names transiently; return numeric rounds only."""
    n = len(rows["response"])
    if len(identities) != n or not np.array_equal(rows["response"], np.arange(n)):
        raise ValueError("identity/response row alignment differs")
    expected = "discussion" if tier == "core" else "final"
    episodes, reverse_episodes, agents, reverse_agents = {}, {}, {}, {}
    rounds = []
    for episode, agent, identity in zip(rows["episode"], rows["agent"], identities, strict=True):
        run, name, phase, channel, round_id = identity["row_key"]
        if phase != expected or channel != "public":
            raise ValueError("pair cache requires the declared public phase")
        e, a = int(episode), int(agent)
        if episodes.setdefault(e, run) != run or reverse_episodes.setdefault(run, e) != e:
            raise ValueError("episode identity is not one-to-one")
        if (agents.setdefault((e, a), name) != name
                or reverse_agents.setdefault((e, name), a) != a):
            raise ValueError("agent identity is not one-to-one within episode")
        if not isinstance(round_id, int):
            raise ValueError("round identity must be an integer")
        rounds.append(round_id)
    return np.asarray(rounds, dtype=np.int64)


def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as sink:
        np.savez(sink, **arrays)
    temporary.replace(path)


def _link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != sha256(source):
            raise ValueError(f"existing linked input changed: {target}")
    else:
        target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()))


def stage_training(model, source_root, output):
    """Link the labels: evaluator routes per tier, Core training targets for the probe."""
    for site in SITES:
        raw = source_root / site / "raw"
        for tier in TIERS:
            _link(raw / "evaluator" / f"{tier}.json", output / site / "evaluator" / f"{tier}.json")
        _link(raw / "input" / "core_training.json", output / site / "training" / "individual.json")


def stage_primary(model, source_root: Path, output: Path, layers, *, sites=SITES):
    """PRIMARY for one model: the staged raw activations (linked), the round table (pairs/) and the labels, with a
    manifest binding each raw file's hash."""
    layers = tuple(layers)
    if not layers or not set(layers) <= set(model.all_layers):
        raise ValueError("invalid released layer set")
    prepare_pairs(model, source_root, output)
    stage_training(model, source_root, output)
    for site in sites:
        path = output / site / "representations.json"
        manifest = json.loads(path.read_text()) if path.exists() else {"model": model.model_id,
                                                                         "revision": model.revision, "site": site, "cells": {}}
        for layer in layers:
            for tier in TIERS:
                key = f"{tier}/{layer}"
                raw = source_root / site / "raw"
                source = raw / "input" / f"{tier}_layer_{layer}.npz"
                if sha256(source) != json.loads((raw / "manifest.json").read_text())["cells"][key]["output_sha256"]:
                    raise ValueError("raw representation source changed")
                target = output / site / "raw" / "input" / source.name
                _link(source, target)
                pair_path = output / site / "pairs" / f"{tier}.npz"
                manifest["cells"][key] = {"raw_sha256": sha256(source), "pair_sha256": sha256(pair_path),
                                          "output_sha256": {str(target): sha256(target)}}
        save_json(path, manifest)
    return output


def pair_rows(episode, round, agent) -> dict[str, np.ndarray]:
    """Index six canonical pairs per complete four-agent episode/round.

    Inputs are numeric identity arrays, never roles or labels. The same four
    agent IDs must be present in every observed round of an episode. Row order
    does not affect canonical output order: episode, round, sorted agent pair.
    """
    columns = [np.asarray(value) for value in (episode, round, agent)]
    if any(value.ndim != 1 or value.dtype.kind not in 'iu' for value in columns):
        raise ValueError('episode, round and agent must be integer identity arrays')
    if len({len(value) for value in columns}) != 1:
        raise ValueError('identity arrays have different lengths')
    groups = {}
    for index, key in enumerate(zip(*columns)):
        episode_id, round_id, agent_id = map(int, key)
        group = groups.setdefault((episode_id, round_id), {})
        if agent_id in group:
            raise ValueError('duplicate agent within episode/round')
        group[agent_id] = index
    membership = {}
    result = {name: [] for name in ('row_i', 'row_j', 'episode', 'round', 'agent_i', 'agent_j', 'pair')}
    for (episode_id, round_id), group in sorted(groups.items()):
        agents = tuple(sorted(group))
        if len(agents) != 4:
            raise ValueError('every episode/round must contain exactly four agents')
        if membership.setdefault(episode_id, agents) != agents:
            raise ValueError('agent membership changes across rounds')
        for pair, (agent_i, agent_j) in enumerate(combinations(agents, 2)):
            values = (group[agent_i], group[agent_j], episode_id, round_id, agent_i, agent_j, pair)
            for name, value in zip(result, values):
                result[name].append(value)
    return {name: np.asarray(value, dtype=np.int64) for name, value in result.items()}


def load_pairs(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        pairs = {key: saved[key] for key in saved.files}
    validate_pairs(pairs)
    return pairs


def prepare_pairs(model, source_root: Path, output: Path) -> dict:
    """Trust existing source hashes, validate alignment and freeze six-pair IDs."""
    manifests = {}
    for site in SITES:
        raw = source_root / site / "raw"
        manifest = json.loads((raw / "manifest.json").read_text())
        if manifest["model"] != model.model_id or manifest["model_revision"] != model.revision:
            raise ValueError("raw model/checkpoint mismatch")
        site_record = {"model": model.model_id, "checkpoint_revision": model.revision,
                       "site": site, "lens_hash": sha256(model.lens_file), "k": 25,
                       "tiers": {}}
        for tier in TIERS:
            layer = model.all_layers[0]
            path = raw / "input" / f"{tier}_layer_{layer}.npz"
            expected = manifest["cells"][f"{tier}/{layer}"]["output_sha256"]
            if sha256(path) != expected:
                raise ValueError("raw row source hash mismatch")
            identity_paths = [Path(p) for p in manifest["tier_sources"][tier]
                              if "/identity/" in p and p.endswith("sites.jsonl")]
            if len(identity_paths) != 1:
                raise ValueError("unique trusted identity source required")
            identity_path = identity_paths[0]
            identity_hash = sha256(identity_path)
            if identity_hash != manifest["tier_sources"][tier][str(identity_path)]:
                raise ValueError("trusted identity hash changed")
            rows = numeric(path)
            identities = [json.loads(line) for line in identity_path.read_text().splitlines()]
            rounds = aligned_rounds(rows, identities, tier)
            pairs = pair_rows(rows["episode"], rounds, rows["agent"])
            validate_pairs(pairs)
            target = output / site / "pairs" / f"{tier}.npz"
            if target.exists():
                prior = load_pairs(target)
                if any(not np.array_equal(prior[k], pairs[k]) for k in pairs):
                    raise ValueError("frozen pair identities changed")
            else:
                save_npz(target, **pairs)
            site_record["tiers"][tier] = {
                "rows": len(rows["response"]), "pair_rows": len(pairs["episode"]),
                "episodes": len(np.unique(pairs["episode"])), "excluded_rows": 0,
                "raw_source": str(path), "raw_sha256": expected,
                "identity_source": str(identity_path), "identity_sha256": identity_hash,
                "pair_sha256": sha256(target),
            }
        target_manifest = output / site / "pairs" / "manifest.json"
        if target_manifest.exists() and json.loads(target_manifest.read_text()) != site_record:
            raise ValueError("frozen pair provenance changed")
        save_json(target_manifest, site_record)
        manifests[site] = site_record
    return manifests
