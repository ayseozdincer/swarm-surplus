"""Growable public-site caches for an expanded J-space layer grid.

Staging reads only already replayed public residuals. Each tier/layer cell has
its own source and output hashes, so later grid additions preserve prior work.
Sparse decomposition delegates to the existing TransformerLens-backed cache.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from pathlib import Path

import numpy as np

from utils.io import RAW_FIELDS, numeric
from cache import sha256
from config import TIERS


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as sink:
        np.savez(sink, **arrays)
    temporary.replace(path)


def _paths(model, root: Path) -> tuple[Path, Path, Path]:
    """Every model reads the replay where utils.build.setup writes it.

    This used to special-case gpt_oss_20b (staged under the expansion root) and qwen3_32b (a
    `probe_layers` staging directory). Both were historical layouts from how those two caches were
    first assembled; a rebuild from public inputs produces neither, so the branches only made the
    expansion fail for exactly those two models."""
    return model.public_sites, model.identity, model.root / "raw"


def _source_tier(model, root: Path, tier: str, layers: tuple[int, ...]) -> tuple[list[dict], dict, dict]:
    sites, identity, template = _paths(model, root)
    observed_path, identity_path = sites / tier / "sites.jsonl", identity / tier / "sites.jsonl"
    replay_path = identity / tier / "replay.json"
    released_path = Path("artifacts/narcbench/activations") / model.cohort / tier / "metadata_gen.json"
    if not all(path.exists() for path in (observed_path, identity_path, replay_path, released_path)):
        raise FileNotFoundError(f"complete public replay required for {model.key}/{tier}")
    replay = json.loads(replay_path.read_text())
    if (not set(layers) <= set(replay["layers"]) or not set(layers) <= set(replay.get("end_layers", []))
            or replay["metadata_sha256"] != sha256(released_path)):
        raise ValueError(f"{model.key}/{tier} replay layer/release provenance differs")
    release = json.loads(released_path.read_text())
    phase = "discussion" if tier == "core" else "final"
    eligible = [i for i, row in enumerate(release) if row["phase"] == phase and row["channel"] == "public"]
    observed = [json.loads(line) for line in observed_path.read_text().splitlines()]
    named = [json.loads(line) for line in identity_path.read_text().splitlines()]
    if not observed or len(observed) != len(named) or len(observed) != len(eligible):
        raise ValueError(f"{model.key}/{tier} public replay row count differs")
    numeric_path = template / "input" / f"{tier}_layer_{model.layers[0]}.npz"
    evaluator_path = template / "evaluator" / f"{tier}.json"
    old_provenance_path = template / "provenance" / f"{tier}.json"
    old_provenance = json.loads(old_provenance_path.read_text())
    needed = [numeric_path, evaluator_path]
    if tier == "core":
        needed.append(template / "input" / "core_training.json")
    if any(old_provenance["cache_sha256"].get(path.name) != sha256(path) for path in needed):
        raise ValueError(f"{model.key}/{tier} template numeric cache hash differs")
    template_rows = numeric(numeric_path)
    if (len(template_rows["response"]) != len(eligible)
            or not np.array_equal(template_rows["response"], np.arange(len(eligible)))):
        raise ValueError(f"{model.key}/{tier} template numeric response order differs")
    routes = json.loads(evaluator_path.read_text())["runs"]
    route_by_episode = {int(row["episode"]): row for row in routes}
    if len(route_by_episode) != len(routes):
        raise ValueError("duplicate template episode")
    keys = ("run_id", "agent_name", "phase", "channel", "round")
    agent_ids = {}
    for i, (row, identity_row, release_index) in enumerate(zip(observed, named, eligible, strict=True)):
        expected_key = [release[release_index][key] for key in keys]
        episode = int(template_rows["episode"][i])
        pair = (expected_key[0], expected_key[1])
        if (row["row_index"] != release_index or identity_row["row_index"] != release_index
                or identity_row["row_key"] != expected_key
                or identity_row["full_input_sha256"] != row["full_input_sha256"]
                or Path(row["sample_file"]).name != row["sample_file"]
                or episode not in route_by_episode
                or route_by_episode[episode]["run_id"] != expected_key[0]):
            raise ValueError(f"{model.key}/{tier} source/template public identity differs at {i}")
        agent_id = int(template_rows["agent"][i])
        if pair in agent_ids and agent_ids[pair] != agent_id:
            raise ValueError(f"{model.key}/{tier} template agent ID changes within episode")
        agent_ids[pair] = agent_id
    if len(agent_ids) != 4 * len(route_by_episode):
        raise ValueError(f"{model.key}/{tier} public four-agent episodes differ")
    sources = {str(path): sha256(path) for path in
               (observed_path, identity_path, replay_path, released_path,
                numeric_path, evaluator_path, old_provenance_path)}
    if tier == "core":
        path = template / "input" / "core_training.json"
        sources[str(path)] = sha256(path)
    return observed, template_rows, {"sources": sources, "sites": sites,
                                     "template": template, "evaluator": evaluator_path}


def _copy_shared(tier: str, output: Path, template: Path) -> None:
    paths = [(template / "evaluator" / f"{tier}.json", output / "evaluator" / f"{tier}.json")]
    if tier == "core":
        paths.append((template / "input" / "core_training.json", output / "input" / "core_training.json"))
    for source, target in paths:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != sha256(source):
                raise ValueError(f"staged shared numeric file changed: {target}")
        else:
            shutil.copy2(source, target)


def stage_raw(model, site: str, layers: tuple[int, ...], root: Path) -> dict:
    """Stage every released layer's end-of-response residuals with the opaque row IDs of prepare."""
    if (site != "end_of_response" or not layers
            or len(set(layers)) != len(layers) or not set(layers) <= set(model.all_layers)):
        raise ValueError("invalid public site or released source layers")
    root = Path(root)
    output = root / site / "raw"
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "model": model.model_id, "model_revision": model.revision, "site": site,
        "cells": {}, "tier_sources": {},
        "operation": "public post-block residuals; original numeric episode/agent/response IDs",
    }
    if (manifest["model"] != model.model_id or manifest["model_revision"] != model.revision
            or manifest["site"] != site):
        raise ValueError("existing expanded raw cache model/site differs")
    field = "end_layer_"
    for tier in TIERS:
        observed, old_rows, source = _source_tier(model, root, tier, layers)
        _copy_shared(tier, output, source["template"])
        sources = source["sources"]
        if tier in manifest["tier_sources"] and manifest["tier_sources"][tier] != sources:
            raise ValueError(f"{model.key}/{tier} expanded raw source changed")
        manifest["tier_sources"][tier] = sources
        pending = []
        for layer in layers:
            key = f"{tier}/{layer}"
            path = output / "input" / f"{tier}_layer_{layer}.npz"
            entry = manifest["cells"].get(key)
            if entry is None:
                pending.append(layer)
            elif (entry["source_sha256"] != sources or entry["output_sha256"] != sha256(path)
                  or entry["rows"] != len(observed)):
                raise ValueError(f"expanded raw cell changed: {key}")
        if not pending:
            continue
        collected = {layer: [] for layer in pending}
        for i, row in enumerate(observed):
            path = source["sites"] / tier / row["sample_file"]
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != row["sample_sha256"]:
                raise ValueError(f"expanded public residual changed: {tier}/{i}")
            with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
                for layer in pending:
                    value = saved[f"{field}{layer}"]
                    if value.ndim != 2 or value.shape[0] != 1 or not np.isfinite(value).all():
                        raise ValueError(f"invalid public residual: {tier}/{i}/{layer}")
                    collected[layer].append(value[0].astype(np.float32))
            if (i + 1) % 80 == 0:
                print(f"expanded raw {model.key} {site} {tier}: {i+1}/{len(observed)}", flush=True)
        ids = {name: old_rows[name] for name in RAW_FIELDS[1:]}
        for layer in pending:
            key = f"{tier}/{layer}"
            path = output / "input" / f"{tier}_layer_{layer}.npz"
            _atomic_npz(path, vectors=np.stack(collected[layer]), **ids)
            manifest["cells"][key] = {"source_sha256": sources,
                                      "output_sha256": sha256(path), "rows": len(observed)}
            _atomic_json(manifest_path, manifest)
    return manifest
