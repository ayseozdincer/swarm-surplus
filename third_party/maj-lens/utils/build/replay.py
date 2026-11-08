"""Replay NARCBench's public transcripts through a model with NARCBench's own renderers, recording the residual at
the end of every public response (lens.end_of_response). Row identities go to a separate identity cache."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
import time
from pathlib import Path

import numpy as np

from cache import sha256
from lens import end_of_response


def _public_phase(tier):
    """NARCBench's phase choice, overridable with NARC_PHASE (see docs: Transfer/Stego ship
    multi-round transcripts but only the final round is extracted by default)."""
    import os
    return os.environ.get("NARC_PHASE") or ("discussion" if tier == "core" else "final")


LAYERS = tuple(range(23))
KEYS = ("run_id", "agent_name", "phase", "channel", "round")


def _upstream():
    from utils.narcbench import modules
    extract_core, extract_stego, extract_transfer = modules('extraction', 'extract_core', 'extract_stego', 'extract_transfer')
    return {
        "core": (extract_core.load_runs, extract_core._build_input_text,
                 extract_core._build_gen_text),
        "transfer": (extract_transfer.load_runs, extract_transfer._build_input_text,
                     extract_transfer._build_gen_text),
        "stego": (extract_stego.load_blackjack_runs, extract_stego._build_input_text,
                  extract_stego._build_gen_text),
    }


def _aligned(dataset: Path, tier: str, loader, model_short: str = "gpt_oss_20b") -> list[tuple[int, dict]]:
    metadata = json.loads((dataset / "activations" / model_short / tier /
                           "metadata_gen.json").read_text())
    scenarios = dataset / "scenarios" / model_short / tier
    samples = loader(scenarios)
    if tier == "stego":
        # Matches the documented final-response join in the released scenarios.
        final_text = {run: json.loads((scenarios / run / "results.json").read_text())["final_texts"]
                      for run in {sample["run_id"] for sample in samples}}
        for sample in samples:
            if sample["phase"] == "final" and sample["channel"] == "public":
                sample["response"] = final_text[sample["run_id"]][sample["agent_name"]]
    indexed = {tuple(row[name] for name in KEYS): row for row in samples}
    if len(indexed) != len(samples):
        raise ValueError("duplicate NARCBench scenario row")
    phase = _public_phase(tier)
    aligned = []
    for index, row in enumerate(metadata):
        if row["phase"] != phase or row["channel"] != "public":
            continue
        key = tuple(row[name] for name in KEYS)
        sample = indexed[key]
        if sample["mode"] != row["mode"] or sample["role"] != row["role"]:
            raise ValueError("scenario/release role mismatch")
        aligned.append((index, sample))
    return aligned


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _provenance(model_dir: Path, layers: tuple[int, ...] | None = None) -> dict:
    """Hash model inputs once per invocation; record the actual environment."""
    import torch

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    paths = set(model_dir.glob("*.json")) | set(model_dir.glob("*.jinja"))
    paths.update(model_dir / name for name in set(index["weight_map"].values()))
    versions = {}
    for package in ("torch", "transformers", "accelerate", "safetensors", "numpy", "kernels"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    from concurrent.futures import ThreadPoolExecutor
    ordered_paths = sorted(paths)
    with ThreadPoolExecutor(max_workers=8) as pool:
        model_hashes = dict(zip((path.name for path in ordered_paths), pool.map(sha256, ordered_paths)))
    return {"model_sha256": model_hashes,
            "runtime": {"versions": versions, "cuda": torch.version.cuda},
            "layers": list(LAYERS if layers is None else layers), "dtype": "bfloat16", "saved_dtype": "float16",
            "forward": "model.model; full original input; use_cache=False",
            "source_sha256": {"utils/build/replay.py": sha256(Path(__file__))}}


def cache_public_tiers(dataset: Path, model_dir: Path, public: Path, identity: Path,
                       *, tiers: tuple[str, ...], limit: int | None = None,
                       model_short: str = "gpt_oss_20b", layers: tuple[int, ...] | None = None,
                       end_layers: tuple[int, ...] = (), dataset_key: str | None = None,
                       chat_family: str | None = None, model_loader=None,
                       tokenizer_configurator=None,
                       stream_reduce=None,
                       runtime_provenance=None, allow_placement_change: bool = False) -> list[dict]:
    """Validate checkpoints first, then use one model for all unfinished tiers.

    Each private row record commits its residual hash and identity atomically.
    An interrupted, uncommitted residual is recomputed; committed rows are
    never overwritten. Complete tier manifests remain the staging interface.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    upstream = _upstream()
    if (not tiers or len(set(tiers)) != len(tiers) or any(tier not in upstream for tier in tiers)
            or limit is not None and limit < 1 or public.resolve() == identity.resolve()):
        raise ValueError("invalid replay tier/limit")
    layers = LAYERS if layers is None else layers
    if not layers or not set(end_layers) <= set(layers):
        raise ValueError("invalid replay layer selection")
    dataset_key = dataset_key or model_short
    family = chat_family or ("qwen" if model_short == "qwen3_32b" else "gpt_oss")
    if family not in ("gpt_oss", "qwen", "llama", "deepseek"):
        raise ValueError("unknown replay chat family")
    common = _provenance(model_dir, layers)
    if chat_family is not None or dataset_key != model_short:
        common.update(dataset_key=dataset_key, chat_family=family)
    if stream_reduce is not None:
        common["stream_reduce"] = stream_reduce
    if runtime_provenance is not None:
        common["loader_provenance"] = runtime_provenance
    if end_layers:
        common["end_layers"] = list(end_layers)
        common["model_short"] = model_short
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer_configurator is not None:
        tokenizer = tokenizer_configurator(tokenizer, model_dir)
    plans = []
    for tier in tiers:
        loader, _, render_full = upstream[tier]
        samples = _aligned(dataset, tier, loader, dataset_key)[:limit]
        if not samples:
            # NARCBench's release is not uniform: qwen3_32b/transfer ships only final-phase rows
            # while gpt_oss_20b/transfer ships both. Under a NARC_PHASE override a tier can
            # legitimately have nothing to replay, so skip it rather than abort the other tiers.
            if os.environ.get("NARC_PHASE"):
                print(f"skipping {tier}: no rows at phase={os.environ['NARC_PHASE']}", flush=True)
                continue
            raise ValueError(f"no public replay samples: {tier}")
        rows = []
        for row_index, sample in samples:
            full = tokenizer(render_full(tokenizer, sample), add_special_tokens=False).input_ids
            full_hash = hashlib.sha256(np.asarray(full, dtype=np.int32).tobytes()).hexdigest()
            record = {"public": {"row_index": row_index, "sample_file": f"row_{row_index:05d}.npz",
                                  "full_input_sha256": full_hash},
                      "identity": {"row_index": row_index, "row_key": [sample[name] for name in KEYS],
                                   "full_input_sha256": full_hash}}
            rows.append({"record": record, "full": full, "complete": False})
        provenance = {**common, "tier": tier,
                      "metadata_sha256": sha256(dataset / "activations" / dataset_key / tier /
                                                "metadata_gen.json"),
                      "inputs_sha256": hashlib.sha256(json.dumps(
                          [row["record"] for row in rows], sort_keys=True).encode()).hexdigest()}
        public_dir, identity_dir = public / tier, identity / tier
        provenance_path = identity_dir / "replay.json"
        if provenance_path.exists():
            saved = json.loads(provenance_path.read_text())
            def comparable(value):
                value = {key: item for key, item in value.items() if key != "source_sha256"}
                if allow_placement_change and "loader_provenance" in value:
                    value["loader_provenance"] = {
                        key: item for key, item in value["loader_provenance"].items()
                        if key not in {"device_map", "max_memory", "offload_buffers"}}
                return value
            if comparable(saved) != comparable(provenance):
                raise ValueError(f"replay provenance changed: {tier}")
        elif ((public_dir.exists() and any(public_dir.iterdir())) or
              (identity_dir.exists() and any(
                  path.name not in {"runtime.json", "placement_history.json"} or not path.is_file()
                  for path in identity_dir.iterdir()))):
            # A model is loaded once for all planned tiers, so runtime evidence
            # can precede the first replay record for a later tier. Such evidence
            # contains no reusable activations or row identities.
            raise ValueError(f"existing replay lacks provenance: {tier}")
        for row in rows:
            expected = row["record"]
            checkpoint = identity_dir / f"row_{expected['public']['row_index']:05d}.json"
            if not checkpoint.exists():
                continue
            saved = json.loads(checkpoint.read_text())
            if (saved.get("identity") != expected["identity"] or
                    {key: value for key, value in saved.get("public", {}).items()
                     if key != "sample_sha256"} != expected["public"]):
                raise ValueError(f"replay input/identity changed: {tier}/{checkpoint.name}")
            residual = public_dir / expected["public"]["sample_file"]
            if not residual.is_file() or sha256(residual) != saved["public"].get("sample_sha256"):
                raise ValueError(f"replay residual changed: {tier}/{residual.name}")
            row.update(record=saved, complete=True)
        plans.append((tier, rows, provenance))

    model, handles, captures = None, [], {}
    summaries = []
    try:
        for tier, rows, provenance in plans:
            public_dir, identity_dir = public / tier, identity / tier
            public_dir.mkdir(parents=True, exist_ok=True)
            identity_dir.mkdir(parents=True, exist_ok=True)
            provenance_path = identity_dir / "replay.json"
            if not provenance_path.exists():
                _atomic_text(provenance_path, json.dumps(provenance, indent=2) + "\n")
            reused = sum(row["complete"] for row in rows)
            for count, row in enumerate(rows, 1):
                if row["complete"]:
                    continue
                if model is None:
                    model = (model_loader(model_dir) if model_loader is not None else
                             AutoModelForCausalLM.from_pretrained(
                                 model_dir, dtype=torch.bfloat16, device_map="cuda", local_files_only=True)).eval()
                    actual = {'device':str(model.device),
                              'device_map':{str(k):str(v) for k,v in getattr(model,'hf_device_map',{}).items()},
                              'model_class':type(model).__name__,
                              'runtime_provenance':getattr(model,'runtime_provenance',None)}
                    for planned_tier, planned_rows, requested in plans:
                        directory = identity / planned_tier
                        directory.mkdir(parents=True, exist_ok=True)
                        runtime_path = directory/'runtime.json'
                        history_path = directory/'placement_history.json'
                        history = json.loads(history_path.read_text()) if history_path.exists() else []
                        original_path = directory/'replay.json'
                        original = json.loads(original_path.read_text()) if original_path.exists() else {}
                        previous = (history[-1]['current'] if history else original.get('loader_provenance'))
                        previous_actual = (history[-1]['actual_runtime'] if history else
                            json.loads(runtime_path.read_text()) if runtime_path.exists() else None)
                        from datetime import datetime, timezone
                        history.append({'recorded_at':datetime.now(timezone.utc).isoformat(),
                            'previous':previous, 'current':requested.get('loader_provenance'),
                            'previous_actual_runtime':previous_actual, 'actual_runtime':actual,
                            'source_sha256':requested['source_sha256'],
                            'placement_change_allowed':allow_placement_change,
                            'reused_row_indices':[r['record']['public']['row_index'] for r in planned_rows if r['complete']],
                            'pending_row_indices':[r['record']['public']['row_index'] for r in planned_rows if not r['complete']]})
                        _atomic_text(history_path, json.dumps(history, indent=2)+'\n')
                        if not runtime_path.exists():
                            _atomic_text(runtime_path, json.dumps(actual, indent=2)+'\n')
                    for layer in end_layers:
                        def hook(_module, _inputs, output, layer=layer):
                            hidden = output[0] if isinstance(output, tuple) else output
                            captures[f"end_{layer}"] = end_of_response(hidden, stream_reduce).to(
                                dtype=torch.float16, device="cpu").numpy()
                        handles.append(model.model.layers[layer].register_forward_hook(hook))
                record = row["record"]
                captures.clear()
                forward_started = time.perf_counter()
                with torch.no_grad():
                    model.model(torch.tensor([row["full"]], dtype=torch.long, device=model.device),
                                use_cache=False)
                forward_seconds = time.perf_counter() - forward_started
                if (set(captures) != {f"end_{layer}" for layer in end_layers}
                        or any(not np.isfinite(x).all() for x in captures.values())):
                    raise ValueError("missing or nonfinite post-block layer capture")
                path = public_dir / record["public"]["sample_file"]
                temporary = path.with_suffix(".npz.tmp")
                with temporary.open("wb") as sink:
                    np.savez_compressed(sink, **{f"end_layer_{layer}": captures[f"end_{layer}"][None, :]
                                           for layer in end_layers})
                temporary.replace(path)
                record["public"]["sample_sha256"] = sha256(path)
                checkpoint = identity_dir / f"row_{record['public']['row_index']:05d}.json"
                _atomic_text(checkpoint, json.dumps(record) + "\n")
                row["complete"] = True
                print(f"{tier} public response {count}/{len(rows)} "
                      f"tokens={len(row['full'])} forward_s={forward_seconds:.3f}", flush=True)
            for directory, field in ((public_dir, "public"), (identity_dir, "identity")):
                _atomic_text(directory / "sites.jsonl", "".join(
                    json.dumps(row["record"][field]) + "\n" for row in rows))
            summaries.append({"tier": tier, "public_responses": len(rows), "layers": len(layers),
                              "reused_responses": reused, "replayed_responses": len(rows) - reused})
    finally:
        for handle in handles:
            handle.remove()
    return summaries


def stage_end_activations(dataset, public, identity, output, *, layers, tiers=("core", "transfer", "stego"), dataset_key="qwen3_32b"):
    """Make NARCBench-format arrays from the end sites of the same replay."""
    keys = ('run_id', 'agent_name', 'phase', 'channel', 'round')
    for tier in tiers:
        source = dataset/'activations'/dataset_key/tier/'metadata_gen.json'
        observed_path, identity_path = public/tier/'sites.jsonl', identity/tier/'sites.jsonl'
        inputs = {str(p): sha256(p) for p in (source, observed_path, identity_path)}
        target = output/tier
        manifest_path = target/'manifest.json'
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text())
            if (saved['inputs'] != inputs or saved['layers'] != list(layers)
                    or any(sha256(target/n) != h for n,h in saved['outputs'].items())):
                raise ValueError('end-response cache provenance changed')
            continue
        if target.exists():
            raise FileExistsError(f'incomplete end-response cache: {target}')
        metadata = json.loads(source.read_text())
        observed = [json.loads(line) for line in observed_path.read_text().splitlines()]
        identities = {r['row_index']:r for r in map(json.loads, identity_path.read_text().splitlines())}
        phase = _public_phase(tier)
        expected = {i for i,r in enumerate(metadata) if r['phase']==phase and r['channel']=='public'}
        if (len(observed)!=len(expected) or {r['row_index'] for r in observed}!=expected
                or set(identities)!=expected):
            raise ValueError('replay does not cover the original public response cohort')
        arrays = {layer:[] for layer in layers}
        ordered = sorted(observed, key=lambda row:row['row_index'])
        for row in ordered:
            index = row['row_index']
            other = identities[index]
            if (other['row_key'] != [metadata[index][key] for key in keys]
                    or not row.get('full_input_sha256')
                    or other.get('full_input_sha256') != row['full_input_sha256']):
                raise ValueError('end-response identity mismatch')
            name = row['sample_file']
            if Path(name).name != name or sha256(public/tier/name) != row['sample_sha256']:
                raise ValueError('replayed end-response file changed')
            with np.load(public/tier/name, allow_pickle=False) as saved:
                for layer in layers:
                    values = saved[f'end_layer_{layer}']
                    if values.ndim != 2 or len(values) != 1 or not np.isfinite(values).all():
                        raise ValueError('invalid end-response activation')
                    arrays[layer].append(values[0])
        target.mkdir(parents=True)
        (target/'metadata_gen.json').write_text(json.dumps([metadata[r['row_index']] for r in ordered])+'\n')
        np.savez_compressed(target/'activations_gen.npz',
                            **{f'layer_{layer}':np.stack(values) for layer,values in arrays.items()})
        manifest_path.write_text(json.dumps({'inputs':inputs,'layers':list(layers),
            'release_row_indices':[r['row_index'] for r in ordered],
            'outputs':{n:sha256(target/n) for n in ('metadata_gen.json','activations_gen.npz')}},indent=2)+'\n')
