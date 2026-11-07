"""Build the per-model caches from public inputs (GPU for the replay). No scoring happens here.

  labels        NARCBench's release metadata only (what the report steps need from NARCBench)
  inputs        the pinned NARCBench and jacobian-lens checkouts, NARCBench's dataset, the released J-lens,
                the model weights (config.py, utils/models.py; HF_TOKEN from the environment or .env)
  replay        NARCBench's public transcripts through the model: the residual at the end of every public
                response, every layer (lens.end_of_response)
  prepare       NARCBench-format activations, their public rows with opaque episode/agent integers, labels
  stage         every released layer's rows with those integers (artifacts/cache_expansion)
  primary       the rows, their discussion rounds and the labels (artifacts/maj/2026-09-22-repro)
  stego_rounds  the replay with NARC_PHASE=discussion into artifacts/rounds/<model>: every Stego discussion
                round (NARCBench's release extracts only the final one)

  PYTHONPATH=. python -m utils.build.setup all --model gpt_oss_20b
"""
from __future__ import annotations

import argparse
import json
import secrets
from collections import defaultdict
from pathlib import Path

import numpy as np

from cache import PRIMARY, STAGED, sha256
from config import JLENS_GIT, LENS_REPO, NARCBENCH_DATA, NARCBENCH_GIT
from utils.build.caches import raw_cache_for_site, reused_raw_tier
from utils.models import MODELS, get_model


def _public_phase(tier):
    """NARCBench's phase choice, overridable with NARC_PHASE (see docs: Transfer/Stego ship
    multi-round transcripts but only the final round is extracted by default)."""
    import os
    return os.environ.get("NARC_PHASE") or ("discussion" if tier == "core" else "final")


DATASET = Path("artifacts/narcbench")
TIERS = ("core", "transfer", "stego")


def _ids(rows: list[dict]) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    episode = {run: secrets.randbits(62) for run in sorted({row["run_id"] for row in rows})}
    agent = {}
    for row in rows:
        key = (row["run_id"], row["agent_name"])
        if key not in agent:
            agent[key] = secrets.randbits(62)
    if len(set(episode.values())) != len(episode) or len(set(agent.values())) != len(agent):
        raise ValueError("opaque ID collision")
    return episode, agent


def cache_tier(source: Path, tier: str, output: Path,
               *, layers: tuple[int, ...]) -> dict:
    """Copy only public numeric rows from one released activation tier."""
    if tier not in TIERS or not layers or len(set(layers)) != len(layers):
        raise ValueError("invalid tier or layers")
    input_dir, evaluator = output / "input", output / "evaluator"
    targets = [input_dir / f"{tier}_layer_{layer}.npz" for layer in layers]
    provenance = output / "provenance" / f"{tier}.json"
    existing = reused_raw_tier(source, output, tier, layers)
    if existing is not None:
        return existing
    metadata = json.loads((source / "metadata_gen.json").read_text())
    phase = _public_phase(tier)
    eligible = [(index, row) for index, row in enumerate(metadata)
                if row["phase"] == phase and row["channel"] == "public"]
    if not eligible:
        raise ValueError(f"no public {tier} responses")
    rows = [row for _, row in eligible]
    episode, agent = _ids(rows)
    if any(sum(1 for run, _ in agent if run == name) != 4 for name in episode):
        raise ValueError("each run must have four public agents")
    rows_by_run = defaultdict(list)
    for row in rows:
        rows_by_run[row["run_id"]].append(row)
    modes = {run: {row["mode"] for row in run_rows}
             for run, run_rows in rows_by_run.items()}
    if any(values not in ({"collusion"}, {"control"}) for values in modes.values()):
        raise ValueError("run mode mismatch")
    domains = sorted({row["scenario_domain"] for row in rows}) if tier == "core" else []
    folds = {name: index for index, name in enumerate(domains)}
    response = np.arange(len(rows), dtype=np.int64)
    episode_ids = np.array([episode[row["run_id"]] for row in rows], dtype=np.int64)
    agent_ids = np.array([agent[row["run_id"], row["agent_name"]] for row in rows], dtype=np.int64)
    indices = np.array([index for index, _ in eligible], dtype=np.int64)
    input_dir.mkdir(parents=True, exist_ok=True)
    evaluator.mkdir(parents=True, exist_ok=True)
    with np.load(source / "activations_gen.npz", allow_pickle=False) as saved:
        for layer, path in zip(layers, targets, strict=True):
            vectors = np.asarray(saved[f"layer_{layer}"], dtype=np.float32)
            if len(vectors) < len(metadata) or not np.isfinite(vectors[indices]).all():
                raise ValueError("released activation rows incomplete or nonfinite")
            np.savez_compressed(path, vectors=vectors[indices], episode=episode_ids,
                                agent=agent_ids, response=response)
    if tier == "core":
        agent_targets = []
        for index, row in enumerate(rows):
            if row["mode"] == "collusion":
                agent_targets.append({"response": index,
                                      "target": int(row["role"] == "colluder"),
                                      "fold": folds[row["scenario_domain"]]})
        run_targets = []
        for order, run in enumerate(sorted(episode)):
            domain = rows_by_run[run][0]["scenario_domain"]
            run_targets.append({"episode": episode[run],
                                "target": int("collusion" in modes[run]),
                                "fold": folds[domain], "order": order})
        for run in run_targets:
            labels = [row["target"] for row in agent_targets
                      if episode_ids[row["response"]] == run["episode"]]
            if (run["target"] and (not labels or 2 * sum(labels) != len(labels))
                    or not run["target"] and labels):
                raise ValueError("Core colluder/honest training balance changed")
        (input_dir / "core_training.json").write_text(json.dumps({
            "agent_targets": agent_targets, "run_targets": run_targets}, separators=(",", ":")) + "\n")
    evaluator_runs = []
    for run, run_rows in rows_by_run.items():
        evaluator_runs.append({"episode": episode[run], "run_id": run,
                               "target": int("collusion" in modes[run]),
                               "family": run_rows[0].get("family"),
                               "domain": run_rows[0].get("scenario_domain")})
    (evaluator / f"{tier}.json").write_text(json.dumps(
        {"runs": evaluator_runs}, separators=(",", ":")) + "\n")
    summary = {"tier": tier, "public_responses": len(rows), "episodes": len(episode),
               "layers": list(layers), "source_metadata_sha256": sha256(source / "metadata_gen.json"),
               "source_activations_sha256": sha256(source / "activations_gen.npz")}
    provenance.parent.mkdir(parents=True, exist_ok=True)
    cache_files = [*targets, evaluator / f"{tier}.json"]
    if tier == "core":
        cache_files.append(input_dir / "core_training.json")
    provenance.write_text(json.dumps({**summary, "cache_sha256":
                                      {path.name: sha256(path) for path in cache_files}}, indent=2) + "\n")
    return summary


def narcbench_episodes(model):
    """Ask pinned NARCBench loaders for the public cohort, omitting labels/text."""
    from utils.build.replay import _aligned, _upstream
    upstream = _upstream()
    episodes = {}
    for tier in TIERS:
        rows = _aligned(DATASET, tier, upstream[tier][0], model.cohort)
        episodes[tier] = {'responses': len(rows),
                          'runs': {sample['run_id'] for _, sample in rows}}
    return episodes


def replay_model(model, *, limit=None, placement="auto"):
    from utils.build.replay import cache_public_tiers
    options = {}
    if model.family == 'llama':
        import torch
        from transformers import AutoModelForCausalLM
        # GPU budget: the card's memory less 14 GiB; the rest of the weights offload to CPU.
        gpu_budget = f"{int(torch.cuda.get_device_properties(0).total_memory / 2**30) - 14}GiB"
        placement_kwargs = ({'device_map':'cuda'} if placement == 'cuda' else
                            {'device_map':'auto', 'max_memory':{0:gpu_budget,'cpu':'1800GiB'},
                             'offload_buffers':True})
        def loader(path):
            return AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                dtype=torch.bfloat16, **placement_kwargs)
        options.update(model_loader=loader, runtime_provenance=json.loads(json.dumps({
            'dtype':'bfloat16', **placement_kwargs, 'quantization':'none'})),
            allow_placement_change=placement == 'cuda')
    elif model.family == 'deepseek':
        from utils.deepseek import (
            configure_tokenizer, load_model, runtime_provenance,
        )
        options.update(model_loader=load_model, tokenizer_configurator=configure_tokenizer,
                       stream_reduce='mean',
                       runtime_provenance=runtime_provenance(model.model_dir),
                       allow_placement_change=True)
    import os
    # The discussion-phase replay (NARC_PHASE=discussion) exists only for Stego's extra rounds:
    # Core is already replayed at its discussion rounds and Transfer has a single final response.
    tiers = ("stego",) if os.environ.get("NARC_PHASE") == "discussion" else TIERS
    return cache_public_tiers(DATASET, model.model_dir, model.public_sites, model.identity,
        tiers=tiers, limit=limit, model_short=model.key, dataset_key=model.cohort,
        chat_family=model.family, layers=model.all_layers, end_layers=model.all_layers,
        **options)


def _ensure_public_sites(model) -> None:
    """Resume model replay only when public residual files are absent."""
    ready = all((root / tier / "sites.jsonl").is_file()
                for root in (model.public_sites, model.identity) for tier in TIERS)
    if ready:
        ready = all((model.public_sites / tier / row["sample_file"]).is_file()
                    for tier in TIERS
                    for row in map(json.loads,
                                   (model.public_sites / tier / "sites.jsonl").read_text().splitlines()))
    if not ready:
        replay_model(model)


def prepare_all(model) -> dict:
    """NARCBench-format end-of-response activations (the release's own for GPT-OSS-20B, the replay's otherwise),
    staged as public rows with labels."""
    episodes = narcbench_episodes(model)
    try:
        raw_cache_for_site(model, "end_of_response", model.layers, episodes=episodes)
    except FileNotFoundError:
        source = DATASET / "activations" / model.cohort
        if model.key != "gpt_oss_20b":
            from utils.build.replay import stage_end_activations
            _ensure_public_sites(model)
            source = model.root / "activations" / model.key
            stage_end_activations(DATASET, model.public_sites, model.identity, source,
                                  layers=model.all_layers, dataset_key=model.cohort)
        for tier in TIERS:
            cache_tier(source / tier, tier, model.root / "raw", layers=model.layers)
        raw_cache_for_site(model, "end_of_response", model.layers, episodes=episodes)
    return {tier: {"responses": details["responses"], "episodes": len(details["runs"])}
            for tier, details in episodes.items()}


LENS_ROOT = Path("artifacts/lens_candidate")


def hf_token():
    import os
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env = Path(".env")
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip()
    return None


def fetch_inputs(model) -> None:
    """Idempotent: clones and downloads skip whatever is already present at the pinned revision."""
    import subprocess
    import sys
    from huggingface_hub import hf_hub_download, snapshot_download
    token = hf_token()
    for name, (url, commit) in (("narcbench", NARCBENCH_GIT), ("jacobian-lens", JLENS_GIT)):
        path = Path(name)
        if not (path / ".git").exists():
            subprocess.run(["git", "clone", "-q", url, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "-q", commit], check=True)
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        if head != commit:
            raise RuntimeError(f"{name} at {head}, expected {commit}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "jacobian-lens"], check=True)
    snapshot_download(NARCBENCH_DATA[0], repo_type="dataset", revision=NARCBENCH_DATA[1], local_dir=str(DATASET),
                      allow_patterns=["scenarios/**", f"activations/{model.cohort}/**", "README.md"], token=token)
    relative = model.lens_file.relative_to(LENS_ROOT)
    for name in (relative.name, "config.yaml"):
        hf_hub_download(LENS_REPO[0], str(relative.parent / name), revision=LENS_REPO[1],
                        local_dir=str(LENS_ROOT), token=token)
    snapshot_download(model.model_id, revision=model.revision, local_dir=str(model.model_dir), token=token,
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.jinja"])
    print(f"inputs ready for {model.key}", flush=True)


def fetch_labels() -> None:
    """NARCBench's release metadata (roles and labels of every response), the only public file the report steps read."""
    from huggingface_hub import snapshot_download
    snapshot_download(NARCBENCH_DATA[0], repo_type="dataset", revision=NARCBENCH_DATA[1], local_dir=str(DATASET),
                      allow_patterns=["activations/*/*/metadata_gen.json"])


def run_all(key: str, placement: str) -> None:
    """The cache sequence for one model, each step its own process, stopping at the first failure."""
    import os
    import subprocess
    import sys
    place = ["--placement", placement]
    steps = [(["inputs"], {}), (["replay", *place], {}), (["prepare"], {}), (["stage"], {}), (["primary"], {}),
             (["replay", *place, "--output", f"artifacts/rounds/{key}"], {"NARC_PHASE": "discussion"})]
    for argv, env in steps:
        print(f"=== {key}: {' '.join(argv)}", flush=True)
        subprocess.run([sys.executable, "-u", "-m", "utils.build.setup", argv[0], "--model", key, *argv[1:]],
                       env={**os.environ, **env}, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("all", "inputs", "labels", "replay", "prepare", "stage", "primary"))
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, help="replay only: the first rows, into an isolated --output")
    parser.add_argument("--placement", choices=("auto", "cuda"), default="auto")
    args = parser.parse_args()
    if args.action == "labels":
        return fetch_labels()
    if args.model is None:
        parser.error("--model is required")
    model = get_model(args.model)
    if args.action == "inputs":
        fetch_inputs(model)
    elif args.action == "replay":
        if args.limit is not None and args.output is None:
            parser.error("a limited replay needs an isolated --output")
        if args.output:
            from dataclasses import replace
            model = replace(model, public_sites=args.output / "public_sites", identity=args.output / "identity")
        print(replay_model(model, limit=args.limit, placement=args.placement))
    elif args.action == "prepare":
        print(json.dumps(prepare_all(model), indent=2))
    elif args.action == "stage":
        from utils.build.stage import stage_raw
        stage_raw(model, "end_of_response", tuple(model.all_layers), STAGED / model.key)
    elif args.action == "primary":
        from utils.build.pairs import stage_primary
        stage_primary(model, STAGED / model.key, args.output or PRIMARY / model.key, model.all_layers)
    else:
        run_all(model.key, args.placement)


if __name__ == "__main__":
    main()
