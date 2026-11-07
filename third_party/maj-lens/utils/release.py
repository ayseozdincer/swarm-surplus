"""Released GPT-OSS Jacobians and output head for J-space projection."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from utils.build.caches import sha256

MODEL = Path("artifacts/gpt_oss_20b")
LENS = Path("artifacts/lens_candidate/gpt-oss-20b/jlens/Salesforce-wikitext/gpt-oss-20b_jacobian_lens.pt")
LENS_CONFIG = LENS.with_name("config.yaml")
MODEL_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
# NOTE 2026-09-22: the neuronpedia lens `config.yaml` files were backfilled upstream on
# 2026-09-20 with an `environment:` block ("the fit did not record its environment").
# The lens weights are bit-identical to the original pins; only this metadata changed, and
# every field validate_release() actually checks (hf_model_name, dataset, target_layer,
# prompts_fitted) is unchanged. Only the .yaml digests were re-pinned.
HASHES = {
    LENS: "2a8abf062eae4b6d39618b1237ad6945db437affd8c2122a09689114571553be",
    LENS_CONFIG: "1c7bc7d7923fbaed2d4c66e43e0af9abcff3653de3d015ec3f6469efcb62e329",
    MODEL / "config.json": "3a2a26ded679375b7928ddeca59764df7cea83220c1961035f6d6e232659e9ce",
    MODEL / "model.safetensors.index.json": "0e085b977c4c9942f85938828e8c989ed7d5cdabf852e4da6a67c116cd502cd1",
    MODEL / "tokenizer.json": "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3",
    MODEL / "model-00000-of-00002.safetensors": "16d0f997dcfc4462089d536bffe51b4bcea2f872f5c430be09ef8ed392312427",
    MODEL / "model-00001-of-00002.safetensors": "4fbe328ab445455d6f58dc73852b85873bd626986310abd91cd4d2ce3245eaea",
    MODEL / "model-00002-of-00002.safetensors": "a18106b209e9ab35c3406db4f6f12a927364a058b21e9d1373d682e20674b303",
}


def load_released_lens(path: Path):
    """Read final means or finalize a released running sum without fitting.

    The sum/count conversion is exactly jlens.fitting.fit's return operation.
    A checkpoint's successful-prompt count, not its resume index, is the divisor.
    """
    from jlens import JacobianLens
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "J" in state:
        lens = JacobianLens(state["J"], n_prompts=state["n_prompts"], d_model=state["d_model"])
        lens.published_provenance = state.get("provenance")
        return lens, "mean"
    count = state.get("n_done")
    if not isinstance(count, int) or count <= 0:
        raise ValueError("invalid released lens checkpoint count")
    sums = state.get("jacobian_sum", {})
    if not sums or sorted(sums) != state.get("source_layers"):
        raise ValueError("released lens checkpoint layers mismatch")
    width = next(iter(sums.values())).shape[0]
    if any(value.shape != (width, width) for value in sums.values()):
        raise ValueError("released lens checkpoint matrix dimensions differ")
    return JacobianLens({layer: value.float() / count for layer, value in sums.items()},
                        n_prompts=count, d_model=width), "running_sum"



def validate_release(root: Path, *, all_model_shards: bool = False,
                     head_shard: bool = False) -> dict:
    """Validate the pinned release and fit metadata without loading model weights.

    Full model-shard hashing is available for a complete source audit. The
    default is a fast metadata check; projection additionally hashes the
    output-head shard. Replay provenance records all shard hashes.
    """
    import yaml
    from transformers import AutoConfig, AutoTokenizer

    skipped = []
    for relative, expected in HASHES.items():
        if (relative.name.startswith("model-0000") and not all_model_shards
                and not (head_shard and relative.name == "model-00002-of-00002.safetensors")):
            skipped.append(str(relative))
            continue
        if sha256(root / relative) != expected:
            raise ValueError(f"released source hash mismatch: {relative}")
    fit = yaml.safe_load((root / LENS_CONFIG).read_text())
    if (fit.get("hf_model_name") != "openai/gpt-oss-20b"
            or fit.get("dataset", {}).get("name") != "Salesforce/wikitext"
            or fit.get("fit", {}).get("target_layer") is not None
            or fit.get("results", {}).get("prompts_fitted") != 457):
        raise ValueError("released lens fit metadata changed")
    config = AutoConfig.from_pretrained(root / MODEL, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(root / MODEL, local_files_only=True)
    if (config.num_hidden_layers != 24 or config.hidden_size != 2880
            or len(tokenizer) != 200019
            or tokenizer.convert_tokens_to_ids("<|message|>") != 200008):
        raise ValueError("GPT-OSS model/tokenizer dimensions or delimiter changed")
    return {"model_revision": MODEL_REVISION, "lens_sha256": HASHES[LENS],
            "tokenizer_sha256": HASHES[MODEL / "tokenizer.json"],
            "model_config_sha256": HASHES[MODEL / "config.json"],
            "model_index_sha256": HASHES[MODEL / "model.safetensors.index.json"],
            "model_shard_sha256": {path.name: digest for path, digest in HASHES.items()
                                    if path.name.startswith("model-0000")},
            "layers": list(range(23)), "d_model": 2880, "vocab_size": len(tokenizer),
            "full_shards_verified_now": all_model_shards,
            "head_shard_verified_now": all_model_shards or head_shard,
            "shards_not_rehashed_now": skipped}


class ReleasedReadout:
    """Hold selected released Jacobians and the matched GPT-OSS output head."""

    def __init__(self, root: Path, layers: set[int], *, device: str = "cuda",
                 model_dir: Path | None = None, lens_file: Path | None = None,
                 model_revision: str | None = None, model_id: str | None = None) -> None:
        from safetensors import safe_open

        head_key = "lm_head.weight"
        if model_dir is None:
            model_dir, lens_file = root / MODEL, root / LENS
            self.source = validate_release(root, head_shard=True)
            expected_prompts = 457
        else:
            import yaml
            from transformers import AutoConfig, AutoTokenizer
            if lens_file is None or not model_revision or not model_id:
                raise ValueError("explicit model/lens paths and model revision/ID required")
            config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            fit_path = lens_file.with_name("config.yaml")
            fit = yaml.safe_load(fit_path.read_text())
            deepseek = model_id == "deepseek-ai/DeepSeek-V4-Flash"
            expected_corpus = "NeelNanda/pile-10k" if deepseek else "Salesforce/wikitext"
            expected_target = 41 if deepseek else None
            if (fit["hf_model_name"] != model_id
                    or fit["dataset"]["name"] != expected_corpus
                    or fit["fit"]["target_layer"] != expected_target):
                raise ValueError("released lens source/model mismatch")
            if deepseek:
                if (config.num_hidden_layers != 43 or config.hidden_size != 4096
                        or fit["lens"]["capture_point"] != "block_output"
                        or fit["lens"]["stream_reduce"] != "mean"):
                    raise ValueError("DeepSeek released lens requires mean post-block streams")
                head_key = "head.weight"
            expected_prompts = fit.get("results", {}).get("prompts_fitted", fit["fit"].get("n_prompts"))
            index_path = model_dir / "model.safetensors.index.json"
            head_name = json.loads(index_path.read_text())["weight_map"][head_key]
            self.source = {
                "model_id": model_id, "model_revision": model_revision,
                "lens_sha256": sha256(lens_file), "lens_config_sha256": sha256(fit_path),
                "model_config_sha256": sha256(model_dir / "config.json"),
                "model_index_sha256": sha256(index_path),
                "tokenizer_sha256": sha256(model_dir / "tokenizer.json"),
                "model_shard_sha256": {head_name: sha256(model_dir / head_name)},
                "layers": list(range(config.num_hidden_layers - 1)),
                "d_model": config.hidden_size, "vocab_size": len(tokenizer),
                "fit_prompts": expected_prompts,
                "target_layer": expected_target,
                "stream_reduce": "mean" if deepseek else None,
                "head_tensor_key": head_key,
            }
        if not layers or not layers <= set(self.source["layers"]):
            raise ValueError("requested layer outside released lens")
        self.vocab_size = self.source["vocab_size"]
        self.device = torch.device(device)
        lens, storage_format = load_released_lens(lens_file)
        if getattr(lens, "published_provenance", None):
            self.source["published_provenance"] = lens.published_provenance
            if model_id == "deepseek-ai/DeepSeek-V4-Flash":
                expected = {"model_id": model_id, "dataset_id": "NeelNanda/pile-10k",
                            "target_layer": 41, "capture_point": "block_output",
                            "stream_reduce": "mean"}
                if any(lens.published_provenance.get(key) != value for key, value in expected.items()):
                    raise ValueError("DeepSeek lens capture provenance changed")
        elif model_id == "deepseek-ai/DeepSeek-V4-Flash":
            raise ValueError("DeepSeek lens lacks stream/capture provenance")
        if storage_format == "running_sum":
            self.source.update({"lens_storage_format": storage_format,
                                "fit_prompts_config": expected_prompts,
                                "fit_prompts": lens.n_prompts,
                                "fit_count_matches_config": lens.n_prompts == expected_prompts,
                                "conversion": "jacobian_sum / n_done; jlens.fitting.fit"})
        if (lens.source_layers != self.source["layers"] or lens.d_model != self.source["d_model"]
                or storage_format == "mean" and lens.n_prompts != expected_prompts):
            raise ValueError("released J-lens matrix metadata mismatch")
        self.jacobians = {layer: lens.jacobians[layer].to(self.device)
                          for layer in sorted(layers)}
        if any(not torch.isfinite(matrix).all() for matrix in self.jacobians.values()):
            raise ValueError("nonfinite released J-lens matrix")
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        head_shard = model_dir / index[head_key]
        with safe_open(head_shard, framework="pt", device="cpu") as source:
            self.head = source.get_tensor(head_key).to(self.device)
        if (self.head.ndim != 2 or self.head.shape[1] != self.source["d_model"]
                or self.head.shape[0] < self.vocab_size):
            raise ValueError("released output head shape mismatch")
