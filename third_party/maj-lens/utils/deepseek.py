"""Loading DeepSeek-V4-Flash (native FP4 experts, FP8 linears) with its official chat formatter; forward passes."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

from cache import sha256

KERNEL_REPO = "kernels-community/finegrained-fp8"
KERNEL_REVISION = "8c178950e8710e26a2210e4e909cd02dc16c8715"
PUBLIC_RESPONSE_PREFIX = "<｜Assistant｜></think>"


def load_kwargs(model_dir: Path) -> dict:
    """Preserve the checkpoint's FP4 experts and FP8 linears under CPU offload."""
    config = json.loads((Path(model_dir) / "config.json").read_text())
    quantization = config.get("quantization_config", {})
    if (config.get("model_type") != "deepseek_v4" or config.get("expert_dtype") != "fp4"
            or config.get("hc_mult") != 4 or config.get("hidden_size") != 4096
            or config.get("num_hidden_layers") != 43
            or quantization.get("quant_method") != "fp8"
            or quantization.get("scale_fmt") != "ue8m0"
            or quantization.get("dequantize")):
        raise ValueError("DeepSeek V4 native FP4/FP8 checkpoint configuration changed")
    gpu_gib = (torch.cuda.get_device_properties(0).total_memory / 2**30
               if torch.cuda.is_available() else 0)
    # 123GiB is deliberate headroom, not a mistake: V4-Flash is 149GiB against a 140GiB card, so
    # some weights must live on CPU either way. Raising the budget to squeeze more weights on
    # leaves too little for activations and accelerate then thrashes them in and out every
    # forward pass (measured: ~90s/row at 132GiB versus ~3s/row of actual compute).
    # On a card that holds the whole checkpoint (B200, ~179GiB), keep 12GiB for activations and put
    # every weight on the GPU.
    gpu_budget = f"{int(gpu_gib) - 12}GiB" if gpu_gib >= 160 else "123GiB"
    return {"dtype": torch.bfloat16, "device_map": "auto",
            "max_memory": {0: gpu_budget, "cpu": "1800GiB"}, "offload_buffers": True,
            "attn_implementation": "eager", "experts_implementation": "grouped_mm",
            "local_files_only": True}


def pin_kernel() -> None:
    """Use the cached fine-grained FP8 implementation validated for this run."""
    from transformers.integrations import hub_kernels

    current = hub_kernels._HUB_KERNEL_MAPPING.get("finegrained-fp8", {})
    if current.get("repo_id") != KERNEL_REPO:
        raise ValueError("DeepSeek FP8 kernel repository mapping changed")
    hub_kernels._HUB_KERNEL_MAPPING["finegrained-fp8"] = {
        "repo_id": KERNEL_REPO, "revision": KERNEL_REVISION}


def chunked_eager_attention_forward(module, query, key, value, attention_mask,
                                    scaling, dropout=0.0, query_chunk_size=128, **kwargs):
    """Evaluate V4's sink-augmented eager attention in bounded query blocks."""
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import repeat_kv

    keys = repeat_kv(key, module.num_key_value_groups)
    values = repeat_kv(value, module.num_key_value_groups)
    outputs = []
    for start in range(0, query.shape[-2], query_chunk_size):
        end = min(start + query_chunk_size, query.shape[-2])
        logits = torch.matmul(query[:, :, start:end], keys.transpose(2, 3)) * scaling
        if attention_mask is not None:
            logits = logits + attention_mask[..., start:end, :]
        sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, end - start, -1)
        combined = torch.cat([logits, sinks], dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        scores = F.softmax(combined, dim=-1, dtype=combined.dtype)[..., :-1]
        scores = F.dropout(scores, p=dropout, training=module.training).to(values.dtype)
        outputs.append(torch.matmul(scores, values).transpose(1, 2).contiguous())
    return torch.cat(outputs, dim=1), None


def install_chunked_attention() -> None:
    """Patch only this process's DeepSeek eager rule; leave pinned source files intact."""
    from transformers.models.deepseek_v4 import modeling_deepseek_v4

    modeling_deepseek_v4.eager_attention_forward = chunked_eager_attention_forward


def load_model(model_dir: Path):
    """Load the original quantized weights; never silently dequantize them."""
    if not torch.cuda.is_available():
        raise RuntimeError("DeepSeek V4 replay requires a CUDA GPU")
    from transformers import AutoModelForCausalLM

    options = load_kwargs(Path(model_dir))
    pin_kernel()
    install_chunked_attention()
    if torch.cuda.get_device_capability(0)[0] < 10:
        # DeepGEMM's FP4 linear path errors on Hopper; the pinned Triton path
        # reads the same quantized weights without changing the checkpoint.
        os.environ["TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR"] = "1"
    model = AutoModelForCausalLM.from_pretrained(model_dir, **options).eval()
    model.runtime_provenance = runtime_provenance(Path(model_dir))
    return model


def configure_tokenizer(tokenizer, model_dir: Path):
    """Use DeepSeek's released `encode_messages` in chat mode for NARCBench."""
    path = Path(model_dir) / "encoding/encoding_dsv4.py"
    spec = importlib.util.spec_from_file_location("deepseek_v4_official_encoding", path)
    if spec is None or spec.loader is None:
        raise ValueError("DeepSeek official chat encoding is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        rendered = module.encode_messages(messages, thinking_mode="chat")
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered

    tokenizer.apply_chat_template = MethodType(apply_chat_template, tokenizer)
    return tokenizer


def runtime_provenance(model_dir: Path) -> dict:
    options = load_kwargs(model_dir)
    triton_linear = (os.environ.get("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR") == "1"
                     or torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 10)
    return {"checkpoint_precision": "native FP4 experts + FP8 linears",
            "kernel_repo": KERNEL_REPO, "kernel_revision": KERNEL_REVISION,
            "attention_backend": "chunked_eager_128",
            "attention_source_sha256": sha256(Path(__file__)),
            "formatter_sha256": sha256(model_dir / "encoding/encoding_dsv4.py"),
            "disable_deepgemm_linear": triton_linear,
            "load_kwargs": {key: str(value) if key == "dtype" else value
                            for key, value in options.items()}}


def forward(model, ids, all_logits=False):
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids], device=model.device), use_cache=False,
                     **({} if all_logits else {'logits_to_keep': 1}))


def generate(model, ids, eos, n):
    """Greedy decoding of up to n tokens."""
    cache, step, out = None, torch.tensor([ids], device=model.device), []
    with torch.no_grad():
        for _ in range(n):
            o = model(input_ids=step, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache, nxt = o.past_key_values, int(o.logits[0, -1].argmax())
            if nxt == eos:
                break
            out.append(nxt); step = torch.tensor([[nxt]], device=model.device)
    return out
