#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantize nvidia/Cosmos3-Nano to a ModelOpt NVFP4 Hugging Face checkpoint.

Cosmos3 is not loaded through the stock diffusers pipeline in vLLM-Omni, so
this script uses the native Cosmos3 pipeline and transformer implementation,
then exports a diffusers-style checkpoint whose ``transformer/config.json``
auto-selects vLLM's ``modelopt_fp4`` runtime path.

Example:
    python examples/quantization/quantize_cosmos3_nano_modelopt_nvfp4.py \\
        --model nvidia/Cosmos3-Nano \\
        --output ./Cosmos3-Nano-ModelOpt-NVFP4 \\
        --height 720 --width 1280 --num-frames 25 \\
        --calib-steps 4 --calib-size 4 \\
        --overwrite
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.serialization import save_torch_state_dict
from transformers import AutoTokenizer
from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
    COSMOS3_DEFAULT_MAX_SEQUENCE_LENGTH,
    COSMOS3_SYSTEM_PROMPT,
    Cosmos3OmniDiffusersPipeline,
)
from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
    Cosmos3VFMTransformer,
    _as_bool,
    _tf_config_get,
)

DEFAULT_PROMPTS = [
    "A robot arm is cleaning a plate in the kitchen.",
    "A red sports car drives through a rainy city street at night, cinematic reflections.",
    "A golden retriever runs across a sunny park while leaves move in the wind.",
    "A close-up macro shot of a flower opening at sunrise with soft natural light.",
]

MODELOPT_SCALE_SUFFIXES = (
    ".input_scale",
    ".output_scale",
    ".weight_scale",
    ".weight_scale_2",
    ".weight_scale_inv",
)
COSMOS3_VAE_SCALE_FACTOR_TEMPORAL = 4
COSMOS3_VAE_SCALE_FACTOR_SPATIAL = 16


@dataclass
class _Cosmos3TransformContext:
    transformer: Cosmos3VFMTransformer
    tokenizer: Any
    dtype: torch.dtype
    vae_scale_factor_temporal: int = COSMOS3_VAE_SCALE_FACTOR_TEMPORAL
    vae_scale_factor_spatial: int = COSMOS3_VAE_SCALE_FACTOR_SPATIAL


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="nvidia/Cosmos3-Nano", help="Input Cosmos3-Nano checkpoint or HF id.")
    parser.add_argument("--output", required=True, help="Output directory for the ModelOpt NVFP4 checkpoint.")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--calib-steps", type=int, default=4, help="Transformer forwards per calibration prompt.")
    parser.add_argument("--calib-size", type=int, default=4, help="How many prompts to use for calibration.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--prompt", action="append", default=[], help="Calibration prompt. Repeat for more prompts.")
    parser.add_argument(
        "--mlp-only",
        action="store_true",
        help="Use ModelOpt NVFP4_MLP_ONLY_CFG instead of all transformer linear layers.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum safetensors shard size for the exported transformer.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser


def _require_modelopt() -> Any:
    try:
        import modelopt.torch.quantization as mtq
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "NVIDIA ModelOpt is not installed. Install with:\n"
            "  pip install 'nvidia-modelopt[all]'\n"
            f"Original error: {exc}"
        ) from exc
    return mtq


def _get_free_tcp_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ensure_single_process_parallel_state() -> None:
    if model_parallel_is_initialized():
        return
    if not torch.distributed.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{_get_free_tcp_port()}",
        )
    initialize_model_parallel(
        data_parallel_size=1,
        cfg_parallel_size=1,
        sequence_parallel_size=1,
        ulysses_degree=1,
        ring_degree=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )


def _select_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _resolve_model_path(model: str) -> Path:
    path = Path(model).expanduser()
    if path.exists():
        return path.resolve()
    return Path(snapshot_download(model)).resolve()


def _prepare_output(output: str, overwrite: bool) -> Path:
    output_dir = Path(output).expanduser().resolve()
    if output_dir.exists():
        if not overwrite:
            raise SystemExit(f"Output directory already exists: {output_dir}\nPass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    return output_dir


def _build_prompts(args: argparse.Namespace) -> list[str]:
    prompts = args.prompt or DEFAULT_PROMPTS
    if args.calib_size <= 0:
        raise SystemExit("--calib-size must be positive.")
    if len(prompts) < args.calib_size:
        repeats = (args.calib_size + len(prompts) - 1) // len(prompts)
        prompts = (prompts * repeats)[: args.calib_size]
    return prompts[: args.calib_size]


def _tokenize_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    device: torch.device,
    max_sequence_length: int = COSMOS3_DEFAULT_MAX_SEQUENCE_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    conversations = [
        {"role": "system", "content": COSMOS3_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    token_ids = Cosmos3OmniDiffusersPipeline._normalize_token_ids(
        tokenizer.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True)
    )
    token_ids = token_ids[:max_sequence_length]
    token_ids.append(tokenizer.eos_token_id)
    token_ids.append(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def _load_cosmos3_transformer(model_path: Path, dtype: torch.dtype) -> _Cosmos3TransformContext:
    with (model_path / "transformer" / "config.json").open(encoding="utf-8") as f:
        transformer_config_dict = json.load(f)

    od_config = OmniDiffusionConfig(
        model=str(model_path),
        model_class_name="Cosmos3OmniDiffusersPipeline",
        dtype=dtype,
        tf_model_config=TransformerConfig.from_dict(transformer_config_dict),
        model_config={"guardrails": False},
        custom_pipeline_args={"guardrails": False},
    )
    sound_gen = _as_bool(_tf_config_get(transformer_config_dict, "sound_gen", False))
    sound_dim = int(_tf_config_get(transformer_config_dict, "sound_dim", 64)) if sound_gen else None
    sound_latent_fps = float(_tf_config_get(transformer_config_dict, "sound_latent_fps", 25.0)) if sound_gen else None

    transformer = Cosmos3VFMTransformer(
        od_config=od_config,
        temporal_compression_factor=COSMOS3_VAE_SCALE_FACTOR_TEMPORAL,
        sound_gen=sound_gen,
        sound_dim=sound_dim,
        sound_latent_fps=sound_latent_fps,
    )

    weight_files = sorted((model_path / "transformer").glob("*.safetensors"))
    if not weight_files:
        raise SystemExit(f"No transformer safetensors found under {model_path / 'transformer'}")
    weights = safetensors_weights_iterator([str(path) for path in weight_files], use_tqdm_on_load=True)
    _load_transformer_weights(transformer, (("transformer." + name, tensor) for name, tensor in weights))
    transformer.to(device="cuda", dtype=dtype)
    transformer.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        subfolder="text_tokenizer",
        local_files_only=True,
    )
    return _Cosmos3TransformContext(transformer=transformer, tokenizer=tokenizer, dtype=dtype)


def _load_transformer_weights(
    transformer: Cosmos3VFMTransformer,
    weights: Any,
) -> set[str]:
    class _Wrapper(torch.nn.Module):
        def __init__(self, wrapped: Cosmos3VFMTransformer) -> None:
            super().__init__()
            self.transformer = wrapped

    wrapper = _Wrapper(transformer)
    state = wrapper.state_dict()
    allowed = set(state.keys())
    tp_aware = {name for name, param in wrapper.named_parameters() if hasattr(param, "weight_loader")}

    def _remapped_weights():
        total = kept = 0
        for name, tensor in weights:
            total += 1
            remapped = Cosmos3OmniDiffusersPipeline._remap_ckpt_key(name)
            if remapped is not None and (remapped in allowed or remapped in tp_aware):
                kept += 1
                yield remapped, tensor
        print(f"Cosmos3 transformer weight remap: kept {kept}/{total} tensors")

    loaded = AutoWeightsLoader(wrapper).load_weights(_remapped_weights())
    transformer.post_load_weights()
    return loaded


def _filter_func_cosmos3(name: str) -> bool:
    """Return True for Cosmos3 modules that must stay BF16.

    vLLM-Omni only threads ModelOpt quant_config into the transformer backbone
    vLLM linear layers. Top-level projections, timestep/audio/action modules,
    embeddings, norms, and modality parameters therefore remain full precision.
    """
    pattern = re.compile(
        r"("
        r"proj_in.*|proj_out.*|time_embedder.*|"
        r"audio_proj_.*|action_proj_.*|.*modality_embed.*|"
        r"language_model\.embed_tokens.*|language_model\.norm.*|"
        r".*layernorm.*|.*norm_q.*|.*norm_k.*|norm_moe_gen.*"
        r")"
    )
    return pattern.fullmatch(name) is not None or pattern.match(name) is not None


def _disable_cosmos3_sensitive_quantizers(mtq: Any, transformer: torch.nn.Module) -> None:
    if hasattr(mtq, "disable_quantizer"):
        mtq.disable_quantizer(transformer, _filter_func_cosmos3)


def _build_forward_loop(
    ctx: _Cosmos3TransformContext,
    args: argparse.Namespace,
    prompts: list[str],
):
    generator = torch.Generator(device="cuda")
    latent_frames = (args.num_frames - 1) // ctx.vae_scale_factor_temporal + 1
    latent_h = args.height // ctx.vae_scale_factor_spatial
    latent_w = args.width // ctx.vae_scale_factor_spatial

    def forward_loop(*_unused_args, **_unused_kwargs) -> None:
        with torch.inference_mode():
            for prompt_idx, prompt in enumerate(prompts):
                generator.manual_seed(args.seed + prompt_idx)
                text_ids, text_mask = _tokenize_prompt(
                    ctx.tokenizer,
                    prompt,
                    device=ctx.transformer.device,
                )
                for step_idx in range(args.calib_steps):
                    ctx.transformer.reset_cache()
                    hidden_states = torch.randn(
                        (
                            1,
                            ctx.transformer.latent_channel_size,
                            latent_frames,
                            latent_h,
                            latent_w,
                        ),
                        generator=generator,
                        device=ctx.transformer.device,
                        dtype=ctx.dtype,
                    )
                    timestep = torch.tensor(
                        [1000.0 - step_idx * (1000.0 / max(args.calib_steps, 1))],
                        device=ctx.transformer.device,
                        dtype=torch.float32,
                    )
                    ctx.transformer(
                        hidden_states=hidden_states,
                        timestep=timestep,
                        text_ids=text_ids,
                        text_mask=text_mask,
                        video_shape=(latent_frames, latent_h, latent_w),
                        fps=args.fps,
                    )

    return forward_loop


def _force_export_quantized_weights(backbone: torch.nn.Module, dtype: torch.dtype) -> int:
    from modelopt.torch.export.quant_utils import (
        QUANTIZATION_NONE,
        get_quantization_format,
        quantizer_attr_names,
        weight_attr_names,
    )
    from modelopt.torch.export.unified_export_hf import _export_quantized_weight

    exported = 0
    for name, module in backbone.named_modules():
        try:
            quantization_format = get_quantization_format(module)
        except Exception as exc:
            print(f"[warn] Could not inspect quantization format for {name}: {exc}", file=sys.stderr)
            continue
        if quantization_format == QUANTIZATION_NONE:
            continue
        for weight_name in weight_attr_names(module):
            quantizer_attrs = quantizer_attr_names(weight_name)
            weight_quantizer = getattr(module, quantizer_attrs.weight_quantizer, None)
            if weight_quantizer is None or not getattr(weight_quantizer, "is_enabled", False):
                continue
            _export_quantized_weight(module, dtype, weight_name)
            exported += 1
    return exported


def _cosmos3_quant_config_block(*, mlp_only: bool) -> dict[str, Any]:
    targets = ["Linear"]
    ignore = [
        "action_modality_embed*",
        "action_proj_in*",
        "action_proj_out*",
        "audio_modality_embed*",
        "audio_proj_in*",
        "audio_proj_out*",
        "language_model.embed_tokens*",
        "language_model.norm*",
        "norm_moe_gen*",
        "proj_in*",
        "proj_out*",
        "time_embedder*",
    ]
    if mlp_only:
        targets = ["*mlp*"]
        ignore.extend(["*self_attn*", "*cross_attention*"])
    return {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "dynamic": True,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                    "scale_bits": 8,
                },
                "weights": {
                    "dynamic": True,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                    "scale_bits": 8,
                },
                "targets": targets,
            }
        },
        "group_size": 16,
        "ignore": sorted(set(ignore)),
        "producer": {"name": "modelopt"},
        "quant_algo": "NVFP4",
        "quant_method": "modelopt_fp4",
        "is_checkpoint_nvfp4_serialized": True,
    }


def _patch_quant_config(output_dir: Path, *, mlp_only: bool) -> None:
    cfg_path = output_dir / "transformer" / "config.json"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)

    existing = cfg.get("quantization_config")
    new_qc = _cosmos3_quant_config_block(mlp_only=mlp_only)
    if isinstance(existing, dict):
        producer = existing.get("producer")
        if isinstance(producer, dict):
            new_qc["producer"] = producer

    cfg["quantization_config"] = new_qc
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _strip_scale_suffix(name: str) -> tuple[str, str]:
    for suffix in MODELOPT_SCALE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    if name.endswith(".weight"):
        return name[: -len(".weight")], ".weight"
    if name.endswith(".bias"):
        return name[: -len(".bias")], ".bias"
    return name, ""


def _inverse_remap_cosmos3_key(name: str) -> str | None:
    base, suffix = _strip_scale_suffix(name)

    direct_prefixes = (
        "proj_in.",
        "proj_out.",
        "time_embedder.",
        "audio_proj_in.",
        "audio_proj_out.",
        "action_proj_in.",
        "action_proj_out.",
    )
    if name.startswith(direct_prefixes):
        return name
    if name in {"audio_modality_embed", "action_modality_embed"}:
        return name

    if name.startswith("language_model.embed_tokens."):
        return name[len("language_model.") :]
    if name.startswith("language_model.norm."):
        return name[len("language_model.") :]
    if name.startswith("norm_moe_gen."):
        return name

    layer_match = re.match(r"language_model\.layers\.(\d+)\.(.+)", base)
    if layer_match:
        layer_idx, rest = layer_match.groups()
        mapping = {
            "self_attn.to_q": "self_attn.to_q",
            "self_attn.to_k": "self_attn.to_k",
            "self_attn.to_v": "self_attn.to_v",
            "self_attn.to_out": "self_attn.to_out",
            "self_attn.norm_q": "self_attn.norm_q",
            "self_attn.norm_k": "self_attn.norm_k",
            "input_layernorm": "input_layernorm",
            "post_attention_layernorm": "post_attention_layernorm",
            "mlp.gate_proj": "mlp.gate_proj",
            "mlp.up_proj": "mlp.up_proj",
            "mlp.down_proj": "mlp.down_proj",
        }
        for src, dst in mapping.items():
            if rest == src:
                return f"layers.{layer_idx}.{dst}{suffix}"

    layer_match = re.match(r"gen_layers\.(\d+)\.(.+)", base)
    if layer_match:
        layer_idx, rest = layer_match.groups()
        mapping = {
            "cross_attention.to_q": "self_attn.add_q_proj",
            "cross_attention.to_k": "self_attn.add_k_proj",
            "cross_attention.to_v": "self_attn.add_v_proj",
            "cross_attention.to_out": "self_attn.to_add_out",
            "cross_attention.norm_q": "self_attn.norm_added_q",
            "cross_attention.norm_k": "self_attn.norm_added_k",
            "input_layernorm": "input_layernorm_moe_gen",
            "post_attention_layernorm": "post_attention_layernorm_moe_gen",
            "mlp.gate_proj": "mlp_moe_gen.gate_proj",
            "mlp.up_proj": "mlp_moe_gen.up_proj",
            "mlp.down_proj": "mlp_moe_gen.down_proj",
        }
        for src, dst in mapping.items():
            if rest == src:
                return f"layers.{layer_idx}.{dst}{suffix}"

    return None


def _diffusers_transformer_state_dict(transformer: torch.nn.Module) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    skipped = 0
    for name, tensor in transformer.state_dict().items():
        mapped = _inverse_remap_cosmos3_key(name)
        if mapped is None:
            skipped += 1
            continue
        result[mapped] = tensor.detach().cpu()
    if skipped:
        print(f"[info] skipped {skipped} internal/non-checkpoint transformer tensors")
    return result


def _save_pipeline_with_nvfp4_transformer(
    transformer: Cosmos3VFMTransformer,
    model_path: Path,
    output_dir: Path,
    *,
    max_shard_size: str,
    mlp_only: bool,
) -> None:
    from modelopt.torch.export.diffusers_utils import hide_quantizers_from_state_dict

    shutil.copytree(model_path, output_dir, ignore=shutil.ignore_patterns("transformer"))
    stale_root_index = output_dir / "model.safetensors.index.json"
    if stale_root_index.exists():
        stale_root_index.unlink()
    transformer_out = output_dir / "transformer"
    transformer_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path / "transformer" / "config.json", transformer_out / "config.json")

    with hide_quantizers_from_state_dict(transformer):
        state_dict = _diffusers_transformer_state_dict(transformer)
    save_torch_state_dict(
        state_dict,
        transformer_out,
        filename_pattern="diffusion_pytorch_model{suffix}.safetensors",
        max_shard_size=max_shard_size,
        safe_serialization=True,
    )
    _patch_quant_config(output_dir, mlp_only=mlp_only)


def _summarize_export(output_dir: Path, started_at: float) -> None:
    cfg_path = output_dir / "transformer" / "config.json"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config", {})
    transformer_size = sum(p.stat().st_size for p in (output_dir / "transformer").rglob("*") if p.is_file())
    print("Export summary:")
    print(f"  output:       {output_dir}")
    print(f"  quant_method: {qc.get('quant_method')}")
    print(f"  quant_algo:   {qc.get('quant_algo')}")
    print(f"  group_size:   {qc.get('group_size')}")
    print(f"  transformer:  {transformer_size / (1024**3):.2f} GiB")
    print(f"  e2e time:     {time.perf_counter() - started_at:.2f} s")


def main() -> None:
    args = _build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for ModelOpt NVFP4 quantization.")

    started_at = time.perf_counter()
    mtq = _require_modelopt()
    model_path = _resolve_model_path(args.model)
    output_dir = _prepare_output(args.output, args.overwrite)
    dtype = _select_dtype(args.dtype)
    prompts = _build_prompts(args)

    print("Quantization plan:")
    print(f"  input:       {model_path}")
    print(f"  output:      {output_dir}")
    print(f"  dtype:       {dtype}")
    print(f"  size:        {args.width}x{args.height}, frames={args.num_frames}, fps={args.fps}")
    print(f"  calib:       prompts={len(prompts)}, steps={args.calib_steps}")
    print(f"  config:      {'NVFP4_MLP_ONLY_CFG' if args.mlp_only else 'NVFP4_DEFAULT_CFG'}")

    try:
        _ensure_single_process_parallel_state()
        ctx = _load_cosmos3_transformer(model_path, dtype)
        quant_config = copy.deepcopy(mtq.NVFP4_MLP_ONLY_CFG if args.mlp_only else mtq.NVFP4_DEFAULT_CFG)
        forward_loop = _build_forward_loop(ctx, args, prompts)
        quantized = mtq.quantize(ctx.transformer, quant_config, forward_loop)
        if quantized is not None:
            ctx.transformer = quantized
        _disable_cosmos3_sensitive_quantizers(mtq, ctx.transformer)

        print("\nForcing NVFP4 weight serialization...")
        exported = _force_export_quantized_weights(ctx.transformer, dtype)
        print(f"  -> {exported} weights converted to NVFP4 in memory")
        if exported == 0:
            raise SystemExit("No quantized weights were exported. Check the quantizer config and ignore filters.")

        print("\nSaving pipeline with NVFP4 transformer...")
        _save_pipeline_with_nvfp4_transformer(
            ctx.transformer,
            model_path,
            output_dir,
            max_shard_size=args.max_shard_size,
            mlp_only=args.mlp_only,
        )
        _summarize_export(output_dir, started_at)
    finally:
        destroy_distributed_env()

    print("\nNext: validate the checkpoint:")
    print(f"  python examples/quantization/check_modelopt_fp8_export.py --output {output_dir} --baseline {model_path}")
    print(
        "  vllm serve "
        f"{output_dir} --omni --model-class-name Cosmos3OmniDiffusersPipeline "
        "--linear-backend cutlass --no-guardrails"
    )


if __name__ == "__main__":
    main()
