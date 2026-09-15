#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Convert FastH3 V2's full transformer to a native MiniMax-H3 checkpoint.

Download the V2 transformer and JSON metadata first. The frozen components are
symlinked from a native MiniMax-H3 FL2VA checkpoint, so that checkpoint must stay
available. The output is published only after every weight has been validated.

    python tools/prepare_fasth3_checkpoint.py \
        --source /path/to/FastH3-8-Step-V2 --base /path/to/MiniMax-H3 \
        --output /path/to/FastH3-8-Step-V2-Omni
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm_omni.diffusion.models.minimax_h3.fasth3 import (
    _place_in_grouped_qkv,
    _resolve_native_target,
    _swap_halves,
)
from vllm_omni.diffusion.models.minimax_h3.fasth3_checkpoint import (
    FASTH3_V2_BASE_SCHEDULE,
    FASTH3_V2_MODEL_ID,
    FastH3CheckpointSpec,
)

_CONFIG_NAMES = {
    "num_refiner_layers": "token_refiner_num_layers",
    "ffn_dim": "ffn_hidden_size",
    "in_channels": "latents_dim",
    "audio_in_channels": "audio_latents_dim",
    "freq_dim": "timestep_input_dim",
    "time_embed_hidden_dim": "time_embed_hidden_size",
    "rope_freq_dim": "rope_inv_freq_len",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def native_weight_target(name: str) -> tuple[str, str]:
    """Return a native name and layout, including full-checkpoint-only norms."""
    module, _, kind = name.rpartition(".")
    if kind not in {"weight", "bias"}:
        raise ValueError(f"unknown FastH3 weight {name}")
    target = _resolve_native_target(module)
    if target is not None:
        return f"{target[0]}.{kind}", target[1]
    if module == "token_refiner.final_norm":
        return name, "plain"
    for source, destination in (
        ("transformer_blocks.", "blocks."),
        ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
    ):
        if not module.startswith(source):
            continue
        block, _, suffix = module[len(source) :].partition(".")
        norms = {"attn.norm_q": "attn.q_norm", "attn.norm_k": "attn.k_norm"}
        if block.isdigit() and suffix in norms:
            return f"{destination}{block}.{norms[suffix]}.{kind}", "plain"
    raise ValueError(f"unknown FastH3 weight {name}")


def iter_native_weights(
    source_shapes: Mapping[str, tuple[int, ...]],
    get_tensor: Callable[[str], torch.Tensor],
    expected_shapes: Mapping[str, tuple[int, ...]],
    *,
    head_dim: int,
    rope_inv_freq: torch.Tensor,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Read only one native parameter at a time, merging separate Q/K/V rows.

    ``expected_shapes`` comes from the base checkpoint headers plus VSA gates.
    Source coverage is checked before yielding anything; each transformed shape
    is checked before it can be written to an output shard.
    """
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for source_name in source_shapes:
        target, layout = native_weight_target(source_name)
        if layout in groups[target]:
            raise ValueError(f"duplicate FastH3 target {target} ({layout})")
        groups[target][layout] = source_name
    expected = set(expected_shapes) - {"rope.inv_freq"}
    if set(groups) != expected:
        raise ValueError(
            f"FastH3 weight coverage mismatch: missing={sorted(expected - groups.keys())}, "
            f"unexpected={sorted(groups.keys() - expected)}"
        )
    for name, sources in sorted(groups.items()):
        if set(sources) & {"q", "k", "v"}:
            if set(sources) != {"q", "k", "v"}:
                raise ValueError(f"incomplete QKV group for {name}: {sorted(sources)}")
            tensor = _place_in_grouped_qkv({slot: get_tensor(key) for slot, key in sources.items()}, head_dim=head_dim)
        elif len(sources) == 1:
            layout, key = next(iter(sources.items()))
            tensor = get_tensor(key)
            if layout == "swap_halves":
                tensor = _swap_halves(tensor)
        else:
            raise ValueError(f"conflicting layouts for {name}: {sorted(sources)}")
        if tuple(tensor.shape) != tuple(expected_shapes[name]):
            raise ValueError(f"shape mismatch for {name}: {tuple(tensor.shape)} != {expected_shapes[name]}")
        yield name, tensor.contiguous()
    if tuple(rope_inv_freq.shape) != tuple(expected_shapes["rope.inv_freq"]):
        raise ValueError("FastH3 RoPE shape does not match the native checkpoint")
    yield "rope.inv_freq", rope_inv_freq


def _open_shards(
    stack: ExitStack, folder: Path, index_file: str
) -> tuple[dict[str, tuple[int, ...]], Callable[[str], torch.Tensor]]:
    weight_map = _read_json(folder / index_file)["weight_map"]
    shards = {
        filename: stack.enter_context(safe_open(folder / filename, framework="pt", device="cpu"))
        for filename in sorted(set(weight_map.values()))
    }
    actual = {key: filename for filename, shard in shards.items() for key in shard.keys()}
    if actual != weight_map or sum(len(shard.keys()) for shard in shards.values()) != len(weight_map):
        raise ValueError(f"{folder}: shard contents do not match the weight index")
    shapes = {name: tuple(shards[filename].get_slice(name).get_shape()) for name, filename in weight_map.items()}
    return shapes, lambda name: shards[weight_map[name]].get_tensor(name)


def prepare_checkpoint(source: Path, base: Path, output: Path) -> Path:
    source, base, output = source.resolve(), base.resolve(), output.absolute()
    if (base / "FL2VA").is_dir():
        base = base / "FL2VA"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}; choose a new directory")
    contract = _read_json(source / "fastvideo_inference.json")
    required = {
        "model_id": FASTH3_V2_MODEL_ID,
        "schema_version": "fasth3-inference-contract-v1",
        "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
        "transformer_forwards": 8,
        "video_scheduler_shift": 10.0,
        "audio_scheduler_shift": 3.0,
        "guidance_scale": 1.0,
        "attention_backend": "VIDEO_SPARSE_ATTN_H3",
        "vsa_sparsity": 0.8,
        "vsa_tile_size": 64,
        "task": "t2av",
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise ValueError(f"unsupported FastH3 contract: {key}={contract.get(key)!r}, expected {expected!r}")
    native_index = _read_json(base / "model_index.json")
    base_release = native_index.get("_minimax_h3", {})
    if base_release.get("partition") != "fl2va" or base_release.get("base_schedule") is not None:
        raise ValueError("--base must be the undistilled native MiniMax-H3 FL2VA checkpoint")
    source_config = _read_json(source / "transformer/config.json")
    base_config = _read_json(base / "transformer/config.json")
    for key, value in source_config.items():
        if key.startswith("_") or key == "rope_theta":
            continue
        native_key = _CONFIG_NAMES.get(key, key)
        if native_key not in base_config or base_config[native_key] != value:
            raise ValueError(f"incompatible base transformer config: {key}={value!r}")
    components = ("text_encoder", "tokenizer", "processor", "video_vae", "audio_vae")
    for component in components:
        if not (base / component).is_dir():
            raise ValueError(f"missing frozen base component: {base / component}")
    release = {
        "schema_version": 1,
        "partition": "fl2va",
        "tasks": ["t2va"],
        "sigma_shift_scales": {"video": 10.0, "audio": 3.0},
        "base_schedule": list(FASTH3_V2_BASE_SCHEDULE.base_schedule),
        "fasth3": {"model_id": FASTH3_V2_MODEL_ID, "vsa_sparsity": 0.8, "vsa_tile_size": 64},
    }
    FastH3CheckpointSpec.from_metadata(release)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary, ExitStack() as stack:
        root = Path(temporary)
        partition = root / "FL2VA"
        transformer = partition / "transformer"
        transformer.mkdir(parents=True)
        source_shapes, get_source = _open_shards(
            stack, source / "transformer", "diffusion_pytorch_model.safetensors.index.json"
        )
        expected_shapes, get_base = _open_shards(stack, base / "transformer", "model.safetensors.index.json")
        for i in range(int(base_config["num_layers"])):
            expected_shapes[f"blocks.{i}.attn.to_gate_compress.weight"] = (
                int(base_config["num_attention_heads"]) * int(base_config["attention_head_dim"]),
                int(base_config["hidden_size"]),
            )
        freq_dim = int(source_config["rope_freq_dim"])
        rope = 1.0 / (
            float(source_config["rope_theta"])
            ** (torch.arange(0, 2 * freq_dim, 2, dtype=torch.float32) / (2 * freq_dim))
        )
        if not torch.equal(rope, get_base("rope.inv_freq").float()):
            raise ValueError("FastH3 analytic RoPE differs from the native base checkpoint")
        shard: dict[str, torch.Tensor] = {}
        weight_map: dict[str, str] = {}
        shard_bytes = total_bytes = 0
        shard_number = 0

        def flush() -> None:
            nonlocal shard_number, shard_bytes
            if not shard:
                return
            shard_number += 1
            filename = f"model-{shard_number:05d}.safetensors"
            save_file(shard, transformer / filename, metadata={"format": "pt"})
            weight_map.update({name: filename for name in shard})
            print(f"Wrote {filename}: {len(shard)} tensors, {shard_bytes} bytes", flush=True)
            shard.clear()
            shard_bytes = 0

        for name, tensor in iter_native_weights(
            source_shapes,
            get_source,
            expected_shapes,
            head_dim=int(base_config["attention_head_dim"]),
            rope_inv_freq=rope,
        ):
            size = tensor.numel() * tensor.element_size()
            if shard_bytes + size > 5 * 1024**3:
                flush()
            shard[name] = tensor
            shard_bytes += size
            total_bytes += size
        flush()
        _write_json(
            transformer / "model.safetensors.index.json",
            {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
        )
        _write_json(transformer / "config.json", base_config)
        for component in components:
            (partition / component).symlink_to(base / component, target_is_directory=True)
        native_index["_minimax_h3"] = release
        _write_json(partition / "model_index.json", native_index)
        # Root discovery still uses the existing MiniMaxH3Pipeline registry key.
        _write_json(root / "model_index.json", native_index)
        _write_json(root / "fastvideo_inference.json", contract)
        _write_json(
            root / "conversion.json",
            {
                "source": str(source),
                "base": str(base),
                "source_tensors": len(source_shapes),
                "native_tensors": len(weight_map),
                "frozen_components": list(components),
            },
        )
        root.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Downloaded FastH3 V2 snapshot")
    parser.add_argument("--base", type=Path, required=True, help="Native base H3 root or FL2VA directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are refused")
    args = parser.parse_args()
    print(prepare_checkpoint(args.source, args.base, args.output))


if __name__ == "__main__":
    main()
