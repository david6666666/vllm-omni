#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Convert a MiniMax-H3 FL2VA SVDQuant checkpoint for vLLM-Omni.

The input checkpoint uses Nunchaku's published fragment layout. The output
uses a canonical row-major NVFP4 layout consumed by vLLM's existing linear
kernel. Conversion changes tensor layout only; it does not recalibrate weights.

The converter:

1. resolves a local or Hugging Face source checkpoint and base pipeline;
2. unpacks 208 SVDQuant linears into canonical NVFP4 tensors;
3. reorders H3 fused MLP halves to vLLM's gate/up convention;
4. restores AdaLN and all other non-SVDQuant tensors from the BF16 base; and
5. embeds the SVDQuant configuration in the FL2VA transformer config.

Usage:

    python -m vllm_omni.quantization.tools.convert_nunchaku_to_svdquant \
        --nunchaku-checkpoint ./svdq-fp4_r32-minimax-h3-fl2va.safetensors \
        --base-pipeline MiniMaxAI/MiniMax-H3 \
        --output-dir ./MiniMax-H3-SVDQuant-NVFP4-r32

Unchanged base files are hard-linked by default. Use --copy to create a
physically independent output folder. Ref2VA conversion is not supported.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from vllm.transformers_utils.repo_utils import hf_api

from vllm_omni.quantization.tools.svdquant_nvfp4_layout import (
    _unpack_nibbles,
    unpack_nunchaku_qweight_fp4,
    unpack_nunchaku_scale_vector,
    unpack_nunchaku_wscales_fp4,
)

# ---------------------------------------------------------------------------
# MiniMax-H3-specific knowledge
# ---------------------------------------------------------------------------


def is_fused_gate_up(layer_prefix: str) -> bool:
    """Whether an H3 MLP needs Diffusers up/gate -> vLLM gate/up."""
    if not layer_prefix.endswith(".mlp.fc1"):
        return False
    return layer_prefix.startswith("blocks.") or layer_prefix.startswith("token_refiner.blocks.")


_MINIMAX_H3_EXPECTED_SVDQ_LINEARS = 208
_MINIMAX_H3_EXPECTED_W4A16_LINEARS = 50


# ---------------------------------------------------------------------------
# Per-linear nunchaku-fragment → row-major (with optional half-swap)
# ---------------------------------------------------------------------------


def unpack_lowrank_weight(weight: torch.Tensor, *, down: bool) -> torch.Tensor:
    """Unpack Nunchaku's 16x16 low-rank fragment layout.

    Keep this pure permutation local to the offline converter: importing the
    equivalent Nunchaku helper also imports its compiled CUDA extension, even
    though unpacking itself only needs PyTorch views and permutations.
    """
    c, r = weight.shape
    assert weight.dtype in (torch.float16, torch.bfloat16)

    lane_n, lane_k = 1, 2
    n_pack_size, k_pack_size = 2, 2
    num_n_lanes, num_k_lanes = 8, 4
    frag_n = n_pack_size * num_n_lanes * lane_n
    frag_k = k_pack_size * num_k_lanes * lane_k

    if down:
        r_frags, c_frags = r // frag_n, c // frag_k
    else:
        c_frags, r_frags = c // frag_n, r // frag_k
    weight = weight.view(
        c_frags,
        r_frags,
        num_n_lanes,
        num_k_lanes,
        n_pack_size,
        k_pack_size,
        lane_k,
    )
    weight = weight.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    weight = weight.view(c_frags, r_frags, frag_n, frag_k)
    if down:
        weight = weight.permute(1, 2, 0, 3).contiguous().view(r, c)
    else:
        weight = weight.permute(0, 2, 1, 3).contiguous().view(c, r)
    return weight


def _pack_qweight_row_major(nibs: torch.Tensor) -> torch.Tensor:
    """`[N, K] uint8 nibbles → [N, K/2] uint8`, low nibble = even k.

    Inverse of `_unpack_nibbles`. The on-disk canonical `qweight` is the
    pair-packed nibble byte exactly as the SM_100 CuTe kernel expects.
    """
    assert nibs.shape[-1] % 2 == 0
    lo = nibs[..., 0::2]
    hi = nibs[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_nvfp4_layer(
    params: dict[str, torch.Tensor],
    *,
    half_swap_n: bool,
) -> dict[str, torch.Tensor]:
    """nunchaku fragment → canonical row-major for one NVFP4 SVDQuant linear.

    Pure permute+view (bit-preserving) for `qweight`, `wscales`, `proj_up`,
    and `proj_down`. Nunchaku also fragment-packs one-dimensional scale-like
    tensors, so `smooth_factor`, `wcscales`, and an optional `bias` must be
    restored to logical channel order as well.

    When `half_swap_n=True`, additionally swap the two N-axis halves on
    `qweight`, `wscales`, `proj_up`, `wcscales`, `bias` — the SiluAndMul
    `[gate; hidden]` reorder. Swap happens on row-major intermediates,
    which is free (it's a slice + cat).
    """
    qweight = params["qweight"]  # [N, K/2] int8 (nunchaku fragment)
    wscales = params["wscales"]  # [K/16, N] fp8 (nunchaku fragment)
    proj_up = params["proj_up"]  # [N, R] bf16 (nunchaku fragment)
    proj_down = params["proj_down"]  # [K, R] bf16 (nunchaku fragment)
    wcscales = params.get("wcscales")  # [N] bf16 (optional)
    bias = params.get("bias")  # [N] bf16 (optional)
    smooth_factor = params.get("smooth_factor")  # [K] bf16 (optional)

    N = qweight.shape[0]
    if half_swap_n:
        assert N % 2 == 0, f"fused gate-up N must be even; got {N}"
    half = N // 2

    # qweight: unpack fragment → [N, K/2] uint8 nibble bytes (low = even-k);
    # then `_unpack_nibbles` → [N, K] full-nibble form so we can slice on N
    # then repack to [N, K/2] for storage.
    qw_rm = unpack_nunchaku_qweight_fp4(qweight.view(torch.int8))  # [N, K/2] uint8
    if half_swap_n:
        nibs = _unpack_nibbles(qw_rm)  # [N, K] uint8
        nibs = torch.cat([nibs[half:], nibs[:half]], dim=0).contiguous()
        qw_rm = _pack_qweight_row_major(nibs)
    qweight_out = qw_rm.contiguous()

    # wscales: unpack to [K/16, N] row-major fp8.
    ws_rm = unpack_nunchaku_wscales_fp4(wscales)
    if half_swap_n:
        ws_rm = torch.cat([ws_rm[:, half:], ws_rm[:, :half]], dim=1).contiguous()
    wscales_out = ws_rm.contiguous()

    # proj_up: down=False → unpack returns [N, R] directly.
    pu_rm = unpack_lowrank_weight(proj_up, down=False)
    if half_swap_n:
        pu_rm = torch.cat([pu_rm[half:], pu_rm[:half]], dim=0).contiguous()
    proj_up_out = pu_rm.contiguous()

    # proj_down: down=True. nunchaku's unpack returns [R, K]; canonical
    # row-major is [K, R] (matches SM_100 CuTe kernel's expected layout).
    # Transpose to [K, R].
    pd_rm = unpack_lowrank_weight(proj_down, down=True)
    K = proj_down.shape[0]
    R = proj_down.shape[1]
    if pd_rm.shape == (R, K):
        pd_rm = pd_rm.transpose(0, 1).contiguous()
    assert pd_rm.shape == (K, R), f"proj_down expected ({K}, {R}); got {tuple(pd_rm.shape)}"
    proj_down_out = pd_rm

    out = dict(params)
    out["qweight"] = qweight_out
    out["wscales"] = wscales_out
    out["proj_up"] = proj_up_out
    out["proj_down"] = proj_down_out
    if smooth_factor is not None:
        out["smooth_factor"] = unpack_nunchaku_scale_vector(smooth_factor)
    wcscales_out = unpack_nunchaku_scale_vector(wcscales) if wcscales is not None else None
    bias_out = unpack_nunchaku_scale_vector(bias) if bias is not None else None
    if half_swap_n:
        if wcscales_out is not None:
            wcscales_out = torch.cat([wcscales_out[half:], wcscales_out[:half]]).contiguous()
        if bias_out is not None:
            bias_out = torch.cat([bias_out[half:], bias_out[:half]]).contiguous()
    if wcscales_out is not None:
        out["wcscales"] = wcscales_out
    if bias_out is not None:
        out["bias"] = bias_out
    return out


# ---------------------------------------------------------------------------
# Input materialization
# ---------------------------------------------------------------------------


def _resolve_nunchaku_checkpoint(arg: str) -> Path:
    """Accept a local file path OR an HF spec `<repo_id>/<filename>`.

    Local path is returned as-is. Otherwise the trailing component is treated
    as the filename within the repo, and the rest is the repo id. Downloads
    only on cache miss.
    """
    p = Path(arg)
    if p.exists() and p.is_file():
        return p
    # Treat as HF spec: split into (repo_id, filename).
    parts = arg.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"--nunchaku-checkpoint {arg!r} is not a local file and not a "
            "<repo_id>/<filename> spec (need owner/name/file.safetensors)"
        )
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    print(f"resolving nunchaku checkpoint: repo={repo_id} file={filename}")
    path = hf_api().hf_hub_download(repo_id=repo_id, filename=filename)
    return Path(path)


def _resolve_base_pipeline(arg: str) -> Path:
    """Accept a local diffusers folder OR an HF repo id."""
    p = Path(arg)
    if p.exists() and p.is_dir():
        # Local diffusers folder.
        return p
    # HF repo id → snapshot_download (uses cache if present).
    print(f"resolving base pipeline: repo={arg}")
    path = hf_api().snapshot_download(
        repo_id=arg,
        allow_patterns=["FL2VA/**"],
    )
    return Path(path)


# ---------------------------------------------------------------------------
# Filesystem mirror (hard-link with copy fallback)
# ---------------------------------------------------------------------------


def _link_or_copy_file(src: Path, dst: Path, prefer_copy: bool) -> None:
    """Hard-link src → dst, falling back to copy. Resolves source symlinks
    (the HF cache uses symlink-from-snapshot-to-blob; we want the blob).
    """
    real = src.resolve()
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if prefer_copy:
        shutil.copy2(real, dst)
        return
    try:
        os.link(real, dst)
    except OSError:
        # Cross-fs or permissions: fall back to copy.
        shutil.copy2(real, dst)


def _link_or_copy_tree(src: Path, dst: Path, prefer_copy: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        d = dst / item.name
        if item.is_dir():
            _link_or_copy_tree(item, d, prefer_copy)
        else:
            _link_or_copy_file(item, d, prefer_copy)


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------


# Suffixes nunchaku publishes alongside every quantized linear that the
# vLLM SVDQuant LinearMethod does not consume — they bloat the output
# checkpoint without serving any backend. Filter them at group time so
# downstream conversion / save never touches them.
#
# `smooth_factor_orig`: declared by nunchaku as "(Unused)" (see
# `nunchaku/models/linear.py:54`) and never read by any quantize/forward
# path. Keeping it triggers a KeyError at load time since
# vLLM does not register a `smooth_factor_orig` parameter.
_DROPPED_NUNCHAKU_SUFFIXES: frozenset[str] = frozenset({"smooth_factor_orig"})


def _group_keys_by_layer(
    keys: list[str],
) -> tuple[dict[str, list[str]], list[str]]:
    """Return (layer_prefix → list-of-suffixes, leftover-keys).

    A "linear" is any key prefix that has a `.qweight` sibling. Suffixes
    in `_DROPPED_NUNCHAKU_SUFFIXES` are filtered out entirely.
    """
    qweight_prefixes = {k.rsplit(".", 1)[0] for k in keys if k.endswith(".qweight")}
    layer_to_suffixes: dict[str, list[str]] = {p: [] for p in qweight_prefixes}
    leftover: list[str] = []
    for k in keys:
        prefix, _, suffix = k.rpartition(".")
        if prefix in layer_to_suffixes:
            if suffix in _DROPPED_NUNCHAKU_SUFFIXES:
                continue
            layer_to_suffixes[prefix].append(suffix)
        else:
            leftover.append(k)
    return layer_to_suffixes, leftover


def _detect_rank_precision(f, sample_prefix: str) -> tuple[int, str]:
    proj_down = f.get_tensor(f"{sample_prefix}.proj_down")
    wscales = f.get_tensor(f"{sample_prefix}.wscales")
    rank = int(proj_down.shape[1])
    if wscales.dtype == torch.float8_e4m3fn:
        precision = "nvfp4"
    elif wscales.dtype in (torch.float16, torch.bfloat16):
        precision = "int4"
    else:
        raise ValueError(f"unexpected wscales dtype {wscales.dtype}")
    return rank, precision


def _classify_quantized_layers(
    layer_to_suffixes: dict[str, list[str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Split checkpoint groups into SVDQ W4A4 and AWQ W4A16 linears."""
    svdq: dict[str, list[str]] = {}
    w4a16: dict[str, list[str]] = {}
    for prefix, suffixes in layer_to_suffixes.items():
        suffix_set = set(suffixes)
        is_svdq = "proj_down" in suffix_set
        is_w4a16 = "wzeros" in suffix_set
        if is_svdq == is_w4a16:
            raise ValueError(
                f"quantized layer {prefix!r} must have exactly one of proj_down (SVDQ) or wzeros (AWQ W4A16)"
            )
        (svdq if is_svdq else w4a16)[prefix] = suffixes
    return svdq, w4a16


def _base_weight_files(base_transformer: Path) -> list[Path]:
    index_paths = sorted(base_transformer.glob("*.safetensors.index.json"))
    if index_paths:
        if len(index_paths) != 1:
            raise ValueError(f"expected one safetensors index in {base_transformer}, found {len(index_paths)}")
        with open(index_paths[0]) as fp:
            index = json.load(fp)
        filenames = sorted(set(index.get("weight_map", {}).values()))
        if not filenames:
            raise ValueError(f"empty weight_map in {index_paths[0]}")
        return [base_transformer / filename for filename in filenames]
    files = sorted(base_transformer.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors weights in {base_transformer}")
    return files


def _load_base_unquantized_tensors(
    base_transformer: Path,
    *,
    quantized_prefixes: set[str],
    existing_keys: set[str],
) -> dict[str, torch.Tensor]:
    """Load only dense H3 tensors not replaced by a quantized group."""
    tensors: dict[str, torch.Tensor] = {}
    for weight_file in _base_weight_files(base_transformer):
        with safe_open(str(weight_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in existing_keys:
                    continue
                prefix, _, suffix = key.rpartition(".")
                if prefix in quantized_prefixes and suffix == "weight":
                    continue
                if key in tensors:
                    raise ValueError(f"duplicate base transformer tensor {key!r}")
                tensors[key] = f.get_tensor(key)
    return tensors


def convert(
    nunchaku_checkpoint: Path,
    base_pipeline: Path,
    output_dir: Path,
    *,
    prefer_copy: bool,
    progress: bool = True,
) -> None:
    """Drive the full conversion. See module docstring for behavior."""
    base_partition = base_pipeline / "FL2VA"
    if not base_partition.is_dir():
        if base_pipeline.name == "FL2VA" and base_pipeline.is_dir():
            base_partition = base_pipeline
        else:
            raise FileNotFoundError(f"MiniMax-H3 FL2VA partition not found in {base_pipeline}")
    output_partition = output_dir / "FL2VA"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_partition.mkdir(parents=True, exist_ok=True)

    # ----- Mirror base pipeline (everything except transformer/) -----
    base_top_level = sorted(base_partition.iterdir(), key=lambda p: p.name)
    for item in base_top_level:
        if item.name == "transformer":
            continue
        d = output_partition / item.name
        if item.is_dir():
            _link_or_copy_tree(item, d, prefer_copy)
        else:
            _link_or_copy_file(item, d, prefer_copy)
    print(f"mirrored {len(base_top_level) - 1} top-level entries from base ({'copy' if prefer_copy else 'hard-link'})")

    # ----- transformer/ -----
    transformer_dir = output_partition / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    base_transformer = base_partition / "transformer"

    # ----- Scan nunchaku checkpoint -----
    with safe_open(str(nunchaku_checkpoint), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        metadata = f.metadata() or {}
        layer_to_suffixes, leftover = _group_keys_by_layer(keys)
        if not layer_to_suffixes:
            raise RuntimeError("no quantized linears found (no .qweight keys)")
        svdq_layers, w4a16_layers = _classify_quantized_layers(layer_to_suffixes)
        if not svdq_layers:
            raise RuntimeError("no SVDQ W4A4 linears found (no .proj_down keys)")
        if len(svdq_layers) != _MINIMAX_H3_EXPECTED_SVDQ_LINEARS:
            raise ValueError(
                "MiniMax-H3 FL2VA requires exactly "
                f"{_MINIMAX_H3_EXPECTED_SVDQ_LINEARS} SVDQuant linears; "
                f"got {len(svdq_layers)}"
            )
        if len(w4a16_layers) not in {0, _MINIMAX_H3_EXPECTED_W4A16_LINEARS}:
            raise ValueError(
                "expected either zero or "
                f"{_MINIMAX_H3_EXPECTED_W4A16_LINEARS} auxiliary W4A16 "
                f"linears, got {len(w4a16_layers)}"
            )
        sample_prefix = next(iter(svdq_layers))
        rank, precision = _detect_rank_precision(f, sample_prefix)
        if precision != "nvfp4":
            raise ValueError(f"MiniMax-H3 Phase 1 requires NVFP4 input, got {precision!r}")

        n_linears = len(svdq_layers)
        n_fused = sum(1 for p in svdq_layers if is_fused_gate_up(p))
        print(
            f"nunchaku checkpoint: {len(svdq_layers)} SVDQuant linears, "
            f"{len(w4a16_layers)} auxiliary W4A16 groups restored from BF16 base, "
            f"{n_fused} fused gate-up (to swap); {len(leftover)} other keys"
        )
        print(f"detected rank={rank} precision={precision}")
        if "model_class" in metadata:
            print(f"nunchaku metadata model_class={metadata['model_class']!r}")

        # ----- Build output state_dict via streaming reads -----
        out_sd: dict[str, torch.Tensor] = {}

        for i, (prefix, suffixes) in enumerate(sorted(svdq_layers.items())):
            params: dict[str, torch.Tensor] = {}
            for suf in suffixes:
                params[suf] = f.get_tensor(f"{prefix}.{suf}")

            # Make each SVDQuant state block self-contained with identity
            # defaults for optional NVFP4 outer scales.
            qweight = params["qweight"]
            n_outputs = qweight.shape[0]
            lora_dtype = params["proj_up"].dtype
            if "wcscales" not in params:
                params["wcscales"] = torch.ones(n_outputs, dtype=lora_dtype)
            if "wtscale" not in params:
                params["wtscale"] = torch.tensor([1.0], dtype=lora_dtype)
            elif params["wtscale"].dim() == 0:
                params["wtscale"] = params["wtscale"].view(1).contiguous()

            params = unpack_nvfp4_layer(
                params,
                half_swap_n=is_fused_gate_up(prefix),
            )

            # H3's offline recipe exports vLLM-native fused layer names.
            out_prefix = prefix
            for suf, t in params.items():
                out_sd[f"{out_prefix}.{suf}"] = t
            if progress and (i % 20 == 0 or i == n_linears - 1):
                print(f"  [{i + 1}/{n_linears}] {prefix}" + (f"  ->  {out_prefix}" if out_prefix != prefix else ""))

    # Diffuse-compressor's leftovers use its Diffusers module names. The
    # official FL2VA checkpoint already uses vLLM's fused names, so source
    # untouched tensors from that base and exclude dense weights replaced
    # by quantized groups.
    quantized_prefixes = set(svdq_layers)
    base_tensors = _load_base_unquantized_tensors(
        base_transformer,
        quantized_prefixes=quantized_prefixes,
        existing_keys=set(out_sd),
    )
    out_sd.update(base_tensors)
    print(f"loaded {len(base_tensors)} untouched tensors from the FL2VA base")

    # ----- transformer/config.json: inject quantization_config -----
    # vllm-omni reads `transformer/config.json["quantization_config"]` to
    # auto-detect the quant method (see `OmniDiffusionConfig` /
    # `TransformerConfig.from_dict`); a sidecar `quantization_config.json`
    # is *not* consulted. Mirror what `merge_mxfp8_checkpoint.py` does:
    # load base config.json, inject the dict, write back.
    #
    # SVDQuant is checkpoint-only. Auxiliary H3 encoders remain BF16 and are
    # routed separately by the component quantization configuration.
    with open(base_transformer / "config.json") as fp:
        tf_config = json.load(fp)
    tf_config["quantization_config"] = {
        "quant_method": "svdquant",
        "rank": rank,
        "precision": "nvfp4",
        "act_unsigned": False,
        # Phase 1 keeps AdaLN dense. The source checkpoint may contain W4A16
        # groups, but their BF16 weights are restored from the official base.
        "modules_to_not_convert": sorted(
            {
                "condition_proj",
                "final_layer.adaln_proj.linear",
                *w4a16_layers,
            }
        ),
    }
    out_config_path = transformer_dir / "config.json"
    # Defensive: a previous run may have hard-linked config.json from the base
    # snapshot; open(..., "w") would truncate the shared inode and corrupt
    # the base's cached blob. Unlink first to detach.
    if out_config_path.exists() or out_config_path.is_symlink():
        out_config_path.unlink()
    with open(out_config_path, "w") as fp:
        json.dump(tf_config, fp, indent=2)
    print(f"wrote {out_config_path} (with embedded quantization_config)")

    # ----- Write the converted single safetensors -----
    out_path = transformer_dir / "diffusion_pytorch_model.safetensors"
    # Preserve nunchaku metadata so downstream can still inspect provenance.
    out_metadata = {k: v for k, v in metadata.items() if isinstance(v, str)}
    out_metadata["conversion"] = json.dumps(
        {
            "tool": "vllm_omni.quantization.tools.convert_nunchaku_to_svdquant",
            "layout": "row_major",
            "model_family": "minimax_h3",
            "svdq_linears": len(svdq_layers),
            "dense_restored_linears": len(w4a16_layers),
            "half_swapped_layers": [p for p in svdq_layers if is_fused_gate_up(p)],
        }
    )
    save_file(out_sd, str(out_path), metadata=out_metadata)
    print(f"wrote {out_path} ({out_path.stat().st_size / 2**30:.2f} GiB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--nunchaku-checkpoint",
        required=True,
        help="Local path to nunchaku merged .safetensors or an HF <repo_id>/<filename> spec.",
    )
    parser.add_argument(
        "--base-pipeline",
        default="MiniMaxAI/MiniMax-H3",
        help="Local MiniMax-H3 folder or HF repo id.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output diffusers folder path.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy non-transformer files instead of hard-linking (slower, "
        "uses ~35 GiB extra). Default: hard-link (HF upload-safe).",
    )
    args = parser.parse_args()

    nunchaku_path = _resolve_nunchaku_checkpoint(args.nunchaku_checkpoint)
    base_path = _resolve_base_pipeline(args.base_pipeline)
    output_dir = Path(args.output_dir).expanduser()

    print(f"nunchaku checkpoint: {nunchaku_path}")
    print(f"base pipeline:       {base_path}")
    print(f"output:              {output_dir}")
    print()

    convert(
        nunchaku_checkpoint=nunchaku_path,
        base_pipeline=base_path,
        output_dir=output_dir,
        prefer_copy=args.copy,
    )
    print("\ndone.")


if __name__ == "__main__":
    main()
