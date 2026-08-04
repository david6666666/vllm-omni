# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N803
"""Small fused elementwise kernels used by the MiniMax-H3 DiT blocks."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

_BLOCK_SIZE = 256
_QK_BLOCK_SIZE = 128


@triton.jit
def _indexed_scale_shift_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    indices_ptr,
    n_cols,
    x_stride_0,
    x_stride_1,
    scale_stride_0,
    scale_stride_1,
    shift_stride_0,
    shift_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    index = tl.load(indices_ptr + row)
    x_offset = row * x_stride_0 + cols * x_stride_1
    scale_offset = index * scale_stride_0 + cols * scale_stride_1
    shift_offset = index * shift_stride_0 + cols * shift_stride_1
    x_value = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)
    scale_value = tl.load(scale_ptr + scale_offset, mask=mask, other=0.0).to(tl.float32)
    shift_value = tl.load(shift_ptr + shift_offset, mask=mask, other=0.0).to(tl.float32)
    result = (x_value * (1.0 + scale_value) + shift_value).to(x_ptr.dtype.element_ty)
    tl.store(x_ptr + x_offset, result, mask=mask)


@triton.jit
def _indexed_gate_kernel(
    x_ptr,
    gate_ptr,
    other_ptr,
    indices_ptr,
    n_cols,
    x_stride_0,
    x_stride_1,
    gate_stride_0,
    gate_stride_1,
    other_stride_0,
    other_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    index = tl.load(indices_ptr + row)
    x_offset = row * x_stride_0 + cols * x_stride_1
    gate_offset = index * gate_stride_0 + cols * gate_stride_1
    other_offset = row * other_stride_0 + cols * other_stride_1
    x_value = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)
    gate_value = tl.load(gate_ptr + gate_offset, mask=mask, other=0.0).to(tl.float32)
    other_value = tl.load(other_ptr + other_offset, mask=mask, other=0.0).to(tl.float32)
    result = (x_value + gate_value * other_value).to(x_ptr.dtype.element_ty)
    tl.store(x_ptr + x_offset, result, mask=mask)


@triton.jit
def _qknorm_rope_kernel(
    x_ptr,
    norm_weight_ptr,
    rope_cache_ptr,
    n_tokens,
    n_heads,
    head_dim,
    rope_dim,
    x_stride_0,
    x_stride_1,
    x_stride_2,
    norm_stride,
    rope_stride_0,
    rope_stride_1,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < head_dim
    x_base = x_ptr + token * x_stride_0 + head * x_stride_1
    values = tl.load(x_base + cols * x_stride_2, mask=mask, other=0.0).to(tl.float32)
    norm_weight = tl.load(norm_weight_ptr + cols * norm_stride, mask=mask, other=1.0).to(tl.float32)
    rms = tl.rsqrt(tl.sum(values * values, axis=0) / head_dim + eps)
    normed = (values * rms * norm_weight).to(x_ptr.dtype.element_ty)

    half_rope = rope_dim // 2
    pair = tl.where(cols < half_rope, cols + half_rope, cols - half_rope)
    pair_mask = (cols < rope_dim) & (pair < head_dim)
    pair_values = tl.load(x_base + pair * x_stride_2, mask=pair_mask, other=0.0).to(tl.float32)
    pair_weight = tl.load(norm_weight_ptr + pair * norm_stride, mask=pair_mask, other=1.0).to(tl.float32)
    pair_normed = (pair_values * rms * pair_weight).to(x_ptr.dtype.element_ty)

    rope_base = rope_cache_ptr + token * rope_stride_0
    cos = tl.load(rope_base + cols * rope_stride_1, mask=pair_mask, other=1.0).to(tl.float32)
    sin = tl.load(rope_base + (rope_dim + cols) * rope_stride_1, mask=pair_mask, other=0.0).to(tl.float32)
    sign = tl.where(cols < half_rope, -1.0, 1.0)
    # Match the reference order: each BF16 product is rounded before the
    # rotated pair is added, just as the unfused Torch path multiplies BF16
    # tensors and then adds the two BF16 products.
    first_product = (normed.to(tl.float32) * cos).to(x_ptr.dtype.element_ty)
    second_product = (pair_normed.to(tl.float32) * sin).to(x_ptr.dtype.element_ty)
    rotated = (first_product.to(tl.float32) + sign * second_product.to(tl.float32)).to(
        x_ptr.dtype.element_ty
    )
    result = tl.where(cols < rope_dim, rotated, normed)
    tl.store(x_base + cols * x_stride_2, result, mask=mask)


def _can_use_indexed_common(
    x: torch.Tensor,
    parameter: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    is_compiling = getattr(torch.compiler, "is_compiling", lambda: False)()
    return (
        not is_compiling
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and parameter.dtype == torch.bfloat16
        and indices.dtype in (torch.int32, torch.int64)
        and x.dim() == 2
        and parameter.dim() == 2
        and indices.dim() == 1
        and x.shape[0] == indices.shape[0]
        and x.shape[1] == parameter.shape[1]
        and x.is_contiguous()
        and parameter.is_contiguous()
        and indices.is_contiguous()
    )


def indexed_scale_shift_bf16_(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    """Apply ``x = x * (1 + scale[index]) + shift[index]`` in place."""
    if not (
        _can_use_indexed_common(x, scale, indices)
        and shift.dtype == torch.bfloat16
        and shift.shape == scale.shape
        and shift.is_contiguous()
    ):
        return False
    grid = (x.shape[0], triton.cdiv(x.shape[1], _BLOCK_SIZE))
    _indexed_scale_shift_kernel[grid](
        x,
        scale,
        shift,
        indices,
        x.shape[1],
        x.stride(0),
        x.stride(1),
        scale.stride(0),
        scale.stride(1),
        shift.stride(0),
        shift.stride(1),
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )
    return True


def indexed_gate_bf16_(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    """Apply ``x += gate[index] * other`` in place."""
    if not (
        _can_use_indexed_common(x, gate, indices)
        and other.dtype == torch.bfloat16
        and other.shape == x.shape
        and other.is_contiguous()
    ):
        return False
    grid = (x.shape[0], triton.cdiv(x.shape[1], _BLOCK_SIZE))
    _indexed_gate_kernel[grid](
        x,
        gate,
        other,
        indices,
        x.shape[1],
        x.stride(0),
        x.stride(1),
        gate.stride(0),
        gate.stride(1),
        other.stride(0),
        other.stride(1),
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )
    return True


def fused_qknorm_rope_bf16_(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_cache: torch.Tensor,
    *,
    eps: float,
    rope_dim: int,
) -> bool:
    """Fuse q/k RMSNorm and H3's partial RoPE into two in-place kernels."""
    is_compiling = getattr(torch.compiler, "is_compiling", lambda: False)()
    if (
        is_compiling
        or not q.is_cuda
        or q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or q_weight.dtype != torch.bfloat16
        or k_weight.dtype != torch.bfloat16
        or rope_cache.dtype != torch.bfloat16
        or q.dim() != 3
        or k.dim() != 3
        or q.shape[-1] != _QK_BLOCK_SIZE
        or k.shape[-1] != _QK_BLOCK_SIZE
        or q.shape[0] != k.shape[0]
        or q_weight.shape != (q.shape[-1],)
        or k_weight.shape != (k.shape[-1],)
        or rope_cache.dim() != 2
        or rope_cache.shape[0] != q.shape[0]
        or rope_cache.shape[1] < 2 * rope_dim
        or not q.is_contiguous()
        or not k.is_contiguous()
        or not q_weight.is_contiguous()
        or not k_weight.is_contiguous()
        or not rope_cache.is_contiguous()
        or q.requires_grad
        or k.requires_grad
    ):
        return False

    for x, weight in ((q, q_weight), (k, k_weight)):
        grid = (x.shape[0], x.shape[1])
        _qknorm_rope_kernel[grid](
            x,
            weight,
            rope_cache,
            x.shape[0],
            x.shape[1],
            x.shape[-1],
            rope_dim,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            weight.stride(0),
            rope_cache.stride(0),
            rope_cache.stride(1),
            eps,
            BLOCK_SIZE=_QK_BLOCK_SIZE,
            num_warps=4,
        )
    return True
