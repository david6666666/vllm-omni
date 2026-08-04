# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N803
"""Small fused elementwise kernels used by the MiniMax-H3 DiT blocks."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

_BLOCK_SIZE = 256


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


def _can_use_indexed_kernel(
    x: torch.Tensor,
    parameter: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    is_compiling = getattr(torch.compiler, "is_compiling", lambda: False)()
    return (
        not is_compiling
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and parameter.dtype == torch.bfloat16
        and other.dtype == torch.bfloat16
        and indices.dtype in (torch.int32, torch.int64)
        and x.dim() == 2
        and parameter.dim() == 2
        and other.dim() == 2
        and indices.dim() == 1
        and x.shape == other.shape
        and x.shape[0] == indices.shape[0]
        and x.shape[1] == parameter.shape[1]
        and x.is_contiguous()
        and parameter.is_contiguous()
        and other.is_contiguous()
        and indices.is_contiguous()
    )


def indexed_scale_shift_bf16_(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    """Apply ``x = x * (1 + scale[index]) + shift[index]`` in place."""
    if not _can_use_indexed_kernel(x, scale, shift, indices):
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
    if not _can_use_indexed_kernel(x, gate, other, indices):
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
