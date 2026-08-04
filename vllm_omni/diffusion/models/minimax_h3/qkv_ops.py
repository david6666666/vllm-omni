# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N803
"""Destination-major QKV packing for the fixed Ulysses-4 attention path."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

_BLOCK_SIZE = 1024


@triton.jit
def _pack_qkv_destination_major_kernel(
    output_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    total_elements,
    rows,
    global_heads,
    local_heads,
    head_size,
    q_stride_0,
    q_stride_1,
    k_stride_0,
    k_stride_1,
    v_stride_0,
    v_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    dim = offsets % head_size
    head_slot = offsets // head_size
    row = head_slot // global_heads
    global_head = head_slot % global_heads
    destination = global_head // local_heads
    local_head = global_head % local_heads
    q_value = tl.load(
        q_ptr + row * q_stride_0 + global_head * q_stride_1 + dim,
        mask=mask,
        other=0.0,
    )
    k_value = tl.load(
        k_ptr + row * k_stride_0 + global_head * k_stride_1 + dim,
        mask=mask,
        other=0.0,
    )
    v_value = tl.load(
        v_ptr + row * v_stride_0 + global_head * v_stride_1 + dim,
        mask=mask,
        other=0.0,
    )
    output_base = ((destination * rows + row) * local_heads + local_head) * (3 * head_size)
    tl.store(output_ptr + output_base + dim, q_value, mask=mask)
    tl.store(output_ptr + output_base + head_size + dim, k_value, mask=mask)
    tl.store(output_ptr + output_base + 2 * head_size + dim, v_value, mask=mask)


@triton.jit
def _unpack_qkv_destination_major_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    packed_ptr,
    total_elements,
    rows,
    local_heads,
    head_size,
    q_stride_0,
    q_stride_1,
    packed_stride_0,
    packed_stride_1,
    packed_stride_2,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    dim = offsets % head_size
    head_slot = offsets // head_size
    row_slot = head_slot // local_heads
    row = row_slot % rows
    source = row_slot // rows
    local_head = head_slot % local_heads
    packed_base = (
        source * packed_stride_0
        + row * packed_stride_1
        + local_head * packed_stride_2
    )
    q_value = tl.load(packed_ptr + packed_base + dim, mask=mask, other=0.0)
    k_value = tl.load(packed_ptr + packed_base + head_size + dim, mask=mask, other=0.0)
    v_value = tl.load(packed_ptr + packed_base + 2 * head_size + dim, mask=mask, other=0.0)
    output_row = source * rows + row
    output_base = output_row * q_stride_0 + local_head * q_stride_1
    tl.store(q_ptr + output_base + dim, q_value, mask=mask)
    tl.store(k_ptr + output_base + dim, k_value, mask=mask)
    tl.store(v_ptr + output_base + dim, v_value, mask=mask)


def _valid_qkv_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    world_size: int,
) -> bool:
    return (
        world_size > 1
        and q.is_cuda
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.device == k.device == v.device
        and q.shape == k.shape == v.shape
        and q.dim() == 3
        and q.shape[1] % world_size == 0
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        and not q.requires_grad
        and not k.requires_grad
        and not v.requires_grad
    )


def pack_qkv_destination_major_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    world_size: int,
) -> torch.Tensor | None:
    """Pack local Q/K/V into one destination-major all-to-all buffer."""
    if not _valid_qkv_inputs(q, k, v, world_size):
        return None
    rows, global_heads, head_size = q.shape
    local_heads = global_heads // world_size
    packed = torch.empty(
        (world_size, rows, local_heads, 3 * head_size),
        dtype=q.dtype,
        device=q.device,
    )
    total_elements = rows * global_heads * head_size
    _pack_qkv_destination_major_kernel[(triton.cdiv(total_elements, _BLOCK_SIZE),)](
        packed,
        q,
        k,
        v,
        total_elements,
        rows,
        global_heads,
        local_heads,
        head_size,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
    )
    return packed


def unpack_qkv_destination_major_bf16(
    packed: torch.Tensor,
    rows: int,
    local_heads: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Materialize contiguous Q/K/V from an exchanged destination-major buffer."""
    if (
        not packed.is_cuda
        or packed.dtype != torch.bfloat16
        or packed.dim() != 4
        or packed.shape[0] <= 1
        or packed.shape[1:] != (rows, local_heads, 3 * head_size)
        or not packed.is_contiguous()
    ):
        return None
    world_size = packed.shape[0]
    output_shape = (rows * world_size, local_heads, head_size)
    q = torch.empty(output_shape, dtype=packed.dtype, device=packed.device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    total_elements = rows * world_size * local_heads * head_size
    _unpack_qkv_destination_major_kernel[(triton.cdiv(total_elements, _BLOCK_SIZE),)](
        q,
        k,
        v,
        packed,
        total_elements,
        rows,
        local_heads,
        head_size,
        q.stride(0),
        q.stride(1),
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
    )
    return q, k, v
