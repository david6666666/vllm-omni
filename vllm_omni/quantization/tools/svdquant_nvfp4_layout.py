# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unpack Nunchaku NVFP4 SVDQuant tensors to canonical row-major layout.

These helpers are pure, bit-preserving view and permutation operations used by
MiniMax-H3 offline checkpoint conversion. Runtime Nunchaku packing is outside
the Phase 1 support contract.
"""

from __future__ import annotations

import torch

_WARP_N = 128
_INSN_K = 64
_GROUP = 16


def _pack_nibbles(nibs: torch.Tensor) -> torch.Tensor:
    """`[*, K] uint8 nibbles → [*, K/2] uint8`. Low nibble = even k."""
    assert nibs.shape[-1] % 2 == 0
    lo = nibs[..., 0::2]
    hi = nibs[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def _unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """`[*, K/2] uint8 → [*, K] uint8 nibbles`. Inverse of `_pack_nibbles`."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.stack([lo, hi], dim=-1)
    return out.view(*packed.shape[:-1], packed.shape[-1] * 2)


def _wscale_view_shape(N: int, K: int) -> tuple[int, ...]:  # noqa: N803
    assert N % _WARP_N == 0, f"N ({N}) must be multiple of {_WARP_N}"
    assert K % _INSN_K == 0, f"K ({K}) must be multiple of {_INSN_K}"
    return (N // _WARP_N, 1, 4, 4, 8, K // _INSN_K, 4)


def unpack_nunchaku_wscales_fp4(scales_nun: torch.Tensor) -> torch.Tensor:
    """nunchaku fragment `[K/16, N]` fp8 → row-major `[K/16, N]` fp8."""
    KG, N = scales_nun.shape
    K = KG * _GROUP
    s = scales_nun.view(N // _WARP_N, K // _INSN_K, 1, 8, 4, 4, 4)
    # Inverse of permute (0, 5, 1, 4, 3, 2, 6) is (0, 2, 5, 4, 3, 1, 6).
    s = s.permute(0, 2, 5, 4, 3, 1, 6).contiguous()
    s = s.view(N, K // _GROUP)
    return s.transpose(0, 1).contiguous()


def unpack_nunchaku_qweight_fp4(q_nun: torch.Tensor) -> torch.Tensor:
    """`[N, K/2] nunchaku fragment int8 → [N, K/2] uint8` (low nibble = even k)."""
    N, K2 = q_nun.shape
    K = K2 * 2
    assert N % _WARP_N == 0
    assert K % _INSN_K == 0
    n_tiles, k_tiles = N // _WARP_N, K // _INSN_K
    q_int = q_nun.contiguous().view(dtype=torch.int32)
    q_int = q_int.reshape(n_tiles, k_tiles, 1, 8, 8, 4, 2, 2, 1)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=q_int.device)
    nibs = ((q_int.unsqueeze(-1) >> shifts) & 0xF).to(torch.uint8)
    # Inverse of permute (0, 5, 6, 1, 3, 8, 2, 7, 4, 9) is (0, 3, 6, 4, 8, 1, 2, 7, 5, 9).
    nibs = nibs.permute(0, 3, 6, 4, 8, 1, 2, 7, 5, 9).contiguous()
    nibs = nibs.view(N, K)
    return _pack_nibbles(nibs)


def unpack_nunchaku_scale_vector(scale_nun: torch.Tensor) -> torch.Tensor:
    """Restore a Nunchaku-packed ``[N]`` scale-like vector to row order."""
    if scale_nun.ndim != 1:
        raise ValueError(f"scale vector must be one-dimensional, got {tuple(scale_nun.shape)}")
    n = scale_nun.numel()
    if n % _WARP_N:
        raise ValueError(f"scale vector length ({n}) must be a multiple of {_WARP_N}")
    s_pack_size = min(max(_WARP_N // 32, 2), 8)
    num_s_lanes = min(32, _WARP_N // s_pack_size)
    num_s_packs = _WARP_N // (s_pack_size * num_s_lanes)
    unpacked = scale_nun.contiguous().view(
        n // _WARP_N,
        1,
        num_s_packs,
        num_s_lanes // 4,
        4,
        s_pack_size // 2,
        2,
    )
    return unpacked.permute(0, 2, 3, 5, 4, 6, 1).contiguous().view(n)


__all__ = [
    "unpack_nunchaku_scale_vector",
    "unpack_nunchaku_qweight_fp4",
    "unpack_nunchaku_wscales_fp4",
]
