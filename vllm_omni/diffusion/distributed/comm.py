# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team & Jiarui Fang
#  from https://github.com/feifeibear/long-context-attention/blob/main/yunchang/comm/all_to_all.py
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from vllm_omni.platforms import current_omni_platform

__all__ = ["all_to_all_4D", "all_to_all_5D", "SeqAllToAll4D", "SeqAllToAll5D", "RingComm"]


def all_to_all_4D(
    input: torch.tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None, use_sync: bool = False
) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group (torch.distributed.ProcessGroup): torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert input.dim() == 4, f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs).transpose(0, 2).contiguous()

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head

        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                current_omni_platform.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)->
        #  (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                current_omni_platform.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -transpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync
        return all_to_all_4D(input, scatter_idx, gather_idx, group=group, use_sync=use_sync)


def all_to_all_5D(
    input: torch.tensor, scatter_idx: int = 3, gather_idx: int = 1, group=None, use_sync: bool = False
) -> torch.tensor:
    """
    all-to-all for QKV
    forward (bs, seqlen/N, 3, hc, hs) -> (bs, seqlen, 3, hc/N, hs)

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group (torch.distributed.ProcessGroup): torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, 3, hc, hs)
    """
    assert input.dim() == 5, f"input must be 5D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 3 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, 3, hc, hs) output: (bs, seqlen, 3, hc/P, hs)
        bs, shard_seqlen, t_cnt, hc, hs = input.shape

        assert t_cnt == 3
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, 3, hc, hs) -reshape-> (bs, seq_len/P, 3, P, hc/P, hs) -transpose(0,3)->
        #  (P, seq_len/P, 3, bs, hc/P, hs)
        input_t = input.reshape(bs, shard_seqlen, 3, seq_world_size, shard_hc, hs).transpose(0, 3).contiguous()

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, 3, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, 3, bs, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                current_omni_platform.synchronize()
        else:
            output = input_t

        # The diffusion FA4 path always uses batch one.  The collective output
        # is already laid out as [destination, local_sequence, 3, batch,
        # local_heads, head_dim], so folding the singleton batch into the
        # leading dimension is a view.  The generic transpose below used to
        # materialize an additional full QKV buffer before the caller split
        # Q/K/V and made their required contiguous copies.
        if bs == 1:
            return output.reshape(1, seqlen, 3, shard_hc, hs)

        # General batched callers retain the original destination-major to
        # batch-major materialization semantics.
        output = output.reshape(seqlen, 3, bs, shard_hc, hs)
        output = output.transpose(0, 2).transpose(1, 2).contiguous()
        return output.reshape(bs, seqlen, 3, shard_hc, hs).contiguous()
    elif scatter_idx == 1 and gather_idx == 3:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, _, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, 3, hc/P, hs) -reshape-> (bs, P, seq_len/P, 3, hc/P, hs) -transpose(0, 4)->
        # (hc/P, P, seqlen/P, 3, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, 3, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, 3, shard_hc, hs)
            .transpose(0, 4)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, 3, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                current_omni_platform.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, 3, bs, hs)

        # (hc, seqlen/N, bs, hs) -transpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 3).contiguous()

        return output.reshape(bs, shard_seqlen, 3, hc, hs).contiguous()
    else:
        raise RuntimeError("scatter_idx must be 1 or 3 and gather_idx must be 1 or 3")


def all_to_all_packed_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    group=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Exchange Q/K/V through one destination-major Ulysses collective.

    This fixed-shape path is intentionally separate from ``all_to_all_4D``:
    H3 has batch one and equal sequence shards, so the three projections can
    share the same communication buffer.  The returned Q/K/V views retain a
    unit stride in the head dimension, which is the layout required by FA4.
    """
    from vllm_omni.diffusion.models.minimax_h3.fused_ops import (
        pack_qkv_destination_major_bf16,
    )

    world_size = dist.get_world_size(group)
    packed = pack_qkv_destination_major_bf16(q, k, v, world_size)
    if packed is None:
        return None

    output = torch.empty_like(packed)
    dist.all_to_all_single(output, packed, group=group)
    # all_to_all_single retains destination/source-major chunks.  Folding
    # the first two dimensions is a view and avoids the old QKV transpose.
    seq_global = q.shape[0] * world_size
    local_heads = q.shape[1] // world_size
    packed = output.reshape(1, seq_global, local_heads, 3 * q.shape[2])
    return packed.split(q.shape[2], dim=-1)


def all_to_all_ulysses_output(
    output: torch.Tensor,
    *,
    group=None,
) -> torch.Tensor | None:
    """Inverse fixed-shape Ulysses collective with a fused output relayout."""
    from vllm_omni.diffusion.models.minimax_h3.fused_ops import (
        merge_ulysses_heads_bf16,
    )

    if output.ndim != 4 or output.shape[0] != 1 or not output.is_cuda:
        return None
    world_size = dist.get_world_size(group)
    if world_size <= 1 or output.dtype != torch.bfloat16:
        return None
    seq_global, local_heads, head_size = output.shape[1:]
    if seq_global % world_size:
        return None

    # [1, S, H/P, D] -> [S, 1, H/P, D].  After the collective, each source
    # rank owns one contiguous local sequence shard.
    input_t = output.permute(1, 0, 2, 3).contiguous()
    exchanged = torch.empty_like(input_t)
    dist.all_to_all_single(exchanged, input_t, group=group)
    seq_local = seq_global // world_size
    exchanged = exchanged.reshape(world_size, seq_local, 1, local_heads, head_size)
    merged = merge_ulysses_heads_bf16(
        exchanged,
        world_size=world_size,
        seq=seq_local,
        local_heads=local_heads,
        head_size=head_size,
    )
    if merged is None:
        return None
    return merged.reshape(1, seq_local, world_size * local_heads, head_size)


class SeqAllToAll5D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int = 3,
        gather_idx: int = 1,
        use_sync: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync

        return all_to_all_5D(input, scatter_idx, gather_idx, group=group, use_sync=use_sync)


class RingComm:
    """Ring communication utility for Ring Attention P2P communication."""

    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(self, to_send: torch.Tensor, recv_tensor: torch.Tensor | None = None) -> torch.Tensor:
        # Ensure to_send is contiguous for P2P
        if not to_send.is_contiguous():
            to_send = to_send.contiguous()

        if recv_tensor is None:
            # Create a contiguous buffer for receiving
            res = torch.empty_like(to_send, memory_format=torch.contiguous_format)
            # print(f"send_recv: empty_like {to_send.shape}")
        else:
            res = recv_tensor
            if not res.is_contiguous():
                res = res.contiguous()

        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []
