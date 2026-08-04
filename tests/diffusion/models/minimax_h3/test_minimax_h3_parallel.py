# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_grouped_qkv_checkpoint_reorder():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        _reorder_grouped_qkv_to_qkv,
    )

    # Two groups with rows [q, k, v] become [q0, q1, k0, k1, v0, v1].
    grouped = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    reordered = _reorder_grouped_qkv_to_qkv(
        grouped,
        num_query_groups=2,
        heads_per_group=1,
        head_dim=1,
    )

    assert reordered[:, 0].tolist() == [0, 3, 1, 4, 2, 5]


def test_transformer_declares_cache_sp_layerwise_offload_and_hsdp():
    from cache_dit import ForwardPattern

    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTModel,
    )

    assert MiniMaxH3DiTModel._repeated_blocks == ["MiniMaxH3DiTBlock"]
    assert MiniMaxH3DiTModel._layerwise_offload_blocks_attrs == ["blocks"]
    assert MiniMaxH3DiTModel._cache_dit_adapter_config.block_forward_patterns["blocks"] == ForwardPattern.Pattern_3
    assert not MiniMaxH3DiTModel._cache_dit_adapter_config.has_separate_cfg
    assert set(MiniMaxH3DiTModel._sp_plan) == {"sp_prepare", "sp_gather"}

    model = object.__new__(MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
    model.token_refiner = nn.Module()
    model.token_refiner.blocks = nn.ModuleList([nn.Linear(4, 4)])
    model.final_layer = nn.Linear(4, 4)

    matched = [
        name
        for name, module in model.named_modules()
        if any(condition(name, module) for condition in MiniMaxH3DiTModel._hsdp_shard_conditions)
    ]
    assert matched == ["blocks.0", "blocks.1"]


def test_packed_attention_is_a_regional_compile_boundary():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3Attention,
    )

    assert getattr(MiniMaxH3Attention._run_packed_attention, "_torchdynamo_disable", False)


def test_mlp_uses_fused_silu_and_mul_activation():
    from vllm.model_executor.layers.activation import SiluAndMul

    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3MLP,
    )

    assert MiniMaxH3MLP.forward.__code__.co_names.count("silu_and_mul") == 1
    assert isinstance(SiluAndMul(), SiluAndMul)


def test_rope_build_cache_matches_raw_frequency_path():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3Rope,
        _apply_rope,
    )

    rope = MiniMaxH3Rope(inv_freq_len=2)
    with torch.no_grad():
        rope.inv_freq.copy_(torch.tensor([1.0, 0.25]))
    positions = torch.tensor([[[0, 1, 2], [1, 2, 3]]], dtype=torch.long)
    raw = rope(positions)
    cache = rope.build_cache(positions)

    assert cache.shape == (2, 2 * raw.shape[-1])
    assert cache.dtype == torch.bfloat16
    x = torch.randn(2, 1, 16, dtype=torch.bfloat16)
    torch.testing.assert_close(
        _apply_rope(x, raw),
        _apply_rope(x, cache),
        atol=0,
        rtol=0,
    )


def test_sequence_parallel_local_span_falls_back_without_initialized_groups():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        _sequence_parallel_local_span,
    )

    assert _sequence_parallel_local_span(64) == (0, 64)


def test_local_embedding_spans_reassemble_to_full_embedding(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    class _IdentityProjection(nn.Module):
        def forward(self, x):
            return x, None

    class _TimeEmbedding(nn.Module):
        def forward(self, x):
            return x

    model = object.__new__(h3.MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.hidden_size = 2
    model.video_patch_proj = _IdentityProjection()
    model.audio_patch_proj = _IdentityProjection()
    model.time_embedder = _TimeEmbedding()

    x = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    audio_x = (x + 100).clone()
    text = torch.tensor([[10.0, 11.0], [12.0, 13.0]])
    common = dict(
        x=x,
        audio_x=audio_x,
        text_embeddings_selected=text,
        unique_timesteps=torch.tensor([0.2, 0.3]),
        img_pos=torch.tensor([2, 4, 6]),
        audio_pos=torch.tensor([3, 5, 7]),
        text_pos=torch.tensor([0, 1]),
        refiner_cu_seqlens=torch.tensor([0, 2, 2], dtype=torch.int32),
        refiner_max_seqlen=2,
        refiner_attn_mask=None,
        seq_len=8,
        device=torch.device("cpu"),
        prompt_embeds_refined=True,
    )

    monkeypatch.setattr(h3, "_sequence_parallel_local_span", lambda seq_len: (0, seq_len))
    full, _ = model._embed(**common)

    spans = []
    for rank in range(4):
        monkeypatch.setattr(h3, "_sequence_parallel_local_span", lambda seq_len, rank=rank: (rank * 2, 2))
        local, _ = model._embed(**common)
        spans.append(local[rank * 2 : (rank + 1) * 2])

    torch.testing.assert_close(torch.cat(spans), full)


@pytest.mark.parametrize(
    ("tp_size", "message"),
    [
        (3, "num_attention_heads"),
        (5, "num_attention_heads"),
    ],
)
def test_tp_rejects_non_divisible_head_counts(tp_size, message):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTModel,
    )

    model = object.__new__(MiniMaxH3DiTModel)
    with pytest.raises(ValueError, match=message):
        model._validate_tp_config(
            arch=MiniMaxH3DiTArchConfig(),
            tp_size=tp_size,
        )


def test_tp_accepts_checkpoint_supported_sizes():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTModel,
    )

    model = object.__new__(MiniMaxH3DiTModel)
    arch = MiniMaxH3DiTArchConfig()
    for tp_size in (1, 2, 4, 7):
        model._validate_tp_config(arch=arch, tp_size=tp_size)
