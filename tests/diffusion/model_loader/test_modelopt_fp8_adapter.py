# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.model_loader.checkpoint_adapters import (
    ModelOptFp8CheckpointAdapter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _PackedModelOptModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.block = nn.Module()
        self.transformer.block.to_qkv = nn.Linear(2, 2, bias=False)
        self.transformer.block.bias = nn.Parameter(torch.empty(2, dtype=torch.float32))


class _QuantizedPackedModelOptModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.block = nn.Module()
        self.transformer.block.to_qkv = nn.Module()
        self.transformer.block.to_qkv.register_parameter(
            "weight",
            nn.Parameter(torch.empty(2, 2, dtype=torch.float8_e4m3fn), requires_grad=False),
        )
        self.transformer.block.to_qkv.register_parameter(
            "weight_scale",
            nn.Parameter(torch.empty(1), requires_grad=False),
        )
        self.transformer.block.to_qkv.register_parameter(
            "input_scale",
            nn.Parameter(torch.empty(1), requires_grad=False),
        )


def _make_source() -> SimpleNamespace:
    return SimpleNamespace(
        subfolder="transformer",
        prefix="transformer.",
    )


def _make_unscaled_fp8_quant_config() -> SimpleNamespace:
    return SimpleNamespace(_omni_unscaled_fp8_checkpoint=True)


def test_modelopt_adapter_dequantizes_fp8_weight_for_full_precision_target():
    model = _PackedModelOptModel()
    adapter = ModelOptFp8CheckpointAdapter(model, _make_source())
    fp8_weight = torch.tensor([[2.0, -4.0], [1.0, 3.0]], dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.5], dtype=torch.float32)

    adapted = list(
        adapter.adapt(
            iter(
                [
                    ("transformer.block.to_q.weight_scale", scale),
                    ("transformer.block.to_q.input_scale", torch.tensor([1.0])),
                    ("transformer.block.to_q.weight", fp8_weight),
                ]
            )
        )
    )

    assert [name for name, _ in adapted] == ["transformer.block.to_q.weight"]
    assert adapted[0][1].dtype == model.transformer.block.to_qkv.weight.dtype
    assert torch.allclose(adapted[0][1], fp8_weight.to(torch.float32) * scale)


def test_modelopt_adapter_requires_scale_for_fp8_weight_by_default():
    model = _PackedModelOptModel()
    adapter = ModelOptFp8CheckpointAdapter(model, _make_source())
    fp8_weight = torch.tensor([[2.0, -4.0], [1.0, 3.0]], dtype=torch.float32).to(torch.float8_e4m3fn)

    with pytest.raises(ValueError, match="Missing ModelOpt FP8 weight_scale"):
        list(adapter.adapt(iter([("transformer.block.to_q.weight", fp8_weight)])))


def test_modelopt_adapter_casts_unscaled_fp8_weight_for_auto_detected_checkpoint():
    model = _PackedModelOptModel()
    adapter = ModelOptFp8CheckpointAdapter(model, _make_source(), _make_unscaled_fp8_quant_config())
    fp8_weight = torch.tensor([[2.0, -4.0], [1.0, 3.0]], dtype=torch.float32).to(torch.float8_e4m3fn)

    adapted = list(adapter.adapt(iter([("transformer.block.to_q.weight", fp8_weight)])))

    assert [name for name, _ in adapted] == ["transformer.block.to_q.weight"]
    assert adapted[0][1].dtype == model.transformer.block.to_qkv.weight.dtype
    torch.testing.assert_close(adapted[0][1], fp8_weight.to(torch.float32))


def test_modelopt_adapter_initializes_missing_scales_for_auto_detected_checkpoint():
    model = _QuantizedPackedModelOptModel()
    adapter = ModelOptFp8CheckpointAdapter(model, _make_source(), _make_unscaled_fp8_quant_config())
    fp8_weight = torch.tensor([[2.0, -4.0], [1.0, 3.0]], dtype=torch.float32).to(torch.float8_e4m3fn)

    adapted = list(adapter.adapt(iter([("transformer.block.to_q.weight", fp8_weight)])))

    assert [name for name, _ in adapted] == [
        "transformer.block.to_q.weight",
        "transformer.block.to_qkv.weight_scale",
        "transformer.block.to_qkv.input_scale",
    ]
    assert torch.equal(adapted[1][1], torch.ones_like(model.transformer.block.to_qkv.weight_scale))
    assert torch.equal(adapted[2][1], torch.ones_like(model.transformer.block.to_qkv.input_scale))


def test_modelopt_adapter_casts_unscaled_fp8_non_weight_tensors():
    model = _PackedModelOptModel()
    adapter = ModelOptFp8CheckpointAdapter(model, _make_source())
    fp8_bias = torch.tensor([2.0, -4.0], dtype=torch.float32).to(torch.float8_e4m3fn)

    adapted = list(adapter.adapt(iter([("transformer.block.bias", fp8_bias)])))

    assert [name for name, _ in adapted] == ["transformer.block.bias"]
    assert adapted[0][1].dtype == model.transformer.block.bias.dtype
    torch.testing.assert_close(adapted[0][1], fp8_bias.to(torch.float32))


def test_modelopt_adapter_keeps_scale_tensors_for_quantized_target():
    model = _QuantizedPackedModelOptModel()
    adapter = ModelOptFp8CheckpointAdapter(model, _make_source())
    scale = torch.tensor([0.5], dtype=torch.float32)

    adapted = list(
        adapter.adapt(
            iter(
                [
                    ("transformer.block.to_q.weight_scale", scale),
                    ("transformer.block.to_q.input_scale", torch.tensor([1.0])),
                ]
            )
        )
    )

    assert [name for name, _ in adapted] == [
        "transformer.block.to_q.weight_scale",
        "transformer.block.to_q.input_scale",
    ]
