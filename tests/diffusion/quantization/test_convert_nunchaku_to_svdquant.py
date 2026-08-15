# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import hashlib
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm_omni.quantization.tools import convert_nunchaku_to_svdquant as converter
from vllm_omni.quantization.tools import svdquant_nvfp4_layout as layout

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("down", "expected_shape", "expected_digest"),
    [
        (
            False,
            (32, 16),
            "44344e9613d3ca5bd360fc9f1a0cd2baf5840dd3fea0cce38e5714c1343b1fde",
        ),
        (
            True,
            (16, 32),
            "901720de4393aa3bad2651492208a651bb2415f1d7895b3b4b844a1a448cd1fb",
        ),
    ],
)
def test_unpack_lowrank_weight_matches_nunchaku_golden(
    down: bool,
    expected_shape: tuple[int, int],
    expected_digest: str,
) -> None:
    packed = torch.arange(32 * 16, dtype=torch.bfloat16).reshape(32, 16)

    unpacked = converter.unpack_lowrank_weight(packed, down=down)
    raw = bytes(unpacked.contiguous().view(torch.uint8).flatten().tolist())

    assert unpacked.shape == expected_shape
    assert hashlib.sha256(raw).hexdigest() == expected_digest


@pytest.mark.parametrize(
    ("size", "expected_digest"),
    [
        (128, "d191c323fa36ac7f763cdb7f448bef790b8a08dca8533a11afec6513253b2144"),
        (256, "f332b85ad192ea70ced60706633cc7808ffd277503dec6a5b77bbf3627718179"),
        (5376, "ea39a087f605481a64b7230029ba64be00948d745efbf4656c309d67c7508e36"),
        (14336, "0e867e585483c578aa10f8f0fb978835678e9a09532a76728e3afe9b112c1637"),
    ],
)
def test_unpack_nunchaku_scale_vector_matches_golden(
    size: int,
    expected_digest: str,
) -> None:
    packed = torch.arange(size, dtype=torch.bfloat16)

    restored = layout.unpack_nunchaku_scale_vector(packed)
    raw = bytes(restored.contiguous().view(torch.uint8).tolist())

    assert hashlib.sha256(raw).hexdigest() == expected_digest


def test_unpack_nvfp4_layer_restores_scale_like_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    n, k, rank = 128, 128, 16
    logical_smooth = torch.arange(k, dtype=torch.bfloat16)
    logical_channel = torch.arange(n, dtype=torch.bfloat16) + 1000
    logical_bias = torch.arange(n, dtype=torch.bfloat16) - 1000
    params = {
        "qweight": torch.zeros(n, k // 2, dtype=torch.int8),
        "wscales": torch.ones(k // 16, n, dtype=torch.float8_e4m3fn),
        "proj_down": torch.zeros(k, rank, dtype=torch.bfloat16),
        "proj_up": torch.zeros(n, rank, dtype=torch.bfloat16),
        "smooth_factor": logical_smooth,
        "wcscales": logical_channel,
        "bias": logical_bias,
    }
    monkeypatch.setattr(converter, "unpack_nunchaku_qweight_fp4", lambda value: value.view(torch.uint8))
    monkeypatch.setattr(converter, "unpack_nunchaku_wscales_fp4", lambda value: value)
    monkeypatch.setattr(converter, "unpack_nunchaku_scale_vector", lambda value: value + 1)
    monkeypatch.setattr(converter, "unpack_lowrank_weight", lambda value, *, down: value)

    restored = converter.unpack_nvfp4_layer(params, half_swap_n=False)

    assert torch.equal(restored["smooth_factor"], logical_smooth + 1)
    assert torch.equal(restored["wcscales"], logical_channel + 1)
    assert torch.equal(restored["bias"], logical_bias + 1)


def test_minimax_h3_fused_gate_up_detection() -> None:
    assert converter.is_fused_gate_up("blocks.0.mlp.fc1")
    assert converter.is_fused_gate_up("token_refiner.blocks.1.mlp.fc1")
    assert not converter.is_fused_gate_up("blocks.0.mlp.fc2")
    assert not converter.is_fused_gate_up("final_layer.adaln_proj.linear")


def test_classify_quantized_layers_distinguishes_svdq_and_w4a16() -> None:
    svdq, w4a16 = converter._classify_quantized_layers(
        {
            "blocks.0.attn.qkv_proj": ["qweight", "proj_down", "wscales"],
            "blocks.0.adaln_proj.linear": ["qweight", "wscales", "wzeros"],
        }
    )
    assert list(svdq) == ["blocks.0.attn.qkv_proj"]
    assert list(w4a16) == ["blocks.0.adaln_proj.linear"]

    with pytest.raises(ValueError, match="exactly one"):
        converter._classify_quantized_layers({"broken": ["qweight"]})


def test_resolve_remote_base_downloads_only_fl2va(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    call: dict[str, object] = {}

    class _Hub:
        def snapshot_download(self, **kwargs):
            call.update(kwargs)
            return str(snapshot)

    monkeypatch.setattr(converter, "hf_api", lambda: _Hub())

    assert converter._resolve_base_pipeline("MiniMaxAI/MiniMax-H3") == snapshot
    assert call == {
        "repo_id": "MiniMaxAI/MiniMax-H3",
        "allow_patterns": ["FL2VA/**"],
    }


def _write_tiny_h3_base(base_root) -> None:
    partition = base_root / "FL2VA"
    transformer = partition / "transformer"
    transformer.mkdir(parents=True)
    (partition / "model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3Pipeline"}),
        encoding="utf-8",
    )
    (transformer / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3DiTModel",
                "hidden_size": 128,
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "blocks.0.mlp.fc1.weight": torch.randn(8, 4),
            "blocks.0.adaln_proj.linear.weight": torch.randn(8, 4),
            "blocks.0.adaln_proj.linear.bias": torch.randn(8),
            "condition_proj.weight": torch.randn(4, 4),
            "condition_proj.bias": torch.randn(4),
            "blocks.0.norm1.weight": torch.randn(4),
        },
        str(transformer / "model.safetensors"),
    )


def _write_tiny_h3_quantized(checkpoint) -> None:
    save_file(
        {
            "blocks.0.mlp.fc1.qweight": torch.zeros(8, 2, dtype=torch.int8),
            "blocks.0.mlp.fc1.wscales": torch.ones(1, 8, dtype=torch.float8_e4m3fn),
            "blocks.0.mlp.fc1.proj_down": torch.zeros(4, 2, dtype=torch.bfloat16),
            "blocks.0.mlp.fc1.proj_up": torch.zeros(8, 2, dtype=torch.bfloat16),
            "blocks.0.mlp.fc1.smooth_factor": torch.ones(4, dtype=torch.bfloat16),
            "blocks.0.adaln_proj.linear.qweight": torch.zeros(2, 32, dtype=torch.int32),
            "blocks.0.adaln_proj.linear.wscales": torch.ones(1, 8, dtype=torch.bfloat16),
            "blocks.0.adaln_proj.linear.wzeros": torch.zeros(1, 8, dtype=torch.bfloat16),
            "blocks.0.adaln_proj.linear.bias": torch.randn(8, dtype=torch.bfloat16),
            # H3 mode must ignore raw Diffusers leftovers.
            "transformer_blocks.0.mlp.fc1.weight": torch.randn(8, 4),
        },
        str(checkpoint),
        metadata={"model_class": "MiniMaxH3Transformer3DModel"},
    )


def test_convert_minimax_h3_mixed_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    base_root = tmp_path / "base"
    checkpoint = tmp_path / "raw.safetensors"
    output = tmp_path / "output"
    _write_tiny_h3_base(base_root)
    _write_tiny_h3_quantized(checkpoint)

    monkeypatch.setattr(converter, "_MINIMAX_H3_EXPECTED_SVDQ_LINEARS", 1)
    monkeypatch.setattr(converter, "_MINIMAX_H3_EXPECTED_W4A16_LINEARS", 1)
    monkeypatch.setattr(
        converter,
        "unpack_nvfp4_layer",
        lambda params, *, half_swap_n: dict(params),
    )

    converter.convert(
        checkpoint,
        base_root,
        output,
        prefer_copy=False,
        progress=False,
    )

    out_transformer = output / "FL2VA" / "transformer"
    config = json.loads((out_transformer / "config.json").read_text())
    quant_config = config["quantization_config"]
    assert quant_config["quant_method"] == "svdquant"
    assert quant_config["rank"] == 2
    assert quant_config["precision"] == "nvfp4"
    assert "w4a16_modules" not in quant_config
    assert quant_config["modules_to_not_convert"] == [
        "blocks.0.adaln_proj.linear",
        "condition_proj",
        "final_layer.adaln_proj.linear",
    ]

    out_weights = out_transformer / "diffusion_pytorch_model.safetensors"
    with safe_open(str(out_weights), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        metadata = f.metadata() or {}
        dense_adaln_weight = f.get_tensor("blocks.0.adaln_proj.linear.weight")
    assert "blocks.0.mlp.fc1.qweight" in keys
    assert "blocks.0.mlp.fc1.weight" not in keys
    assert "blocks.0.adaln_proj.linear.qweight" not in keys
    assert dense_adaln_weight.shape == (8, 4)
    assert "blocks.0.adaln_proj.linear.bias" in keys
    assert "condition_proj.weight" in keys
    assert "blocks.0.norm1.weight" in keys
    assert not any(key.startswith("transformer_blocks.") for key in keys)
    conversion = json.loads(metadata["conversion"])
    assert conversion["model_family"] == "minimax_h3"
    assert conversion["svdq_linears"] == 1
    assert conversion["dense_restored_linears"] == 1
    assert conversion["half_swapped_layers"] == ["blocks.0.mlp.fc1"]
