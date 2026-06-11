# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json

import pytest

from examples.quantization.quantize_cosmos3_nano_modelopt_nvfp4 import (
    _cosmos3_quant_config_block,
    _inverse_remap_cosmos3_key,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("internal", "expected"),
    [
        ("language_model.layers.3.self_attn.to_q.weight", "layers.3.self_attn.to_q.weight"),
        (
            "language_model.layers.3.self_attn.to_q.weight_scale_2",
            "layers.3.self_attn.to_q.weight_scale_2",
        ),
        ("language_model.layers.3.mlp.down_proj.input_scale", "layers.3.mlp.down_proj.input_scale"),
        ("gen_layers.4.cross_attention.to_q.weight", "layers.4.self_attn.add_q_proj.weight"),
        ("gen_layers.4.cross_attention.to_out.weight_scale", "layers.4.self_attn.to_add_out.weight_scale"),
        ("gen_layers.4.mlp.up_proj.weight", "layers.4.mlp_moe_gen.up_proj.weight"),
        ("proj_in.weight", "proj_in.weight"),
        ("time_embedder.linear_1.weight", "time_embedder.linear_1.weight"),
        ("language_model.embed_tokens.weight", "embed_tokens.weight"),
        ("norm_moe_gen.weight", "norm_moe_gen.weight"),
    ],
)
def test_cosmos3_modelopt_export_inverse_key_mapping(internal: str, expected: str) -> None:
    assert _inverse_remap_cosmos3_key(internal) == expected


def test_cosmos3_nvfp4_quant_config_is_modelopt_fp4_serialized() -> None:
    config = _cosmos3_quant_config_block(mlp_only=False)

    assert config["quant_method"] == "modelopt_fp4"
    assert config["quant_algo"] == "NVFP4"
    assert config["is_checkpoint_nvfp4_serialized"] is True
    assert config["group_size"] == 16
    assert "proj_out*" in config["ignore"]
    assert "time_embedder*" in config["ignore"]

    # Keep the block JSON-serializable because it is written into config.json.
    json.dumps(config)
