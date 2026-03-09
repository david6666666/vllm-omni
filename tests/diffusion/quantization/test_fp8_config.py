# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the unified quantization framework."""

import pytest

# ---------------------------------------------------------------------------
# build_quant_config — string specs
# ---------------------------------------------------------------------------


def test_build_quant_config_fp8():
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config("fp8")
    assert config is not None
    assert config.get_name() == "fp8"
    assert config.activation_scheme == "dynamic"


def test_build_quant_config_none():
    from vllm_omni.quantization import build_quant_config

    assert build_quant_config(None) is None
    assert build_quant_config("none") is None


def test_build_quant_config_invalid():
    from vllm_omni.quantization import build_quant_config

    with pytest.raises(ValueError, match="Unknown quantization method"):
        build_quant_config("invalid_method")


# ---------------------------------------------------------------------------
# build_quant_config — dict specs
# ---------------------------------------------------------------------------


def test_build_quant_config_dict():
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config({"method": "fp8", "activation_scheme": "static"})
    assert config is not None
    assert config.get_name() == "fp8"
    assert config.activation_scheme == "static"


def test_build_quant_config_dict_not_mutated():
    from vllm_omni.quantization import build_quant_config

    original = {"method": "fp8", "activation_scheme": "static"}
    copy = original.copy()
    build_quant_config(original)
    assert original == copy


# ---------------------------------------------------------------------------
# build_quant_config — per-component specs
# ---------------------------------------------------------------------------


def test_build_quant_config_per_component():
    from vllm_omni.quantization import ComponentQuantizationConfig, build_quant_config

    config = build_quant_config(
        {
            "transformer": {"method": "fp8"},
            "vae": None,
        }
    )
    assert isinstance(config, ComponentQuantizationConfig)
    assert config.component_configs["transformer"].get_name() == "fp8"
    assert config.component_configs["vae"] is None


def test_build_quant_config_per_component_string():
    from vllm_omni.quantization import ComponentQuantizationConfig, build_quant_config

    config = build_quant_config({"transformer": "fp8", "vae": None})
    assert isinstance(config, ComponentQuantizationConfig)
    assert config.component_configs["transformer"].get_name() == "fp8"


# ---------------------------------------------------------------------------
# build_quant_config — passthrough
# ---------------------------------------------------------------------------


def test_build_quant_config_passthrough():
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    from vllm_omni.quantization import build_quant_config

    fp8 = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    assert build_quant_config(fp8) is fp8


# ---------------------------------------------------------------------------
# ComponentQuantizationConfig
# ---------------------------------------------------------------------------


def test_component_config_routing():
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    from vllm_omni.quantization import ComponentQuantizationConfig

    fp8 = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    config = ComponentQuantizationConfig(
        component_configs={"transformer": fp8, "vae": None},
    )

    assert config.get_name() == "component"
    assert config._resolve("transformer.blocks.0.attn") is fp8
    assert config._resolve("vae.encoder.conv_in") is None
    assert config._resolve("unknown.layer") is None


def test_component_config_with_default():
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    from vllm_omni.quantization import ComponentQuantizationConfig

    fp8 = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    config = ComponentQuantizationConfig(
        component_configs={"vae": None},
        default_config=fp8,
    )

    assert config._resolve("transformer.blocks.0") is fp8
    assert config._resolve("vae.encoder") is None


# ---------------------------------------------------------------------------
# GGUF config
# ---------------------------------------------------------------------------


def test_gguf_config():
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.gguf_config import DiffusionGGUFConfig

    config = build_quant_config(
        {
            "method": "gguf",
            "gguf_model": "path/to/model.gguf",
        }
    )
    assert isinstance(config, DiffusionGGUFConfig)
    assert config.gguf_model == "path/to/model.gguf"
    assert config.get_name() == "gguf"


# ---------------------------------------------------------------------------
# OmniDiffusionConfig integration
# ---------------------------------------------------------------------------


def test_integration_string():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(model="test", quantization="fp8")
    assert config.quantization_config is not None
    assert config.quantization_config.get_name() == "fp8"


def test_integration_dict():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(
        model="test",
        quantization_config={"method": "fp8", "activation_scheme": "static"},
    )
    assert config.quantization_config is not None
    assert config.quantization_config.get_name() == "fp8"
    assert config.quantization_config.activation_scheme == "static"


def test_integration_no_quant():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(model="test")
    assert config.quantization_config is None


# ---------------------------------------------------------------------------
# Supported methods
# ---------------------------------------------------------------------------


def test_supported_methods_includes_vllm():
    from vllm_omni.quantization import SUPPORTED_QUANTIZATION_METHODS

    # Must include core vLLM methods
    for method in ["fp8", "gguf", "awq", "gptq", "bitsandbytes"]:
        assert method in SUPPORTED_QUANTIZATION_METHODS, f"{method} missing"


def test_supported_methods_count():
    from vllm_omni.quantization import SUPPORTED_QUANTIZATION_METHODS

    # vLLM has 35+ methods
    assert len(SUPPORTED_QUANTIZATION_METHODS) >= 30
