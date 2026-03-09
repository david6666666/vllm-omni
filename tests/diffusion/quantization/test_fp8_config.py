# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FP8 quantization config."""

import pytest


def test_fp8_config_creation():
    """Test that FP8 config can be created via legacy API."""
    from vllm_omni.diffusion.quantization import get_diffusion_quant_config

    config = get_diffusion_quant_config("fp8")
    assert config is not None
    assert config.get_name() == "fp8"


def test_vllm_config_extraction():
    """Test that vLLM config can be extracted from diffusion config."""
    from vllm_omni.diffusion.quantization import (
        get_diffusion_quant_config,
        get_vllm_quant_config_for_layers,
    )

    diff_config = get_diffusion_quant_config("fp8")
    vllm_config = get_vllm_quant_config_for_layers(diff_config)
    assert vllm_config is not None
    assert vllm_config.activation_scheme == "dynamic"


def test_none_quantization():
    """Test that None quantization returns None config."""
    from vllm_omni.diffusion.quantization import (
        get_diffusion_quant_config,
        get_vllm_quant_config_for_layers,
    )

    config = get_diffusion_quant_config(None)
    assert config is None
    vllm_config = get_vllm_quant_config_for_layers(config)
    assert vllm_config is None


def test_invalid_quantization():
    """Test that invalid quantization method raises error."""
    from vllm_omni.diffusion.quantization import get_diffusion_quant_config

    with pytest.raises(ValueError, match="Unknown quantization method"):
        get_diffusion_quant_config("invalid_method")


def test_fp8_config_with_custom_params():
    """Test FP8 config with custom parameters via legacy API."""
    from vllm_omni.diffusion.quantization import get_diffusion_quant_config

    config = get_diffusion_quant_config(
        "fp8",
        activation_scheme="static",
        ignored_layers=["proj_out"],
    )
    assert config is not None
    assert config.activation_scheme == "static"
    assert "proj_out" in config.ignored_layers


def test_supported_methods():
    """Test that supported methods list is correct."""
    from vllm_omni.diffusion.quantization import SUPPORTED_QUANTIZATION_METHODS

    assert "fp8" in SUPPORTED_QUANTIZATION_METHODS


def test_quantization_integration():
    """Test end-to-end quantization flow through OmniDiffusionConfig."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    # Test with quantization string only
    config = OmniDiffusionConfig(model="test", quantization="fp8")
    assert config.quantization_config is not None
    assert config.quantization_config.get_name() == "fp8"

    # Test with quantization_config dict
    config2 = OmniDiffusionConfig(
        model="test",
        quantization_config={"method": "fp8", "activation_scheme": "static"},
    )
    assert config2.quantization_config is not None
    assert config2.quantization_config.get_name() == "fp8"
    assert config2.quantization_config.activation_scheme == "static"


def test_quantization_dict_not_mutated():
    """Test that passing a dict to quantization_config doesn't mutate it."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    original_dict = {"method": "fp8", "activation_scheme": "static"}
    dict_copy = original_dict.copy()

    OmniDiffusionConfig(model="test", quantization_config=original_dict)

    # Original dict should be unchanged
    assert original_dict == dict_copy


def test_quantization_conflicting_methods_warning(caplog):
    """Test that both quantization and quantization_config['method'] work together."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(
        model="test",
        quantization="fp8",
        quantization_config={"method": "fp8", "activation_scheme": "static"},
    )
    assert config.quantization_config is not None


def test_fp8_delegates_to_vllm_config():
    """Test that DiffusionFp8Config delegates to vLLM's Fp8Config."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    from vllm_omni.diffusion.quantization import DiffusionFp8Config

    # Test that quant_config_cls is set correctly
    assert DiffusionFp8Config.quant_config_cls is Fp8Config

    # Test that get_name() delegates to vLLM
    assert DiffusionFp8Config.get_name() == Fp8Config.get_name()

    # Test that get_min_capability() delegates to vLLM
    assert DiffusionFp8Config.get_min_capability() == Fp8Config.get_min_capability()


# ==================== New API tests ====================


def test_build_quant_config_string():
    """Test build_quant_config with string spec."""
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config("fp8")
    assert config is not None
    assert config.get_name() == "fp8"

    config = build_quant_config(None)
    assert config is None

    config = build_quant_config("none")
    assert config is None


def test_build_quant_config_dict():
    """Test build_quant_config with dict spec."""
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config({"method": "fp8", "activation_scheme": "static"})
    assert config is not None
    assert config.get_name() == "fp8"
    assert config.activation_scheme == "static"


def test_build_quant_config_per_component():
    """Test build_quant_config with per-component dict."""
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


def test_build_quant_config_passthrough():
    """Test build_quant_config passthrough of existing QuantizationConfig."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    from vllm_omni.quantization import build_quant_config

    fp8 = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    config = build_quant_config(fp8)
    assert config is fp8


def test_build_quant_config_invalid():
    """Test build_quant_config with invalid method."""
    from vllm_omni.quantization import build_quant_config

    with pytest.raises(ValueError, match="Unknown quantization method"):
        build_quant_config("invalid_method")


def test_component_config_routing():
    """Test ComponentQuantizationConfig prefix routing."""

    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    from vllm_omni.quantization import ComponentQuantizationConfig

    fp8 = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    config = ComponentQuantizationConfig(
        component_configs={"transformer": fp8, "vae": None},
    )

    assert config.get_name() == "component"

    # transformer prefix should resolve to fp8
    assert config._resolve("transformer.blocks.0.attn") is fp8

    # vae prefix should resolve to None
    assert config._resolve("vae.encoder.conv_in") is None

    # unknown prefix should resolve to default (None)
    assert config._resolve("unknown.layer") is None


def test_gguf_config():
    """Test DiffusionGGUFConfig creation."""
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


def test_get_vllm_quant_config_for_layers_new_api():
    """Test that get_vllm_quant_config_for_layers works with new QuantizationConfig."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    from vllm_omni.quantization.compat import get_vllm_quant_config_for_layers

    fp8 = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")

    # New API: QuantizationConfig passthrough
    result = get_vllm_quant_config_for_layers(fp8)
    assert result is fp8

    # None passthrough
    result = get_vllm_quant_config_for_layers(None)
    assert result is None
