# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified quantization framework for vLLM-OMNI.

This module provides a single entry point for all quantization in vLLM-OMNI,
replacing the previous fragmented approach. It supports:

- All vLLM quantization methods (FP8, GGUF, and future methods)
- Per-component quantization for multi-stage models
- Online (BF16 -> quantize) and offline (pre-quantized checkpoint) modes
- All platforms (CUDA, ROCm, CPU, XPU, NPU)

Quick start:
    from vllm_omni.quantization import build_quant_config

    # Simple: single method
    config = build_quant_config("fp8")

    # Advanced: per-component
    config = build_quant_config({
        "transformer": {"method": "fp8", "activation_scheme": "dynamic"},
        "vae": None,  # skip VAE quantization
    })

    # Pass to vLLM layers
    linear = QKVParallelLinear(..., quant_config=config)
"""

# Backward compatibility: re-export for code that imports from the old location.
# These will be removed once all callers are migrated.
from .compat import (
    DiffusionFp8Config,
    DiffusionGgufConfig,
    DiffusionQuantizationConfig,
    get_diffusion_quant_config,
    get_vllm_quant_config_for_layers,
)
from .component_config import ComponentQuantizationConfig
from .defaults import COMPONENT_SKIP_DEFAULTS, get_default_skip_patterns
from .factory import SUPPORTED_QUANTIZATION_METHODS, build_quant_config
from .gguf_config import DiffusionGGUFConfig, DiffusionGGUFLinearMethod
from .validation import validate_quant_config

__all__ = [
    # New API
    "build_quant_config",
    "ComponentQuantizationConfig",
    "DiffusionGGUFConfig",
    "DiffusionGGUFLinearMethod",
    "validate_quant_config",
    "get_default_skip_patterns",
    "COMPONENT_SKIP_DEFAULTS",
    "SUPPORTED_QUANTIZATION_METHODS",
    # Backward compat (deprecated)
    "DiffusionQuantizationConfig",
    "DiffusionFp8Config",
    "DiffusionGgufConfig",
    "get_diffusion_quant_config",
    "get_vllm_quant_config_for_layers",
]
