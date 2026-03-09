# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization support for diffusion models.

DEPRECATED: This module re-exports from vllm_omni.quantization for backward
compatibility. New code should import from vllm_omni.quantization directly.
"""

# Re-export everything from the new unified location
from vllm_omni.quantization.compat import (  # noqa: F401
    DiffusionFp8Config,
    DiffusionGgufConfig,
    DiffusionQuantizationConfig,
    get_diffusion_quant_config,
    get_vllm_quant_config_for_layers,
)
from vllm_omni.quantization.factory import (  # noqa: F401
    SUPPORTED_QUANTIZATION_METHODS,
)

__all__ = [
    "DiffusionQuantizationConfig",
    "DiffusionFp8Config",
    "DiffusionGgufConfig",
    "get_diffusion_quant_config",
    "get_vllm_quant_config_for_layers",
    "SUPPORTED_QUANTIZATION_METHODS",
]
