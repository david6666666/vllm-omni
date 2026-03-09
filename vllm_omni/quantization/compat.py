# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backward compatibility shims for the old diffusion quantization API.

These are thin wrappers that delegate to the new unified API.
They exist so that existing code (pipelines, configs, tests) continues
to work during the migration period.

Deprecated — will be removed once all callers are migrated to the new API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from .factory import build_quant_config
from .gguf_config import DiffusionGGUFConfig

if TYPE_CHECKING:
    pass


class DiffusionQuantizationConfig:
    """Deprecated: Use build_quant_config() instead.

    This shim wraps a vLLM QuantizationConfig and provides the old API
    (get_name, get_vllm_quant_config) for backward compatibility.
    """

    def __init__(self, vllm_config: QuantizationConfig | None = None) -> None:
        self._vllm_config = vllm_config

    def get_name(self) -> str:
        if self._vllm_config is not None:
            return self._vllm_config.get_name()
        raise NotImplementedError

    def get_vllm_quant_config(self) -> QuantizationConfig | None:
        return self._vllm_config


class DiffusionFp8Config(DiffusionQuantizationConfig):
    """Deprecated: Use build_quant_config("fp8", ...) instead."""

    quant_config_cls = Fp8Config

    def __init__(
        self,
        activation_scheme: str = "dynamic",
        weight_block_size: list[int] | None = None,
        ignored_layers: list[str] | None = None,
    ) -> None:
        self.activation_scheme = activation_scheme
        self.weight_block_size = weight_block_size
        self.ignored_layers = ignored_layers or []
        vllm_config = Fp8Config(
            is_checkpoint_fp8_serialized=False,
            activation_scheme=activation_scheme,
            weight_block_size=weight_block_size,
            ignored_layers=ignored_layers,
        )
        super().__init__(vllm_config)

    @classmethod
    def get_name(cls) -> str:
        return Fp8Config.get_name()

    @classmethod
    def get_min_capability(cls) -> int:
        return Fp8Config.get_min_capability()


class DiffusionGgufConfig(DiffusionQuantizationConfig):
    """Deprecated: Use build_quant_config("gguf", ...) instead."""

    def __init__(
        self,
        gguf_model: str | None = None,
        unquantized_modules: list[str] | None = None,
    ) -> None:
        self.gguf_model = gguf_model
        vllm_config = DiffusionGGUFConfig(
            gguf_model=gguf_model,
            unquantized_modules=unquantized_modules,
        )
        super().__init__(vllm_config)


def get_diffusion_quant_config(
    quantization: str | None,
    **kwargs: Any,
) -> DiffusionQuantizationConfig | None:
    """Deprecated: Use build_quant_config() instead."""
    if quantization is None or quantization.lower() == "none":
        return None

    config = build_quant_config(quantization, **kwargs)
    if config is None:
        return None

    # Wrap in compat shim
    wrapper = DiffusionQuantizationConfig(config)
    # Preserve gguf_model if present
    if hasattr(config, "gguf_model"):
        wrapper.gguf_model = config.gguf_model
    return wrapper


def get_vllm_quant_config_for_layers(
    diffusion_quant_config: Any,
) -> QuantizationConfig | None:
    """Extract vLLM QuantizationConfig from either old or new config types.

    Handles:
    - None -> None
    - QuantizationConfig (new API) -> passthrough
    - DiffusionQuantizationConfig (old API) -> unwrap
    """
    if diffusion_quant_config is None:
        return None

    # New API: already a QuantizationConfig
    if isinstance(diffusion_quant_config, QuantizationConfig):
        return diffusion_quant_config

    # Old API: DiffusionQuantizationConfig wrapper
    if hasattr(diffusion_quant_config, "get_vllm_quant_config"):
        return diffusion_quant_config.get_vllm_quant_config()

    raise TypeError(
        f"Expected QuantizationConfig or DiffusionQuantizationConfig, got {type(diffusion_quant_config).__name__}"
    )
