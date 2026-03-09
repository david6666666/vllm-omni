# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factory for building quantization configs.

build_quant_config() accepts flexible inputs:
  - None or "none"  -> None (no quantization)
  - "fp8"           -> Fp8Config with diffusion defaults
  - "gguf"          -> DiffusionGGUFConfig
  - dict            -> Parsed per-component or single-method config
  - QuantizationConfig instance -> passthrough

Examples:
    # Simple string
    config = build_quant_config("fp8")

    # Dict with method + params
    config = build_quant_config({"method": "fp8", "activation_scheme": "dynamic"})

    # Per-component (for multi-stage models)
    config = build_quant_config({
        "transformer": {"method": "fp8"},
        "vae": None,
    })

    # GGUF with model path
    config = build_quant_config({
        "method": "gguf",
        "gguf_model": "path/to/model.gguf",
    })
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from .component_config import ComponentQuantizationConfig
from .gguf_config import DiffusionGGUFConfig

logger = init_logger(__name__)


def _build_fp8(
    is_checkpoint_fp8_serialized: bool = False,
    activation_scheme: str = "dynamic",
    weight_block_size: list[int] | None = None,
    ignored_layers: list[str] | None = None,
    **_extra: Any,
) -> Fp8Config:
    """Build FP8 config with diffusion-friendly defaults."""
    return Fp8Config(
        is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
        activation_scheme=activation_scheme,
        weight_block_size=weight_block_size,
        ignored_layers=ignored_layers,
    )


def _build_gguf(
    gguf_model: str | None = None,
    unquantized_modules: list[str] | None = None,
    **_extra: Any,
) -> DiffusionGGUFConfig:
    """Build GGUF config for diffusion models."""
    return DiffusionGGUFConfig(
        gguf_model=gguf_model,
        unquantized_modules=unquantized_modules,
    )


# Registry of quantization method builders.
# Each builder accepts **kwargs and returns a QuantizationConfig.
# To add a new method: implement a _build_X function and register it here.
_QUANT_BUILDERS: dict[str, Any] = {
    "fp8": _build_fp8,
    "gguf": _build_gguf,
}

SUPPORTED_QUANTIZATION_METHODS = list(_QUANT_BUILDERS.keys())


def _build_single(method: str, **kwargs: Any) -> QuantizationConfig:
    """Build a single quantization config by method name."""
    method = method.lower()
    if method not in _QUANT_BUILDERS:
        raise ValueError(f"Unknown quantization method: {method!r}. Supported: {SUPPORTED_QUANTIZATION_METHODS}")
    return _QUANT_BUILDERS[method](**kwargs)


def _is_per_component_dict(spec: dict[str, Any]) -> bool:
    """Check if a dict spec describes per-component quantization.

    Per-component dicts have component prefixes as keys with dict/None/str values.
    Single-method dicts have a "method" key.
    """
    if "method" in spec:
        return False
    # If all values are dicts, None, or strings -> per-component
    return all(isinstance(v, (dict, str, type(None))) for v in spec.values())


def build_quant_config(
    spec: str | dict[str, Any] | QuantizationConfig | None,
    **kwargs: Any,
) -> QuantizationConfig | None:
    """Build a quantization config from a flexible specification.

    Args:
        spec: One of:
            - None or "none": No quantization
            - str: Method name (e.g., "fp8", "gguf")
            - dict with "method" key: Single method with params
            - dict without "method" key: Per-component config
            - QuantizationConfig instance: Passthrough
        **kwargs: Extra params merged with dict spec (for backward compat)

    Returns:
        QuantizationConfig or None
    """
    if spec is None:
        return None

    # Passthrough existing config instances
    if isinstance(spec, QuantizationConfig):
        return spec

    # String spec: "fp8", "gguf", "none"
    if isinstance(spec, str):
        if spec.lower() == "none":
            return None
        logger.info("Building quantization config: %s", spec)
        return _build_single(spec, **kwargs)

    # Dict spec
    if isinstance(spec, Mapping):
        spec = dict(spec)  # Handle DictConfig etc.

        # Per-component config
        if _is_per_component_dict(spec):
            return _build_component_config(spec)

        # Single method with params
        method = spec.pop("method", None)
        if method is None:
            raise ValueError(
                "Dict quantization config must have a 'method' key or "
                "be a per-component config with component prefixes as keys."
            )
        merged = {**spec, **kwargs}
        logger.info("Building quantization config: %s", method)
        return _build_single(method, **merged)

    raise TypeError(f"quantization config must be str, dict, QuantizationConfig, or None, got {type(spec).__name__}")


def _build_component_config(
    spec: dict[str, Any],
) -> ComponentQuantizationConfig:
    """Build a ComponentQuantizationConfig from a per-component dict.

    Example input:
        {"transformer": {"method": "fp8"}, "vae": None}
        {"transformer": "fp8", "vae": None}
    """
    component_configs: dict[str, QuantizationConfig | None] = {}
    default_config: QuantizationConfig | None = None

    for prefix, value in spec.items():
        if value is None:
            config = None
        elif isinstance(value, str):
            config = _build_single(value)
        elif isinstance(value, dict):
            method = value.pop("method", None)
            if method is None:
                raise ValueError(f"Component '{prefix}' config dict must have a 'method' key")
            config = _build_single(method, **value)
        else:
            raise TypeError(f"Component '{prefix}' config must be str, dict, or None, got {type(value).__name__}")

        if prefix == "default":
            default_config = config
        else:
            component_configs[prefix] = config

    logger.info(
        "Building per-component quantization: %s",
        {k: (v.get_name() if v else None) for k, v in component_configs.items()},
    )
    return ComponentQuantizationConfig(component_configs, default_config)
