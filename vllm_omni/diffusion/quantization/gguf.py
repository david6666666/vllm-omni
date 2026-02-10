# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF quantization config for diffusion transformers."""

from vllm.model_executor.layers.quantization.gguf import GGUFConfig

from .base import DiffusionQuantizationConfig


class DiffusionGgufConfig(DiffusionQuantizationConfig):
    """GGUF quantization config for diffusion transformers.

    This is a thin wrapper around vLLM's GGUFConfig and also carries
    the GGUF model reference for loader use.

    Args:
        gguf_model: GGUF model path or HF reference (repo/file or repo:quant_type)
        unquantized_modules: Optional list of module name patterns to skip GGUF
            quantization. Note: diffusion linear layers often use short prefixes
            (e.g., "to_qkv"), so these patterns are matched as substrings.
    """

    quant_config_cls = GGUFConfig

    def __init__(
        self,
        gguf_model: str | None = None,
        unquantized_modules: list[str] | None = None,
    ) -> None:
        self.gguf_model = gguf_model
        self.unquantized_modules = unquantized_modules or []
        self._vllm_config = GGUFConfig(unquantized_modules=self.unquantized_modules)
