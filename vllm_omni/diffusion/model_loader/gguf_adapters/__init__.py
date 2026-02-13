# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import GGUFAdapter
from .flux2_klein import Flux2KleinGGUFAdapter
from .qwen_image import QwenImageGGUFAdapter
from .z_image import ZImageGGUFAdapter

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.model_loader.diffusers_loader import (
        DiffusersPipelineLoader,
    )


def get_gguf_adapter(
    gguf_file: str,
    model: torch.nn.Module,
    source: DiffusersPipelineLoader.ComponentSource,
    od_config: OmniDiffusionConfig,
) -> GGUFAdapter:
    for adapter_cls in (QwenImageGGUFAdapter, ZImageGGUFAdapter, Flux2KleinGGUFAdapter):
        if adapter_cls.is_compatible(od_config, model, source):
            return adapter_cls(gguf_file, model, source, od_config)
    return GGUFAdapter(gguf_file, model, source, od_config)


__all__ = [
    "GGUFAdapter",
    "Flux2KleinGGUFAdapter",
    "QwenImageGGUFAdapter",
    "ZImageGGUFAdapter",
    "get_gguf_adapter",
]
