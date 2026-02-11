# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from .base import GGUFAdapter
from .flux2 import Flux2GGUFAdapter


def get_gguf_adapter(gguf_file: str, model, source, od_config) -> GGUFAdapter:
    for adapter_cls in (Flux2GGUFAdapter,):
        if adapter_cls.is_compatible(od_config, model, source):
            return adapter_cls(gguf_file, model, source, od_config)
    return GGUFAdapter(gguf_file, model, source, od_config)


__all__ = ["GGUFAdapter", "Flux2GGUFAdapter", "get_gguf_adapter"]
