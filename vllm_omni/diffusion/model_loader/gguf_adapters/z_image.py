# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable

import torch
from vllm.model_executor.models.utils import WeightsMapper

from .base import GGUFAdapter, gguf_quant_weights_iterator


Z_IMAGE_KEYS_RENAME_DICT = {
    "final_layer.": "all_final_layer.2-1.",
    "x_embedder.": "all_x_embedder.2-1.",
    ".attention.out.bias": ".attention.to_out.0.bias",
    ".attention.k_norm": ".attention.norm_k.weight",
    ".attention.q_norm": ".attention.norm_q.weight",
    ".attention.out.weight": ".attention.to_out.0.weight",
    "model.diffusion_model.": "",
}


class ZImageGGUFAdapter(GGUFAdapter):
    """GGUF adapter for Z-Image models with QKV/FFN shard support."""

    gguf_to_hf_mapper = WeightsMapper(
        orig_to_new_substr=Z_IMAGE_KEYS_RENAME_DICT,
    )

    def weights_iterator(self) -> Iterable[tuple[str, torch.Tensor]]:
        weights = gguf_quant_weights_iterator(self.gguf_file)
        yield from self.gguf_to_hf_mapper.apply(weights)
