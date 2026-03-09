# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable

import torch
from vllm.model_executor.models.utils import WeightsMapper

from .base import GGUFAdapter, gguf_quant_weights_iterator

QWEN_IMAGE_KEYS_RENAME_DICT = {
    ".to_q.": ".to_qkv.",
    ".to_k.": ".to_qkv.",
    ".to_v.": ".to_qkv.",
    ".add_q_proj.": ".add_kv_proj.",
    ".add_k_proj.": ".add_kv_proj.",
    ".add_v_proj.": ".add_kv_proj.",
    ".to_out.0.": ".to_out.",
}


class QwenImageGGUFAdapter(GGUFAdapter):
    """GGUF adapter for the Qwen-Image transformer family."""

    @staticmethod
    def is_compatible(od_config, model: torch.nn.Module, source) -> bool:
        model_class = od_config.model_class_name or ""
        if model_class.startswith("QwenImage"):
            return True
        cfg = od_config.tf_model_config
        if cfg is not None:
            model_type = str(cfg.get("model_type", "")).lower()
            if model_type.startswith("qwen_image"):
                return True
        return False

    gguf_to_hf_mapper = WeightsMapper(
        orig_to_new_substr=QWEN_IMAGE_KEYS_RENAME_DICT,
    )

    def weights_iterator(self) -> Iterable[tuple[str, torch.Tensor]]:
        weights = gguf_quant_weights_iterator(self.gguf_file)
        yield from self.gguf_to_hf_mapper.apply(weights)
