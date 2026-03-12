# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable

import torch

from .base import GGUFAdapter, gguf_quant_weights_iterator


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

    def _get_alias_names(self, names: set[str]) -> set[str]:
        """Return GGUF aliases consumed by QwenImageTransformer.load_weights()."""
        virtual_names = set()
        for name in names:
            if ".to_qkv." in name:
                virtual_names.add(name.replace(".to_qkv.", ".to_q."))
                virtual_names.add(name.replace(".to_qkv.", ".to_k."))
                virtual_names.add(name.replace(".to_qkv.", ".to_v."))
            if ".add_kv_proj." in name:
                virtual_names.add(name.replace(".add_kv_proj.", ".add_q_proj."))
                virtual_names.add(name.replace(".add_kv_proj.", ".add_k_proj."))
                virtual_names.add(name.replace(".add_kv_proj.", ".add_v_proj."))
            if ".to_out." in name:
                virtual_names.add(name.replace(".to_out.", ".to_out.0."))
        return virtual_names

    def weights_iterator(self) -> Iterable[tuple[str, torch.Tensor]]:
        prefix = getattr(self.source, "prefix", "")
        target = self.model.get_submodule(prefix.rstrip(".")) if prefix else self.model
        loadable_names = {name for name, _ in target.named_parameters()}
        loadable_names.update(name for name, _ in target.named_buffers())
        loadable_names.update(self._get_alias_names(loadable_names))
        weights = gguf_quant_weights_iterator(self.gguf_file)
        for name, tensor in weights:
            if name in loadable_names:
                yield name, tensor
