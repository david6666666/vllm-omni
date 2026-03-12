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

    def _get_loadable_names(self) -> set[str]:
        """Return GGUF tensor names that this transformer can actually consume.

        Qwen-Image GGUF checkpoints include tensors for modules that stay dense in
        vllm-omni, such as the modulation MLP heads inside ``img_mod``/``txt_mod``.
        Those weights must fall back to the base HF checkpoint instead of being
        routed through the GGUF loader.

        The model also packs attention projections as ``to_qkv`` and
        ``add_kv_proj``, while GGUF exports split shard names. Expose those split
        aliases here so ``QwenImageTransformer.load_weights()`` can merge them via
        its packed shard loaders.
        """
        prefix = getattr(self.source, "prefix", "")
        target = self.model.get_submodule(prefix.rstrip(".")) if prefix else self.model
        loadable_names = {name for name, _ in target.named_parameters()}
        loadable_names.update(name for name, _ in target.named_buffers())

        virtual_names = set()
        for name in loadable_names:
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
        loadable_names.update(virtual_names)
        return loadable_names

    def weights_iterator(self) -> Iterable[tuple[str, torch.Tensor]]:
        loadable_names = self._get_loadable_names()
        weights = gguf_quant_weights_iterator(self.gguf_file)
        for name, tensor in weights:
            if name in loadable_names:
                yield name, tensor
