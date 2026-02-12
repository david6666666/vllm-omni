# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .base import GGUFAdapter


@dataclass
class _MappedTensor:
    name: str
    tensor: Any
    tensor_type: Any


class QwenImageGGUFAdapter(GGUFAdapter):
    """GGUF adapter for Qwen-Image models with QKV shard support."""

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

    def weights_iterator(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        try:
            import gguf  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency error
            raise RuntimeError(
                "GGUF support requires the 'gguf' package to be installed."
            ) from exc

        reader = gguf.GGUFReader(self.gguf_file)
        gguf_name_map = self._build_gguf_name_map(reader)
        allowed_names = self._build_allowed_names()
        param_names = self._build_param_names()
        mapped: list[_MappedTensor] = []

        for tensor in reader.tensors:
            mapped_name = gguf_name_map.get(tensor.name)
            if mapped_name is None:
                mapped_name = self._normalize_name(tensor.name)
            if (
                mapped_name not in allowed_names
                and self._resolve_linear_qweight(mapped_name, param_names) is None
            ):
                continue
            mapped.append(
                _MappedTensor(
                    name=mapped_name,
                    tensor=tensor,
                    tensor_type=tensor.tensor_type,
                )
            )

        if not mapped:
            raise RuntimeError(
                "No GGUF tensors were mapped for Qwen-Image GGUF loader. "
                "Please verify the GGUF file and model structure."
            )

        for item in mapped:
            linear_qweight = self._resolve_linear_qweight(item.name, param_names)
            if linear_qweight is None:
                continue
            weight_type_name = linear_qweight.replace("qweight", "qweight_type")
            yield weight_type_name, torch.tensor(item.tensor_type)

        for item in mapped:
            weight = item.tensor.data
            weight_type = item.tensor_type
            linear_qweight = self._resolve_linear_qweight(item.name, param_names)
            if linear_qweight is not None:
                name = linear_qweight
            else:
                name = item.name

            if weight_type.name == "BF16" and weight.dtype == np.uint8:
                weight = weight.view(np.uint16)
                if reader.byte_order == "S":
                    weight = weight.byteswap()
                param = torch.tensor(weight).view(torch.bfloat16)
            else:
                param = torch.tensor(weight)

            yield name, param

    def _normalize_name(self, name: str) -> str:
        if name.endswith(".scale"):
            name = name[:-6] + ".weight"
        if name.endswith("_weight"):
            name = name[:-7] + ".weight"
        if ".to_out.0." in name:
            name = name.replace(".to_out.0.", ".to_out.")
        return name

    def _build_allowed_names(self) -> set[str]:
        prefix = getattr(self.source, "prefix", "")
        target = self.model.get_submodule(prefix.rstrip(".")) if prefix else self.model
        allowed = {name for name, _ in target.named_parameters()}
        allowed.update(name for name, _ in target.named_buffers())
        for name in list(allowed):
            if name.endswith(".qweight"):
                allowed.add(name.replace(".qweight", ".weight"))
            elif name.endswith(".qweight_type"):
                allowed.add(name.replace(".qweight_type", ".weight"))

        virtual_names = set()
        for name in allowed:
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
        allowed.update(virtual_names)
        return allowed

    def _build_param_names(self) -> set[str]:
        prefix = getattr(self.source, "prefix", "")
        target = self.model.get_submodule(prefix.rstrip(".")) if prefix else self.model
        return {name for name, _ in target.named_parameters()}

    def _resolve_linear_qweight(self, name: str, param_names: set[str]) -> str | None:
        if not name.endswith(".weight"):
            return None
        if ".to_out.0." in name:
            name = name.replace(".to_out.0.", ".to_out.")
        for shard_token in (
            ".to_q.",
            ".to_k.",
            ".to_v.",
            ".add_q_proj.",
            ".add_k_proj.",
            ".add_v_proj.",
        ):
            if shard_token in name:
                return name.replace(".weight", ".qweight")
        candidate = name.replace(".weight", ".qweight")
        if candidate in param_names:
            return candidate
        return None

    def _build_gguf_name_map(self, reader) -> dict[str, str]:
        try:
            import gguf  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency error
            raise RuntimeError(
                "GGUF support requires the 'gguf' package to be installed."
            ) from exc

        gguf_tensor_names = {tensor.name for tensor in reader.tensors}

        def resolve_model_type() -> str:
            cfg = self.od_config.tf_model_config
            model_type = None
            if cfg is not None:
                model_type = cfg.get("model_type")
            if model_type:
                return model_type
            model_class = self.od_config.model_class_name or ""
            if model_class.startswith("QwenImage"):
                return "qwen_image"
            raise ValueError("Cannot infer gguf model_type for Qwen-Image.")

        def resolve_arch(model_type: str):
            for key, value in gguf.MODEL_ARCH_NAMES.items():
                if value == model_type:
                    return key
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")

        def resolve_num_layers(target_module: torch.nn.Module) -> int:
            if hasattr(target_module, "transformer_blocks"):
                return len(getattr(target_module, "transformer_blocks"))
            cfg = self.od_config.tf_model_config
            if cfg is not None:
                for key in ("num_hidden_layers", "num_layers", "n_layers"):
                    value = cfg.get(key)
                    if isinstance(value, int) and value > 0:
                        return value
            raise ValueError("Cannot infer gguf num_layers for Qwen-Image.")

        def get_target_module(root: torch.nn.Module, prefix: str) -> torch.nn.Module:
            if not prefix:
                return root
            prefix = prefix.rstrip(".")
            if hasattr(root, "get_submodule"):
                return root.get_submodule(prefix)
            current = root
            for part in prefix.split("."):
                current = getattr(current, part)
            return current

        def split_name(name: str) -> tuple[str, str]:
            if name.endswith("_weight"):
                return name[:-7], "weight"
            if "." in name:
                base, suffix = name.rsplit(".", 1)
                return base, suffix
            return name, ""

        model_type = resolve_model_type()
        arch = resolve_arch(model_type)
        target_module = get_target_module(self.model, self.source.prefix)
        num_layers = resolve_num_layers(target_module)
        name_map = gguf.get_tensor_name_map(arch, num_layers)

        candidate_names = {name for name, _ in target_module.named_parameters()}
        candidate_names.update(name for name, _ in target_module.named_buffers())
        for name in list(candidate_names):
            if ".to_qkv." in name:
                candidate_names.add(name.replace(".to_qkv.", ".to_q."))
                candidate_names.add(name.replace(".to_qkv.", ".to_k."))
                candidate_names.add(name.replace(".to_qkv.", ".to_v."))
            if ".add_kv_proj." in name:
                candidate_names.add(name.replace(".add_kv_proj.", ".add_q_proj."))
                candidate_names.add(name.replace(".add_kv_proj.", ".add_k_proj."))
                candidate_names.add(name.replace(".add_kv_proj.", ".add_v_proj."))
            if ".to_out." in name:
                candidate_names.add(name.replace(".to_out.", ".to_out.0."))

        gguf_to_model_map: dict[str, str] = {}
        for name in candidate_names:
            base_name, suffix = split_name(name)
            gguf_base = name_map.get_name(base_name)
            if gguf_base is None:
                continue
            candidates = []
            if suffix:
                candidates.append(f"{gguf_base}.{suffix}")
                if suffix == "weight":
                    candidates.append(f"{gguf_base}.scale")
            else:
                candidates.append(gguf_base)
            gguf_name = next((c for c in candidates if c in gguf_tensor_names), None)
            if gguf_name is None:
                continue
            gguf_to_model_map[gguf_name] = name

        return gguf_to_model_map
