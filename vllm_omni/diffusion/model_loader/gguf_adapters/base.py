# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Generator

import torch
from vllm.model_executor.model_loader.weight_utils import gguf_quant_weights_iterator


class GGUFAdapter:
    """Default GGUF adapter using gguf-py's tensor name mapping."""

    def __init__(self, gguf_file: str, model: torch.nn.Module, source, od_config) -> None:
        self.gguf_file = gguf_file
        self.model = model
        self.source = source
        self.od_config = od_config

    @staticmethod
    def is_compatible(od_config, model: torch.nn.Module, source) -> bool:
        # Default adapter matches any model.
        return True

    def weights_iterator(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        name_map = self._build_gguf_name_map()
        return gguf_quant_weights_iterator(self.gguf_file, name_map)

    def _build_gguf_name_map(self) -> dict[str, str]:
        try:
            import gguf  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency error
            raise RuntimeError("GGUF support requires the 'gguf' package to be installed.") from exc

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
            if model_class.startswith("Flux2"):
                return "flux"
            raise ValueError("Cannot infer gguf model_type for diffusion model.")

        def resolve_arch(model_type: str):
            for key, value in gguf.MODEL_ARCH_NAMES.items():
                if value == model_type:
                    return key
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")

        def resolve_num_layers(target_module: torch.nn.Module) -> int:
            if hasattr(target_module, "transformer_blocks"):
                return len(getattr(target_module, "transformer_blocks"))
            if hasattr(target_module, "double_blocks"):
                return len(getattr(target_module, "double_blocks"))
            cfg = self.od_config.tf_model_config
            if cfg is not None:
                for key in ("num_hidden_layers", "num_layers", "n_layers"):
                    value = cfg.get(key)
                    if isinstance(value, int) and value > 0:
                        return value
            raise ValueError("Cannot infer gguf num_layers for diffusion model.")

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

        reader = gguf.GGUFReader(self.gguf_file)
        gguf_tensor_names = {tensor.name for tensor in reader.tensors}

        model_type = resolve_model_type()
        arch = resolve_arch(model_type)
        target_module = get_target_module(self.model, self.source.prefix)
        num_layers = resolve_num_layers(target_module)
        name_map = gguf.get_tensor_name_map(arch, num_layers)

        gguf_to_model_map: dict[str, str] = {}
        for name, _ in target_module.named_parameters():
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

        for name, _ in target_module.named_buffers():
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

        if not gguf_to_model_map:
            raise RuntimeError(f"No GGUF tensors were mapped for model_class_name={self.od_config.model_class_name!r}.")
        return gguf_to_model_map
