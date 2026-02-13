# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Generator

import numpy as np
import torch

from .base import GGUFAdapter, MappedTensor


class ZImageGGUFAdapter(GGUFAdapter):
    """GGUF adapter for Z-Image models with QKV/FFN shard support."""

    _include_qkv_virtuals = True
    _include_to_out_virtuals = True
    _include_w13_virtuals = True
    _shard_tokens = (
        ".to_q.",
        ".to_k.",
        ".to_v.",
        ".w1.",
        ".w3.",
    )

    @staticmethod
    def is_compatible(od_config, model: torch.nn.Module, source) -> bool:
        model_class = od_config.model_class_name or ""
        if model_class.startswith("ZImage"):
            return True
        cfg = od_config.tf_model_config
        if cfg is not None:
            model_type = str(cfg.get("model_type", "")).lower()
            if model_type in {"z_image", "zimage", "z-image"}:
                return True
        return False

    def weights_iterator(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        try:
            import gguf  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency error
            raise RuntimeError("GGUF support requires the 'gguf' package to be installed.") from exc

        reader = gguf.GGUFReader(self.gguf_file)
        gguf_name_map = self._build_gguf_name_map(reader)
        allowed_names = self._build_allowed_names()
        param_names = self._build_param_names()
        mapped: list[MappedTensor] = []

        for tensor in reader.tensors:
            for mapped_tensor in self._map_tensor_name(tensor, gguf_name_map):
                linear_qweight = self._resolve_linear_qweight(mapped_tensor.name, param_names)
                if mapped_tensor.name not in allowed_names and linear_qweight is None:
                    continue
                if linear_qweight is None and tensor.tensor_type.name not in ("F32", "BF16", "F16"):
                    # Skip quantized tensors that map to non-quantized parameters.
                    continue
                mapped.append(mapped_tensor)

        if not mapped:
            raise RuntimeError(
                "No GGUF tensors were mapped for Z-Image GGUF loader. Please verify the GGUF file and model structure."
            )

        for item in mapped:
            linear_qweight = self._resolve_linear_qweight(item.name, param_names)
            if linear_qweight is None:
                continue
            weight_type_name = linear_qweight.replace("qweight", "qweight_type")
            yield weight_type_name, torch.tensor(item.tensor_type)

        for item in mapped:
            weight = item.tensor.data
            if item.row_slice is not None:
                weight = weight[item.row_slice]
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

    def _get_patch_key(self) -> str:
        prefix = getattr(self.source, "prefix", "")
        target = self.model.get_submodule(prefix.rstrip(".")) if prefix else self.model
        if hasattr(target, "all_x_embedder"):
            keys = list(getattr(target, "all_x_embedder").keys())
            if "2-1" in keys:
                # Default to the standard Z-Image Turbo patch/frequency config
                # (patch_size=2, f_patch_size=1) when available.
                return "2-1"
            if keys:
                return sorted(keys)[0]
        return "2-1"

    def _apply_zimage_renames(self, name: str) -> str:
        if name.startswith("model.diffusion_model."):
            name = name.replace("model.diffusion_model.", "", 1)

        patch_key = self._get_patch_key()
        if name.startswith("x_embedder.") and not name.startswith("all_x_embedder."):
            name = name.replace("x_embedder.", f"all_x_embedder.{patch_key}.", 1)
        if name.startswith("final_layer.") and not name.startswith("all_final_layer."):
            name = name.replace("final_layer.", f"all_final_layer.{patch_key}.", 1)

        name = name.replace(".attention.out.bias", ".attention.to_out.0.bias")
        name = name.replace(".attention.out.weight", ".attention.to_out.0.weight")
        name = name.replace(".attention.k_norm.weight", ".attention.norm_k.weight")
        name = name.replace(".attention.q_norm.weight", ".attention.norm_q.weight")
        return name

    def _map_tensor_name(self, tensor, gguf_name_map: dict[str, str]) -> list[MappedTensor]:
        name = gguf_name_map.get(tensor.name)
        if name is None:
            name = self._normalize_name(tensor.name)
        name = self._apply_zimage_renames(name)

        if ".attention.qkv.weight" in name:
            weight = tensor.data
            dim0 = weight.shape[0]
            split = dim0 // 3
            return [
                MappedTensor(
                    name=name.replace(".attention.qkv.weight", ".attention.to_q.weight"),
                    tensor=tensor,
                    tensor_type=tensor.tensor_type,
                    row_slice=slice(0, split),
                ),
                MappedTensor(
                    name=name.replace(".attention.qkv.weight", ".attention.to_k.weight"),
                    tensor=tensor,
                    tensor_type=tensor.tensor_type,
                    row_slice=slice(split, 2 * split),
                ),
                MappedTensor(
                    name=name.replace(".attention.qkv.weight", ".attention.to_v.weight"),
                    tensor=tensor,
                    tensor_type=tensor.tensor_type,
                    row_slice=slice(2 * split, 3 * split),
                ),
            ]

        return [
            MappedTensor(
                name=name,
                tensor=tensor,
                tensor_type=tensor.tensor_type,
            )
        ]

    def _build_gguf_name_map(self, reader) -> dict[str, str]:
        try:
            import gguf  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency error
            raise RuntimeError("GGUF support requires the 'gguf' package to be installed.") from exc

        gguf_tensor_names = {tensor.name for tensor in reader.tensors}

        def resolve_model_type() -> str:
            cfg = self.od_config.tf_model_config
            model_type = None
            if cfg is not None:
                model_type = cfg.get("model_type")
            if model_type:
                return model_type
            model_class = self.od_config.model_class_name or ""
            if model_class.startswith("ZImage"):
                return "z_image"
            raise ValueError("Cannot infer gguf model_type for Z-Image.")

        def resolve_arch(model_type: str):
            for key, value in gguf.MODEL_ARCH_NAMES.items():
                if value == model_type:
                    return key
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")

        def resolve_num_layers(target_module: torch.nn.Module) -> int:
            if hasattr(target_module, "layers"):
                return len(getattr(target_module, "layers"))
            cfg = self.od_config.tf_model_config
            if cfg is not None:
                for key in ("num_hidden_layers", "num_layers", "n_layers"):
                    value = cfg.get(key)
                    if isinstance(value, int) and value > 0:
                        return value
            raise ValueError("Cannot infer gguf num_layers for Z-Image.")

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
        try:
            arch = resolve_arch(model_type)
        except RuntimeError:
            # Fallback: some gguf versions may not register z_image arch.
            # In that case, rely on direct tensor names from the GGUF file.
            return {}
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
            if ".w13." in name:
                candidate_names.add(name.replace(".w13.", ".w1."))
                candidate_names.add(name.replace(".w13.", ".w3."))
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
