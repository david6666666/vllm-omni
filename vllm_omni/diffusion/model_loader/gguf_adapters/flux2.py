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
    row_slice: slice | None = None
    swap_scale_shift: bool = False


class Flux2GGUFAdapter(GGUFAdapter):
    """GGUF adapter for Flux2 models with qkv splitting and adaLN swap."""

    @staticmethod
    def is_compatible(od_config, model: torch.nn.Module, source) -> bool:
        model_class = od_config.model_class_name or ""
        if model_class.startswith("Flux2"):
            return True
        cfg = od_config.tf_model_config
        if cfg is not None:
            model_type = str(cfg.get("model_type", "")).lower()
            if model_type.startswith("flux2"):
                return True
        # Fallback: Flux2 transformer has single_transformer_blocks
        prefix = getattr(source, "prefix", "")
        target = model.get_submodule(prefix.rstrip(".")) if prefix else model
        return hasattr(target, "single_transformer_blocks")

    def weights_iterator(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        try:
            import gguf  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency error
            raise RuntimeError(
                "GGUF support requires the 'gguf' package to be installed."
            ) from exc

        reader = gguf.GGUFReader(self.gguf_file)
        allowed_names = self._build_allowed_names()
        mapped: list[_MappedTensor] = []

        for tensor in reader.tensors:
            for mapped_tensor in self._map_tensor_name(tensor):
                if mapped_tensor.name not in allowed_names:
                    continue
                mapped.append(mapped_tensor)

        if not mapped:
            raise RuntimeError(
                "No GGUF tensors were mapped for Flux2 GGUF loader. "
                "Please verify the GGUF file and model structure."
            )

        for item in mapped:
            weight_type = item.tensor_type
            if weight_type.name not in ("F32", "BF16", "F16"):
                weight_type_name = item.name.replace("weight", "qweight_type")
                yield weight_type_name, torch.tensor(weight_type)

        for item in mapped:
            weight = item.tensor.data
            if item.row_slice is not None:
                weight = weight[item.row_slice]
            weight_type = item.tensor_type
            if weight_type.name not in ("F32", "BF16", "F16"):
                name = item.name.replace("weight", "qweight")
            else:
                name = item.name

            if weight_type.name == "BF16" and weight.dtype == np.uint8:
                weight = weight.view(np.uint16)
                if reader.byte_order == "S":
                    weight = weight.byteswap()
                param = torch.tensor(weight).view(torch.bfloat16)
            else:
                param = torch.tensor(weight)

            if item.swap_scale_shift:
                shift, scale = param.chunk(2, dim=0)
                param = torch.cat([scale, shift], dim=0)

            yield name, param

    def _build_allowed_names(self) -> set[str]:
        prefix = getattr(self.source, "prefix", "")
        target = self.model.get_submodule(prefix.rstrip(".")) if prefix else self.model
        allowed = {name for name, _ in target.named_parameters()}
        allowed.update(name for name, _ in target.named_buffers())

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
        allowed.update(virtual_names)
        return allowed

    def _map_tensor_name(self, tensor) -> list[_MappedTensor]:
        name = tensor.name

        if name.startswith("double_blocks."):
            return self._map_double_blocks(tensor)
        if name.startswith("single_blocks."):
            return self._map_single_blocks(tensor)
        if name.startswith("final_layer.adaLN_modulation.1") and name.endswith(".weight"):
            return [
                _MappedTensor(
                    name="norm_out.linear.weight",
                    tensor=tensor,
                    tensor_type=tensor.tensor_type,
                    swap_scale_shift=True,
                )
            ]

        for src, dst in _FLUX2_TRANSFORMER_KEYS_RENAME_DICT.items():
            name = name.replace(src, dst)

        return [
            _MappedTensor(
                name=name,
                tensor=tensor,
                tensor_type=tensor.tensor_type,
            )
        ]

    def _map_double_blocks(self, tensor) -> list[_MappedTensor]:
        name = tensor.name
        parts = name.split(".")
        block_idx = parts[1]
        within_block_name = ".".join(parts[2:-1])
        param_type = parts[-1]
        if param_type == "scale":
            param_type = "weight"

        if "qkv" in within_block_name:
            if "img_attn" in within_block_name:
                q_name = f"transformer_blocks.{block_idx}.attn.to_q.{param_type}"
                k_name = f"transformer_blocks.{block_idx}.attn.to_k.{param_type}"
                v_name = f"transformer_blocks.{block_idx}.attn.to_v.{param_type}"
            elif "txt_attn" in within_block_name:
                q_name = f"transformer_blocks.{block_idx}.attn.add_q_proj.{param_type}"
                k_name = f"transformer_blocks.{block_idx}.attn.add_k_proj.{param_type}"
                v_name = f"transformer_blocks.{block_idx}.attn.add_v_proj.{param_type}"
            else:
                return []

            weight = tensor.data
            dim0 = weight.shape[0]
            split = dim0 // 3
            return [
                _MappedTensor(q_name, tensor, tensor.tensor_type, slice(0, split)),
                _MappedTensor(k_name, tensor, tensor.tensor_type, slice(split, 2 * split)),
                _MappedTensor(v_name, tensor, tensor.tensor_type, slice(2 * split, 3 * split)),
            ]

        mapped_name = _FLUX2_TRANSFORMER_DOUBLE_BLOCK_KEY_MAP.get(within_block_name)
        if mapped_name is None:
            return []
        target = f"transformer_blocks.{block_idx}.{mapped_name}.{param_type}"
        return [_MappedTensor(target, tensor, tensor.tensor_type)]

    def _map_single_blocks(self, tensor) -> list[_MappedTensor]:
        name = tensor.name
        parts = name.split(".")
        block_idx = parts[1]
        within_block_name = ".".join(parts[2:-1])
        param_type = parts[-1]
        if param_type == "scale":
            param_type = "weight"

        mapped_name = _FLUX2_TRANSFORMER_SINGLE_BLOCK_KEY_MAP.get(within_block_name)
        if mapped_name is None:
            return []
        target = f"single_transformer_blocks.{block_idx}.{mapped_name}.{param_type}"
        return [_MappedTensor(target, tensor, tensor.tensor_type)]


_FLUX2_TRANSFORMER_KEYS_RENAME_DICT = {
    "img_in": "x_embedder",
    "txt_in": "context_embedder",
    "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
    "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
    "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
    "single_stream_modulation.lin": "single_stream_modulation.linear",
    "final_layer.linear": "proj_out",
}

_FLUX2_TRANSFORMER_DOUBLE_BLOCK_KEY_MAP = {
    "img_attn.norm.query_norm": "attn.norm_q",
    "img_attn.norm.key_norm": "attn.norm_k",
    "img_attn.proj": "attn.to_out.0",
    "img_mlp.0": "ff.linear_in",
    "img_mlp.2": "ff.linear_out",
    "txt_attn.norm.query_norm": "attn.norm_added_q",
    "txt_attn.norm.key_norm": "attn.norm_added_k",
    "txt_attn.proj": "attn.to_add_out",
    "txt_mlp.0": "ff_context.linear_in",
    "txt_mlp.2": "ff_context.linear_out",
}

_FLUX2_TRANSFORMER_SINGLE_BLOCK_KEY_MAP = {
    "linear1": "attn.to_qkv_mlp_proj",
    "linear2": "attn.to_out",
    "norm.query_norm": "attn.norm_q",
    "norm.key_norm": "attn.norm_k",
}
