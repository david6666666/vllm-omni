# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Generator

import numpy as np
import torch

from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.model_loader.weight_utils import gguf_quant_weights_iterator

from .base import GGUFAdapter, MappedTensor


FLUX2_TRANSFORMER_KEYS_RENAME_DICT = {
    # Image and text input projections
    "img_in": "x_embedder",
    "txt_in": "context_embedder",
    # Timestep and guidance embeddings
    "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
    "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
    # Modulation parameters
    "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
    "single_stream_modulation.lin": "single_stream_modulation.linear",
    # Final output layer
    # "final_layer.adaLN_modulation.1": "norm_out.linear",  # Handle separately since we need to swap mod params
    "final_layer.linear": "proj_out",
}

FLUX2_TRANSFORMER_ADA_LAYER_NORM_KEY_MAP = {
    "final_layer.adaLN_modulation.1": "norm_out.linear",
}

FLUX2_TRANSFORMER_DOUBLE_BLOCK_KEY_MAP = {
    # Handle fused QKV projections separately as we need to break into Q, K, V projections
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
    # Additional for fuse qkv
    "img_attn.qkv": "attn.to_qkv_mlp_proj",
    "txt_attn.qkv": "attn.add_kv_proj",
}

FLUX2_TRANSFORMER_SINGLE_BLOCK_KEY_MAP = {
    "linear1": "attn.to_qkv_mlp_proj",
    "linear2": "attn.to_out",
    "norm.query_norm": "attn.norm_q",
    "norm.key_norm": "attn.norm_k",
}


class Flux2KleinGGUFAdapter(GGUFAdapter):
    """GGUF adapter for Flux2-Klein models with qkv splitting and adaLN swap."""

    gguf_to_hf_mapper = WeightsMapper(
        # double_stream_modulation
        orig_to_new_prefix = FLUX2_TRANSFORMER_KEYS_RENAME_DICT | FLUX2_TRANSFORMER_ADA_LAYER_NORM_KEY_MAP,
        orig_to_new_substr = FLUX2_TRANSFORMER_DOUBLE_BLOCK_KEY_MAP | FLUX2_TRANSFORMER_SINGLE_BLOCK_KEY_MAP,
    )

    def weights_iterator(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        weights = gguf_quant_weights_iterator(self.gguf_file, {})
        yield from self.gguf_to_hf_mapper.apply(weights)
        # try:
        #     import gguf  # type: ignore
        # except Exception as exc:  # pragma: no cover - dependency error
        #     raise RuntimeError("GGUF support requires the 'gguf' package to be installed.") from exc

        # reader = gguf.GGUFReader(self.gguf_file)
        # allowed_names = self._build_allowed_names()
        # param_names = self._build_param_names()
        # mapped: list[MappedTensor] = []

        # for tensor in reader.tensors:
        #     for mapped_tensor in self._map_tensor_name(tensor):
        #         if (
        #             mapped_tensor.name not in allowed_names
        #             and self._resolve_linear_qweight(mapped_tensor.name, param_names) is None
        #         ):
        #             continue
        #         mapped.append(mapped_tensor)

        # if not mapped:
        #     raise RuntimeError(
        #         "No GGUF tensors were mapped for Flux2 GGUF loader. Please verify the GGUF file and model structure."
        #     )

        # for item in mapped:
        #     linear_qweight = self._resolve_linear_qweight(item.name, param_names)
        #     is_linear_weight = linear_qweight is not None
        #     if not is_linear_weight:
        #         continue
        #     weight_type_name = linear_qweight.replace("qweight", "qweight_type")
        #     yield weight_type_name, torch.tensor(item.tensor_type)

        # for item in mapped:
        #     weight = item.tensor.data
        #     if item.row_slice is not None:
        #         weight = weight[item.row_slice]
        #     weight_type = item.tensor_type
        #     linear_qweight = self._resolve_linear_qweight(item.name, param_names)
        #     is_linear_weight = linear_qweight is not None
        #     if is_linear_weight:
        #         name = linear_qweight
        #     else:
        #         name = item.name

        #     if weight_type.name == "BF16" and weight.dtype == np.uint8:
        #         weight = weight.view(np.uint16)
        #         if reader.byte_order == "S":
        #             weight = weight.byteswap()
        #         param = torch.tensor(weight).view(torch.bfloat16)
        #     else:
        #         param = torch.tensor(weight)

        #     if item.swap_scale_shift:
        #         shift, scale = param.chunk(2, dim=0)
        #         param = torch.cat([scale, shift], dim=0)

        #     yield name, param

    def _map_tensor_name(self, tensor) -> list[MappedTensor]:
        name = tensor.name

        if name.startswith("double_blocks."):
            return self._map_double_blocks(tensor)
        if name.startswith("single_blocks."):
            return self._map_single_blocks(tensor)
        if name.startswith("final_layer.adaLN_modulation.1") and name.endswith(".weight"):
            return [
                MappedTensor(
                    name="norm_out.linear.weight",
                    tensor=tensor,
                    tensor_type=tensor.tensor_type,
                    swap_scale_shift=True,
                )
            ]

        for src, dst in _FLUX2_TRANSFORMER_KEYS_RENAME_DICT.items():
            name = name.replace(src, dst)

        return [
            MappedTensor(
                name=name,
                tensor=tensor,
                tensor_type=tensor.tensor_type,
            )
        ]

    def _map_double_blocks(self, tensor) -> list[MappedTensor]:
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
                MappedTensor(q_name, tensor, tensor.tensor_type, slice(0, split)),
                MappedTensor(k_name, tensor, tensor.tensor_type, slice(split, 2 * split)),
                MappedTensor(v_name, tensor, tensor.tensor_type, slice(2 * split, 3 * split)),
            ]

        mapped_name = _FLUX2_TRANSFORMER_DOUBLE_BLOCK_KEY_MAP.get(within_block_name)
        if mapped_name is None:
            return []
        target = f"transformer_blocks.{block_idx}.{mapped_name}.{param_type}"
        return [MappedTensor(target, tensor, tensor.tensor_type)]

    def _map_single_blocks(self, tensor) -> list[MappedTensor]:
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
        return [MappedTensor(target, tensor, tensor.tensor_type)]


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
