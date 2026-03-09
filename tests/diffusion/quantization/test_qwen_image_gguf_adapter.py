# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from vllm_omni.diffusion.model_loader.gguf_adapters import get_gguf_adapter
from vllm_omni.diffusion.model_loader.gguf_adapters.qwen_image import (
    QwenImageGGUFAdapter,
)


def _make_od_config(
    model_class_name: str = "QwenImagePipeline",
    model_type: str = "qwen_image",
):
    return SimpleNamespace(
        model_class_name=model_class_name,
        tf_model_config={"model_type": model_type},
    )


def _make_source(prefix: str = "", subfolder: str = "transformer"):
    return SimpleNamespace(prefix=prefix, subfolder=subfolder)


class _FakeTransformer:
    pass


def test_qwen_adapter_selected_for_qwen_image_family():
    adapter = get_gguf_adapter(
        "dummy.gguf",
        _FakeTransformer(),
        _make_source(),
        _make_od_config(),
    )
    assert isinstance(adapter, QwenImageGGUFAdapter)


def test_qwen_adapter_matches_multiple_pipeline_variants():
    for model_class_name in (
        "QwenImagePipeline",
        "QwenImageEditPipeline",
        "QwenImageEditPlusPipeline",
        "QwenImageLayeredPipeline",
    ):
        assert QwenImageGGUFAdapter.is_compatible(
            _make_od_config(model_class_name=model_class_name),
            _FakeTransformer(),
            _make_source(),
        )


def test_qwen_adapter_maps_fused_projection_names():
    mapper = QwenImageGGUFAdapter.gguf_to_hf_mapper

    assert mapper.apply_list(["transformer_blocks.0.attn.to_q.weight"]) == [
        "transformer_blocks.0.attn.to_qkv.weight"
    ]
    assert mapper.apply_list(["transformer_blocks.0.attn.add_q_proj.weight"]) == [
        "transformer_blocks.0.attn.add_kv_proj.weight"
    ]
    assert mapper.apply_list(["transformer_blocks.0.attn.to_out.0.weight"]) == [
        "transformer_blocks.0.attn.to_out.weight"
    ]


def test_qwen_adapter_keeps_already_fused_names_stable():
    mapper = QwenImageGGUFAdapter.gguf_to_hf_mapper

    assert mapper.apply_list(["transformer_blocks.0.attn.to_qkv.weight"]) == [
        "transformer_blocks.0.attn.to_qkv.weight"
    ]
    assert mapper.apply_list(["transformer_blocks.0.attn.add_kv_proj.weight"]) == [
        "transformer_blocks.0.attn.add_kv_proj.weight"
    ]
