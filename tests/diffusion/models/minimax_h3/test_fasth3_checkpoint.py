# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.prepare_fasth3_checkpoint import (
    FASTH3_V2_MODEL_ID,
    iter_native_weights,
    native_weight_target,
    prepare_checkpoint,
)
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3.fasth3_checkpoint import (
    FASTH3_V2_BASE_SCHEDULE,
    FastH3CheckpointSpec,
)
from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _v2_release() -> dict[str, object]:
    return {
        "schema_version": 1,
        "partition": "fl2va",
        "tasks": ["t2va"],
        "sigma_shift_scales": {"video": 10.0, "audio": 3.0},
        "base_schedule": list(FASTH3_V2_BASE_SCHEDULE.base_schedule),
        "fasth3": {
            "model_id": FASTH3_V2_MODEL_ID,
            "vsa_sparsity": 0.8,
            "vsa_tile_size": 64,
        },
    }


def _sampling(
    *,
    num_inference_steps: int | str | None = 8,
    guidance_scale: float | None = 1.0,
    extra_args: dict[str, object] | None = None,
    lora_request: LoRARequest | None = None,
) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        # The string form is intentionally accepted here to exercise request
        # validation of malformed client input.
        num_inference_steps=num_inference_steps,  # type: ignore[arg-type]
        guidance_scale=guidance_scale,
        extra_args=extra_args or {},
        lora_request=lora_request,
    )


def _od_config(
    *,
    backend: str = "FASTVIDEO_VSA",
    topk: int | None = None,
    per_role: dict[str, AttentionSpec] | None = None,
    lora_path: str | None = None,
    ring_degree: int = 1,
    allgather_degree: int = 1,
) -> OmniDiffusionConfig:
    attention_config = AttentionConfig(
        default=AttentionSpec(backend=backend, fastvideo_vsa_topk=topk),
        per_role=per_role or {},
    )
    return OmniDiffusionConfig(
        diffusion_attention_config=attention_config,
        lora_path=lora_path,
        parallel_config=DiffusionParallelConfig(ring_degree=ring_degree, allgather_degree=allgather_degree),
    )


def test_v2_release_has_nine_sigma_nodes_and_eight_intervals():
    assert FASTH3_V2_BASE_SCHEDULE.base_schedule == (
        0.999,
        0.874,
        0.749,
        0.624,
        0.5,
        0.375,
        0.25,
        0.125,
        0.0,
    )
    assert len(FASTH3_V2_BASE_SCHEDULE.base_schedule) == 9
    assert FASTH3_V2_BASE_SCHEDULE.num_inference_steps == 8

    # These are the release positions after each modality's rectified-flow
    # shift, rather than a schedule derived from a requested step count.
    assert FASTH3_V2_BASE_SCHEDULE.shifted_sigmas(10.0) == pytest.approx(
        [
            0.9998999099189271,
            0.985788405143244,
            0.9675752486758817,
            0.94316807738815,
            0.9090909090909091,
            0.8571428571428571,
            0.7692307692307693,
            0.5882352941176471,
            0.0,
        ],
        abs=1e-12,
    )
    assert FASTH3_V2_BASE_SCHEDULE.shifted_sigmas(3.0) == pytest.approx(
        [
            0.9996664442961973,
            0.9541484716157204,
            0.8995196156925539,
            0.8327402135231315,
            0.75,
            0.6428571428571429,
            0.5,
            0.3,
            0.0,
        ],
        abs=1e-12,
    )


def test_v2_metadata_is_self_consistent():
    spec = FastH3CheckpointSpec.from_metadata(_v2_release())
    assert spec is not None
    assert spec.vsa_sparsity == 0.8


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "base_schedule",
            [0.998, 0.874, 0.749, 0.624, 0.5, 0.375, 0.25, 0.125, 0.0],
            "exact eight-step",
        ),
        ("sigma_shift_scales", {"video": 12.0, "audio": 3.0}, "shifts 10/3"),
        ("sigma_shift_scales", {"video": 10.0, "audio": 4.0}, "shifts 10/3"),
        ("partition", "ref2va", "FL2VA partition"),
        ("tasks", ["fl2va"], "T2VA"),
    ],
)
def test_v2_metadata_rejects_schedule_shift_and_task_drift(field, value, match):
    release = _v2_release()
    release[field] = value
    with pytest.raises(ValueError, match=match):
        FastH3CheckpointSpec.from_metadata(release)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("model_id", "FastVideo/FastH3-other", "identity"),
        ("vsa_sparsity", 0.9, "sparsity=0.8"),
        ("vsa_tile_size", 32, "tile_size=64"),
    ],
)
def test_v2_metadata_rejects_identity_and_vsa_policy_drift(field, value, match):
    release = _v2_release()
    cast(dict[str, object], release["fasth3"])[field] = value
    with pytest.raises(ValueError, match=match):
        FastH3CheckpointSpec.from_metadata(release)


def test_legacy_release_without_a_fasth3_marker_is_not_claimed():
    old_release = {
        "schema_version": 1,
        "partition": "fl2va",
        "tasks": ["t2va", "fl2va"],
        "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
        "base_schedule": None,
    }
    assert FastH3CheckpointSpec.from_metadata(old_release) is None


def test_v2_serving_contract_accepts_backend_selected_for_h3_self_role():
    config = _od_config(
        backend="TORCH_SDPA",
        per_role={"self": AttentionSpec(backend="FASTVIDEO_VSA")},
    )
    FastH3CheckpointSpec.from_metadata(_v2_release()).check_serving_contract(partition="fl2va", od_config=config)


@pytest.mark.parametrize(
    ("kwargs", "partition", "match"),
    [
        ({}, "ref2va", "task-type fl2va"),
        ({"lora_path": "/tmp/adapter"}, "fl2va", "LoRA"),
        ({"backend": "TORCH_SDPA"}, "fl2va", "FASTVIDEO_VSA"),
        ({"topk": 2}, "fl2va", "fixed fastvideo_vsa_topk"),
        ({"ring_degree": 2}, "fl2va", "local attention"),
        ({"allgather_degree": 2}, "fl2va", "local attention"),
    ],
)
def test_v2_serving_contract_rejects_incompatible_runtime(kwargs, partition, match):
    spec = FastH3CheckpointSpec.from_metadata(_v2_release())
    with pytest.raises(ValueError, match=match):
        spec.check_serving_contract(partition=partition, od_config=_od_config(**kwargs))


def test_v2_serving_contract_reads_a_per_role_fixed_topk_override():
    spec = FastH3CheckpointSpec.from_metadata(_v2_release())
    config = _od_config(
        per_role={"self": AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_topk=1)},
    )
    with pytest.raises(ValueError, match="fixed fastvideo_vsa_topk"):
        spec.check_serving_contract(partition="fl2va", od_config=config)


def test_v2_request_contract_accepts_omitted_steps_and_default_shifts():
    spec = FastH3CheckpointSpec.from_metadata(_v2_release())
    spec.check_request(_sampling(num_inference_steps=None))
    spec.check_request(_sampling(extra_args={"flow_shift": 10, "audio_flow_shift": 3}, guidance_scale=1))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_inference_steps": 7}, "num_inference_steps=8"),
        ({"num_inference_steps": 9}, "num_inference_steps=8"),
        ({"num_inference_steps": "8"}, "num_inference_steps=8"),
        ({"extra_args": {"flow_shift": 9}}, "flow_shift=10"),
        ({"extra_args": {"audio_flow_shift": 4}}, "audio_flow_shift=3"),
        ({"extra_args": {"flow_shift": "bad"}}, "flow_shift=10"),
        ({"guidance_scale": 2.0}, "guidance_scale=1"),
        (
            {"lora_request": LoRARequest(lora_name="test", lora_int_id=1, lora_path="/tmp/test.safetensors")},
            "per-request LoRA",
        ),
    ],
)
def test_v2_request_contract_rejects_sampling_drift(kwargs, match):
    spec = FastH3CheckpointSpec.from_metadata(_v2_release())
    with pytest.raises(OmniClientError, match=match):
        spec.check_request(_sampling(**kwargs))


def test_native_weight_target_covers_full_norms_and_vsa_gate():
    expected = {
        "token_refiner.final_norm.weight": ("token_refiner.final_norm.weight", "plain"),
        "transformer_blocks.2.attn.norm_q.bias": ("blocks.2.attn.q_norm.bias", "plain"),
        "transformer_blocks.2.attn.norm_k.weight": ("blocks.2.attn.k_norm.weight", "plain"),
        "transformer_blocks.2.norm1.weight": ("blocks.2.norm1.weight", "plain"),
        "transformer_blocks.2.norm2.bias": ("blocks.2.norm2.bias", "plain"),
        "transformer_blocks.2.attn.to_gate_compress.weight": (
            "blocks.2.attn.to_gate_compress.weight",
            "plain",
        ),
        "norm_out.norm.weight": ("final_layer.norm.weight", "plain"),
        "norm_out.linear.bias": ("final_layer.adaln_proj.linear.bias", "plain"),
    }
    for source_name, target in expected.items():
        assert native_weight_target(source_name) == target


def test_iter_native_weights_groups_non_symmetric_qkv_by_head():
    q = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [[101, 102, 103], [104, 105, 106], [107, 108, 109], [110, 111, 112]],
        dtype=torch.float32,
    )
    v = torch.tensor(
        [[201, 202, 203], [204, 205, 206], [207, 208, 209], [210, 211, 212]],
        dtype=torch.float32,
    )
    source = {
        "transformer_blocks.0.attn.to_q.weight": q,
        "transformer_blocks.0.attn.to_k.weight": k,
        "transformer_blocks.0.attn.to_v.weight": v,
    }
    expected_shapes = {"blocks.0.attn.qkv_proj.weight": (12, 3), "rope.inv_freq": (2,)}
    output = dict(
        iter_native_weights(
            {name: tuple(tensor.shape) for name, tensor in source.items()},
            source.__getitem__,
            expected_shapes,
            head_dim=2,
            rope_inv_freq=torch.tensor([1.0, 0.01]),
        )
    )
    expected_qkv = torch.stack(
        [q[0], q[1], k[0], k[1], v[0], v[1], q[2], q[3], k[2], k[3], v[2], v[3]],
    )
    torch.testing.assert_close(output["blocks.0.attn.qkv_proj.weight"], expected_qkv)


def test_iter_native_weights_swaps_non_symmetric_mlp_halves():
    mlp = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    source = {"transformer_blocks.0.ff.net.0.proj.weight": mlp}
    output = dict(
        iter_native_weights(
            {next(iter(source)): tuple(mlp.shape)},
            source.__getitem__,
            {"blocks.0.mlp.fc1.weight": tuple(mlp.shape), "rope.inv_freq": (2,)},
            head_dim=2,
            rope_inv_freq=torch.ones(2),
        )
    )
    torch.testing.assert_close(output["blocks.0.mlp.fc1.weight"], torch.cat((mlp[3:], mlp[:3])))


def test_iter_native_weights_rejects_incomplete_qkv_coverage():
    source = {
        "transformer_blocks.0.attn.to_k.weight": torch.zeros(4, 3),
        "transformer_blocks.0.attn.to_v.weight": torch.zeros(4, 3),
    }
    with pytest.raises(ValueError, match="incomplete QKV"):
        list(
            iter_native_weights(
                {name: tuple(tensor.shape) for name, tensor in source.items()},
                source.__getitem__,
                {"blocks.0.attn.qkv_proj.weight": (12, 3), "rope.inv_freq": (2,)},
                head_dim=2,
                rope_inv_freq=torch.ones(2),
            )
        )


@pytest.mark.parametrize("expected_shape", [(9, 3), (12, 4)])
def test_iter_native_weights_rejects_transformed_shape_mismatch(expected_shape):
    source = {
        "transformer_blocks.0.attn.to_q.weight": torch.zeros(4, 3),
        "transformer_blocks.0.attn.to_k.weight": torch.zeros(4, 3),
        "transformer_blocks.0.attn.to_v.weight": torch.zeros(4, 3),
    }
    expected = {"blocks.0.attn.qkv_proj.weight": expected_shape, "rope.inv_freq": (2,)}
    with pytest.raises(ValueError, match="shape"):
        list(
            iter_native_weights(
                {name: tuple(tensor.shape) for name, tensor in source.items()},
                source.__getitem__,
                expected,
                head_dim=2,
                rope_inv_freq=torch.ones(2),
            )
        )


def test_iter_native_weights_rejects_rope_shape_mismatch():
    source = {
        "transformer_blocks.0.attn.to_q.weight": torch.zeros(4, 3),
        "transformer_blocks.0.attn.to_k.weight": torch.zeros(4, 3),
        "transformer_blocks.0.attn.to_v.weight": torch.zeros(4, 3),
    }
    with pytest.raises(ValueError, match="RoPE shape"):
        list(
            iter_native_weights(
                {name: tuple(tensor.shape) for name, tensor in source.items()},
                source.__getitem__,
                {"blocks.0.attn.qkv_proj.weight": (12, 3), "rope.inv_freq": (2,)},
                head_dim=2,
                rope_inv_freq=torch.ones(3),
            )
        )


def _write_indexed_safetensors(folder: Path, index_name: str, tensors: dict[str, torch.Tensor]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    shard_name = "model-00001.safetensors"
    save_file(tensors, str(folder / shard_name), metadata={"format": "pt"})
    (folder / index_name).write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
                "weight_map": {name: shard_name for name in tensors},
            }
        ),
        encoding="utf-8",
    )


def _make_converter_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    source = tmp_path / "source"
    base = tmp_path / "base"
    base_partition = base / "FL2VA"

    source_config = {
        "num_layers": 1,
        "num_attention_heads": 2,
        "attention_head_dim": 2,
        "hidden_size": 4,
        "ffn_dim": 3,
        "rope_freq_dim": 2,
        "rope_theta": 10000.0,
    }
    base_config = {
        "num_layers": 1,
        "num_attention_heads": 2,
        "attention_head_dim": 2,
        "hidden_size": 4,
        "ffn_hidden_size": 3,
        "rope_inv_freq_len": 2,
        "rope_theta": 10000.0,
    }
    (source / "transformer").mkdir(parents=True)
    (source / "transformer" / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
    (source / "fastvideo_inference.json").write_text(
        json.dumps(
            {
                "model_id": FASTH3_V2_MODEL_ID,
                "schema_version": "fasth3-inference-contract-v1",
                "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
                "transformer_forwards": 8,
                "video_scheduler_shift": 10.0,
                "audio_scheduler_shift": 3.0,
                "guidance_scale": 1.0,
                "attention_backend": "VIDEO_SPARSE_ATTN_H3",
                "vsa_sparsity": 0.8,
                "vsa_tile_size": 64,
                "task": "t2av",
            }
        ),
        encoding="utf-8",
    )

    q = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    k = q + 100
    v = q + 200
    mlp = torch.arange(24, 48, dtype=torch.float32).reshape(6, 4)
    gate = torch.arange(16, dtype=torch.float32).reshape(4, 4) + 300
    rope = 1.0 / (10000.0 ** (torch.arange(0, 4, 2, dtype=torch.float32) / 4))
    source_tensors = {
        "transformer_blocks.0.attn.to_q.weight": q,
        "transformer_blocks.0.attn.to_k.weight": k,
        "transformer_blocks.0.attn.to_v.weight": v,
        "transformer_blocks.0.ff.net.0.proj.weight": mlp,
        "transformer_blocks.0.attn.to_gate_compress.weight": gate,
    }
    _write_indexed_safetensors(
        source / "transformer",
        "diffusion_pytorch_model.safetensors.index.json",
        source_tensors,
    )

    base_tensors = {
        "blocks.0.attn.qkv_proj.weight": torch.zeros(12, 4),
        "blocks.0.mlp.fc1.weight": torch.zeros(6, 4),
        "rope.inv_freq": rope,
    }
    _write_indexed_safetensors(base_partition / "transformer", "model.safetensors.index.json", base_tensors)
    (base_partition / "transformer" / "config.json").write_text(json.dumps(base_config), encoding="utf-8")
    (base_partition / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3Pipeline",
                "_minimax_h3": {
                    "partition": "fl2va",
                    "tasks": ["t2va"],
                    "base_schedule": None,
                },
            }
        ),
        encoding="utf-8",
    )
    for component in ("text_encoder", "tokenizer", "processor", "video_vae", "audio_vae"):
        (base_partition / component).mkdir(parents=True)

    return source, base, source_tensors, base_tensors


def test_converter_publishes_native_weights_release_metadata_and_frozen_links(tmp_path):
    source, base, source_tensors, base_tensors = _make_converter_fixture(tmp_path)
    output = tmp_path / "converted"

    assert prepare_checkpoint(source, base, output) == output
    partition = output / "FL2VA"
    assert output.is_dir()
    assert partition.is_dir()

    for component in ("text_encoder", "tokenizer", "processor", "video_vae", "audio_vae"):
        link = partition / component
        assert link.is_symlink()
        assert link.resolve() == (base / "FL2VA" / component).resolve()

    release = json.loads((partition / "model_index.json").read_text(encoding="utf-8"))["_minimax_h3"]
    assert FastH3CheckpointSpec.from_metadata(release) is not None
    assert json.loads((output / "model_index.json").read_text(encoding="utf-8"))["_minimax_h3"] == release
    conversion = json.loads((output / "conversion.json").read_text(encoding="utf-8"))
    assert conversion["source_tensors"] == len(source_tensors)
    assert conversion["native_tensors"] == len(base_tensors) + 1  # one gate injection

    weight_index = json.loads((partition / "transformer" / "model.safetensors.index.json").read_text(encoding="utf-8"))
    with safe_open(partition / "transformer" / "model-00001.safetensors", framework="pt", device="cpu") as shard:
        qkv = shard.get_tensor("blocks.0.attn.qkv_proj.weight")
        fc1 = shard.get_tensor("blocks.0.mlp.fc1.weight")
        loaded_gate = shard.get_tensor("blocks.0.attn.to_gate_compress.weight")
        loaded_rope = shard.get_tensor("rope.inv_freq")
    expected_qkv = torch.stack(
        [
            source_tensors["transformer_blocks.0.attn.to_q.weight"][0],
            source_tensors["transformer_blocks.0.attn.to_q.weight"][1],
            source_tensors["transformer_blocks.0.attn.to_k.weight"][0],
            source_tensors["transformer_blocks.0.attn.to_k.weight"][1],
            source_tensors["transformer_blocks.0.attn.to_v.weight"][0],
            source_tensors["transformer_blocks.0.attn.to_v.weight"][1],
            source_tensors["transformer_blocks.0.attn.to_q.weight"][2],
            source_tensors["transformer_blocks.0.attn.to_q.weight"][3],
            source_tensors["transformer_blocks.0.attn.to_k.weight"][2],
            source_tensors["transformer_blocks.0.attn.to_k.weight"][3],
            source_tensors["transformer_blocks.0.attn.to_v.weight"][2],
            source_tensors["transformer_blocks.0.attn.to_v.weight"][3],
        ]
    )
    torch.testing.assert_close(qkv, expected_qkv)
    torch.testing.assert_close(
        fc1,
        torch.cat(
            (
                source_tensors["transformer_blocks.0.ff.net.0.proj.weight"][3:],
                source_tensors["transformer_blocks.0.ff.net.0.proj.weight"][:3],
            )
        ),
    )
    torch.testing.assert_close(loaded_gate, source_tensors["transformer_blocks.0.attn.to_gate_compress.weight"])
    torch.testing.assert_close(loaded_rope, base_tensors["rope.inv_freq"])
    assert set(weight_index["weight_map"]) == {
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.mlp.fc1.weight",
        "blocks.0.attn.to_gate_compress.weight",
        "rope.inv_freq",
    }


def test_converter_rejects_bad_contract_before_publishing_output(tmp_path):
    source, base, _, _ = _make_converter_fixture(tmp_path)
    contract_path = source / "fastvideo_inference.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["video_scheduler_shift"] = 12.0
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    output = tmp_path / "bad-contract-output"

    with pytest.raises(ValueError, match="video_scheduler_shift"):
        prepare_checkpoint(source, base, output)
    assert not output.exists()


def test_converter_rejects_bad_gate_shape_before_publishing_output(tmp_path):
    source, base, _, _ = _make_converter_fixture(tmp_path)
    bad_gate = torch.zeros(3, 4)
    source_tensors = {
        "transformer_blocks.0.attn.to_q.weight": torch.zeros(4, 4),
        "transformer_blocks.0.attn.to_k.weight": torch.zeros(4, 4),
        "transformer_blocks.0.attn.to_v.weight": torch.zeros(4, 4),
        "transformer_blocks.0.ff.net.0.proj.weight": torch.zeros(6, 4),
        "transformer_blocks.0.attn.to_gate_compress.weight": bad_gate,
    }
    _write_indexed_safetensors(
        source / "transformer",
        "diffusion_pytorch_model.safetensors.index.json",
        source_tensors,
    )
    output = tmp_path / "bad-shape-output"

    with pytest.raises(ValueError, match="shape mismatch"):
        prepare_checkpoint(source, base, output)
    assert not output.exists()


def test_resolve_sigma_positions_uses_the_nine_node_native_schedule():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline._fasth3 = None
    pipeline._lora_sigma_schedules = {}
    pipeline._base_schedule_by_partition = {
        "fl2va": DMD2SigmaSchedule.from_metadata(_v2_release()),
    }

    positions, steps = pipeline._resolve_sigma_positions("t2va", _sampling())
    assert positions == FASTH3_V2_BASE_SCHEDULE.base_schedule
    assert steps == 8 == len(positions) - 1
    with pytest.raises(OmniClientError, match="must be 8"):
        pipeline._resolve_sigma_positions("t2va", _sampling(num_inference_steps=9))
