# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sampling contract for converted FastH3 full checkpoints."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule
from vllm_omni.errors import OmniClientError

from .fasth3 import _resolve_dit_attention_backend

FASTH3_V2_MODEL_ID = "FastVideo/FastVideo-FastH3-8-Step-V2"
FASTH3_V2_BASE_SCHEDULE = DMD2SigmaSchedule.from_positions((0.999, 0.874, 0.749, 0.624, 0.5, 0.375, 0.25, 0.125, 0.0))


@dataclass(frozen=True)
class FastH3CheckpointSpec:
    """The full V2 release has its own schedule and trained attention policy.

    This is independent of ``FastH3WeightFusion``: the converted weights already
    contain the entire student, including its learned compression gates.
    """

    vsa_sparsity: float = 0.8

    @classmethod
    def from_metadata(cls, release: Mapping[str, Any]) -> FastH3CheckpointSpec | None:
        metadata = release.get("fasth3")
        if metadata is None:
            return None
        if not isinstance(metadata, Mapping) or metadata.get("model_id") != FASTH3_V2_MODEL_ID:
            raise ValueError("unsupported FastH3 full checkpoint identity")
        if metadata.get("vsa_sparsity") != 0.8 or metadata.get("vsa_tile_size") != 64:
            raise ValueError("FastH3 V2 requires VSA sparsity=0.8 and tile_size=64")
        if DMD2SigmaSchedule.from_metadata(release) != FASTH3_V2_BASE_SCHEDULE:
            raise ValueError("FastH3 V2 requires its exact eight-step base_schedule")
        if release.get("sigma_shift_scales") != {"video": 10.0, "audio": 3.0}:
            raise ValueError("FastH3 V2 requires video/audio sigma shifts 10/3")
        if release.get("partition") != "fl2va" or set(release.get("tasks", ())) != {"t2va"}:
            raise ValueError("FastH3 V2 supports T2VA in the FL2VA partition only")
        return cls()

    def check_serving_contract(self, *, partition: str, od_config: Any) -> None:
        if partition != "fl2va":
            raise ValueError("FastH3 V2 requires --task-type fl2va (T2VA requests only)")
        if getattr(od_config, "lora_path", None):
            raise ValueError("FastH3 V2 is a full checkpoint; additional LoRA adapters are unsupported")
        if _resolve_dit_attention_backend(od_config) != "FASTVIDEO_VSA":
            raise ValueError("FastH3 V2 requires --diffusion-attention-backend FASTVIDEO_VSA")
        attention_config = getattr(od_config, "diffusion_attention_config", None)
        per_role = getattr(attention_config, "per_role", None) or {}
        spec = per_role.get("self") or getattr(attention_config, "default", None)
        if getattr(spec, "fastvideo_vsa_topk", None) is not None:
            raise ValueError("FastH3 V2 pins VSA sparsity=0.8; remove the fixed fastvideo_vsa_topk override")
        parallel = getattr(od_config, "parallel_config", None)
        if any(int(getattr(parallel, key, 1) or 1) != 1 for key in ("ring_degree", "allgather_degree")):
            raise ValueError("FastH3 V2 supports local attention or pure Ulysses sequence parallelism")

    def check_request(self, sampling: Any) -> None:
        if sampling.lora_request is not None:
            raise OmniClientError("FastH3 V2 does not support per-request LoRA adapters")
        steps = sampling.num_inference_steps
        if steps is not None and steps != FASTH3_V2_BASE_SCHEDULE.num_inference_steps:
            raise OmniClientError("FastH3 V2 requires num_inference_steps=8 (nine sigma points), or omitted")
        extra = sampling.extra_args or {}
        for key, expected in (("flow_shift", 10.0), ("audio_flow_shift", 3.0)):
            try:
                value = float(extra.get(key, expected))
            except (TypeError, ValueError) as exc:
                raise OmniClientError(f"FastH3 V2 requires {key}={expected:g}") from exc
            if not math.isclose(value, expected):
                raise OmniClientError(f"FastH3 V2 requires {key}={expected:g}, got {value:g}")
        if sampling.guidance_scale is not None and sampling.guidance_scale != 1.0:
            raise OmniClientError("FastH3 V2 requires guidance_scale=1")
