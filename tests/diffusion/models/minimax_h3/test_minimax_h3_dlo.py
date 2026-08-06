# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MiniMax-H3 distributed layerwise offload (DLO) function tests.

The 2x H100 (80 GiB/GPU) case is a memory-first serving configuration
validated on B300 hardware with DLO streaming the non-resident DiT blocks
from pinned host memory. With TP2 + encoder TP2 + DLO (``dlo_resident_layers=8``,
``--dlo-no-use-allgather``) and a 2-step smoke request, the per-GPU peak was
~42 GiB, far below the 80 GiB H100 budget.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from huggingface_hub import snapshot_download

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.diffusion]

MODEL_REPO_ID = "MiniMaxAI/MiniMax-H3"
MODEL_REVISION = "main"
MODEL_ENV_VAR = "VLLM_TEST_MINIMAX_H3_FL2VA_MODEL"


def _fl2va_model_path() -> str:
    configured = os.environ.get(MODEL_ENV_VAR)
    if configured:
        return configured
    snapshot_root = snapshot_download(
        repo_id=MODEL_REPO_ID,
        revision=MODEL_REVISION,
        allow_patterns=["FL2VA/**"],
    )
    return str(Path(snapshot_root) / "FL2VA")


@pytest.mark.full_model
@pytest.mark.parametrize(
    "parallel_config",
    [
        pytest.param(
            {
                "tensor_parallel_size": 2,
                "text_encoder_tp_size": 2,
                "ulysses_degree": 1,
                "ring_degree": 1,
                "vae_patch_parallel_size": 1,
            },
            marks=hardware_marks(res={"cuda": "H100"}, num_cards=2),
            id="dlo-2xH100",
        ),
    ],
)
def test_minimax_h3_t2va_dlo_2card_h100_smoke(parallel_config: dict):
    """2-card DLO smoke: small steps, verifies DLO works on 2x H100."""
    import torch

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    if torch.accelerator.device_count() < 2:
        pytest.skip("MiniMax H3 DLO smoke requires two GPUs")

    engine = Omni(
        model=_fl2va_model_path(),
        parallel_config=DiffusionParallelConfig(**parallel_config),
        trust_remote_code=True,
        enable_distributed_layerwise_offload=True,
        dlo_use_allgather=False,
        dlo_resident_layers=8,
        vae_parallel_mode="tile",
        vae_use_tiling=True,
        enforce_eager=True,
        diffusion_attention_backend="CUDNN_ATTN",
    )
    try:
        outputs = engine.generate(
            "A quiet cinematic night scene with matching ambient sound.",
            OmniDiffusionSamplingParams(
                height=256,
                width=448,
                num_frames=29,
                fps=24,
                num_inference_steps=2,
                seed=42,
                output_type="np",
                extra_args={
                    "task": "t2va",
                    "duration": 4.0,
                    "aspect_ratio": "16:9",
                    "flow_shift": 12.0,
                    "audio_flow_shift": 3.0,
                },
            ),
            use_tqdm=False,
        )
    finally:
        engine.close()

    assert len(outputs) == 1
    frames = np.asarray(outputs[0].images[0])
    assert frames.shape == (107, 256, 448, 3)
    multimodal = outputs[0].multimodal_output
    assert multimodal is not None
    assert np.asarray(multimodal["audio"]).shape[1] == 2
    assert multimodal["audio_sample_rate"] == 32000
    assert multimodal["fps"] == 24
