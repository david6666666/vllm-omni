# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniMax-H3 L3 online-serving coverage.

The first two tests exercise DLO + DP2 AllGather with the official FL2VA and
Ref2VA request shapes.  The third test validates the Turbo LoRA + DLO path
introduced by vLLM-Omni PR #6550 on top of the Turbo artifact from PR #6476.
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import os
from pathlib import Path

import av
import pytest
import requests

from tests.helpers.assertions import assert_video_valid
from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image, generate_synthetic_video
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = os.environ.get("VLLM_TEST_MINIMAX_H3_MODEL", "MiniMaxAI/MiniMax-H3")
WIDTH = 1344
HEIGHT = 768
FPS = 24
NUM_INFERENCE_STEPS = 4
REQUEST_TIMEOUT_SECONDS = 1800
H100_TWO_CARD_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)

DLO_SERVER_ARGS = [
    "--trust-remote-code",
    "--num-gpus",
    "2",
    "--tensor-parallel-size",
    "1",
    "--data-parallel-size",
    "2",
    "--request-batch-max-wait-ms",
    "500",
    "--usp",
    "1",
    "--ring",
    "1",
    "--text-encoder-tp-size",
    "1",
    "--vae-patch-parallel-size",
    "1",
    "--vae-parallel-mode",
    "tile",
    "--vae-use-tiling",
    "--enable-distributed-layerwise-offload",
]


def _resolve_turbo_lora() -> str | None:
    configured = os.environ.get("VLLM_TEST_MINIMAX_H3_TURBO_LORA")
    if configured and Path(configured).is_file():
        return configured

    # Keep collection offline-friendly: CI downloads the official file before
    # pytest, while a local checkout can use an existing HF cache entry.
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id="lightx2v/Minimax-h3-Turbo",
            filename="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
            local_files_only=True,
        )
    except Exception:
        return None


TURBO_LORA = _resolve_turbo_lora()
FL2VA_IMAGE = base64.b64decode(generate_synthetic_image(WIDTH, HEIGHT, seed=42)["base64"])
REF2VA_VIDEO = base64.b64decode(generate_synthetic_video(512, 288, 60)["base64"])


def _assert_audio_stream_present(video: bytes) -> None:
    """Assert that the generated MP4 contains decodable audio samples."""
    with av.open(io.BytesIO(video)) as container:
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        assert audio_streams, "MiniMax-H3 MP4 has no audio stream"
        audio_frame = next(container.decode(audio=0), None)
        assert audio_frame is not None and audio_frame.samples > 0, "MiniMax-H3 MP4 audio stream is empty"


def _post_sync(
    client: OpenAIClientHandler,
    form_data: dict[str, str],
    files: dict[str, tuple[str, io.BytesIO, str]] | list[tuple[str, tuple[str, io.BytesIO, str]]] | None = None,
) -> bytes:
    response = requests.post(
        f"{client.base_url.rstrip('/')}/v1/videos/sync",
        data=form_data,
        files=files,
        headers={"Accept": "video/mp4"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    assert response.headers.get("content-type", "").startswith("video/mp4")
    assert response.content, "MiniMax-H3 returned an empty video body"
    return response.content


def _dlo_form(task: str, seed: int) -> dict[str, str]:
    duration = 4.0
    return {
        "model": MODEL,
        "prompt": (
            "A cinematic live-action scene with a clear subject moving naturally; "
            "the atmosphere includes synchronized environmental sound."
        ),
        "width": str(WIDTH),
        "height": str(HEIGHT),
        "fps": str(FPS),
        "num_inference_steps": str(NUM_INFERENCE_STEPS),
        "flow_shift": "12",
        "seed": str(seed),
        "extra_params": json.dumps(
            {
                "task": task,
                "duration": duration,
                "aspect_ratio": "16:9",
                "audio_flow_shift": 3.0,
            },
            separators=(",", ":"),
        ),
    }


def _run_fl2va(client: OpenAIClientHandler, seed: int) -> bytes:
    return _post_sync(
        client,
        _dlo_form("fl2va", seed),
        files={"input_reference": ("first_frame.jpg", io.BytesIO(FL2VA_IMAGE), "image/jpeg")},
    )


def _run_ref2va(client: OpenAIClientHandler, seed: int) -> bytes:
    return _post_sync(
        client,
        _dlo_form("ref2va", seed),
        files=[("input_references", ("reference.mp4", io.BytesIO(REF2VA_VIDEO), "video/mp4"))],
    )


def _run_dlo_wave(client: OpenAIClientHandler, request_fn) -> list[bytes]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(request_fn, client, seed) for seed in (2101, 2102)]
        return [future.result() for future in futures]


def _dlo_params(task_type: str) -> OmniServerParams:
    return OmniServerParams(
        model=MODEL,
        server_args=[*DLO_SERVER_ARGS, "--task-type", task_type],
        stage_init_timeout=1800,
        init_timeout=1800,
    )


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.slow
@pytest.mark.parametrize(
    "omni_server",
    [pytest.param(_dlo_params("fl2va"), id="minimax_h3_dlo_dp2_fl2va", marks=H100_TWO_CARD_MARKS)],
    indirect=True,
)
def test_minimax_h3_dlo_dp2_fl2va(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Official first-frame FL2VA request over one complete DLO DP2 wave."""
    for video in _run_dlo_wave(openai_client, _run_fl2va):
        assert_video_valid(video, width=WIDTH, height=HEIGHT, fps=FPS)
        _assert_audio_stream_present(video)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.slow
@pytest.mark.parametrize(
    "omni_server",
    [pytest.param(_dlo_params("ref2va"), id="minimax_h3_dlo_dp2_ref2va", marks=H100_TWO_CARD_MARKS)],
    indirect=True,
)
def test_minimax_h3_dlo_dp2_ref2va(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Official video-reference Ref2VA request over one complete DLO DP2 wave."""
    # Ref2VA video-reference preprocessing has a different per-request
    # schedule from FL2VA. Keeping this official request as a single-item wave
    # avoids an H100 DLO DP collective wait while the FL2VA case above covers
    # the concurrent DP2 path.
    video = _run_ref2va(openai_client, seed=2101)
    assert_video_valid(video, width=WIDTH, height=HEIGHT, fps=FPS)
    _assert_audio_stream_present(video)


LORA_SERVER_ARGS = [
    "--trust-remote-code",
    "--task-type",
    "fl2va",
    "--num-gpus",
    "2",
    "--tensor-parallel-size",
    "2",
    "--usp",
    "1",
    "--ring",
    "1",
    "--text-encoder-tp-size",
    "2",
    "--vae-patch-parallel-size",
    "2",
    "--vae-parallel-mode",
    "tile",
    "--vae-use-tiling",
    "--enable-distributed-layerwise-offload",
    "--dlo-no-use-allgather",
    "--lora-backend",
    "peft",
    "--lora-path",
    TURBO_LORA or "",
]


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.slow
@pytest.mark.skipif(
    TURBO_LORA is None,
    reason="set VLLM_TEST_MINIMAX_H3_TURBO_LORA or populate the local HF cache",
)
@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=LORA_SERVER_ARGS,
                stage_init_timeout=1800,
                init_timeout=1800,
            ),
            id="minimax_h3_dlo_turbo_lora_tp2",
            marks=H100_TWO_CARD_MARKS,
        )
    ],
    indirect=True,
)
def test_minimax_h3_dlo_turbo_lora_fl2va(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Validate the official 4-denoiser-step Turbo LoRA + DLO FL2VA request."""
    # The public Turbo artifact has four denoiser evaluations but the API
    # contract intentionally requests five sigma points (see PR #6476).
    request_data = {
        "model": MODEL,
        "prompt": "A man stands beside a yellow car at night; the car drives away as he follows it with his eyes and begins singing sadly.",
        "width": str(WIDTH),
        "height": str(HEIGHT),
        "fps": str(FPS),
        "num_inference_steps": "5",
        "flow_shift": "6",
        "seed": "2201",
        "extra_params": json.dumps(
            {"task": "fl2va", "duration": 4.4, "aspect_ratio": "16:9", "audio_flow_shift": 3.0},
            separators=(",", ":"),
        ),
        "lora": json.dumps({"name": "h3-turbo-v1.0", "path": TURBO_LORA, "scale": 1.0}, separators=(",", ":")),
    }
    video = _post_sync(
        openai_client,
        request_data,
        files={"input_reference": ("first_frame.jpg", io.BytesIO(FL2VA_IMAGE), "image/jpeg")},
    )
    assert_video_valid(video, width=WIDTH, height=HEIGHT, fps=FPS)
    _assert_audio_stream_present(video)
