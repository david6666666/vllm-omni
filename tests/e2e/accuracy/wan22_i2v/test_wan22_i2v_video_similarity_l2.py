from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import requests
import torch

from tests.conftest import OmniServerParams
from tests.e2e.accuracy.wan22_i2v.wan22_i2v_video_similarity_common import (
    FLOW_SHIFT,
    FPS,
    GUIDANCE_SCALE,
    GUIDANCE_SCALE_2,
    HEIGHT,
    MODEL_NAME,
    NEGATIVE_PROMPT,
    NUM_FRAMES,
    NUM_INFERENCE_STEPS,
    PROMPT,
    PSNR_THRESHOLD,
    RABBIT_IMAGE_URL,
    SEED,
    SIZE,
    SSIM_THRESHOLD,
    WIDTH,
)
from tests.utils import hardware_test


def test_parse_video_metadata_extracts_dimensions_and_fps() -> None:
    payload = {
        "streams": [
            {
                "width": 832,
                "height": 480,
                "avg_frame_rate": "16/1",
                "nb_read_frames": "93",
            }
        ]
    }

    metadata = _parse_video_metadata(payload)

    assert metadata["width"] == 832
    assert metadata["height"] == 480
    assert metadata["fps"] == 16.0
    assert metadata["frame_count"] == 93


def test_parse_ssim_summary_extracts_all_score() -> None:
    output = """
    [Parsed_ssim_0 @ 000001] SSIM Y:0.971903 (15.512007) U:0.965077 (14.569044) V:0.962414 (14.252637) All:0.968311 (15.035654)
    """

    assert _parse_ssim_score(output) == 0.968311


def test_parse_psnr_summary_extracts_average_score() -> None:
    output = """
    [Parsed_psnr_0 @ 000001] PSNR y:32.670157 u:31.844621 v:31.513839 average:32.148004 min:31.563744 max:33.201457
    """

    assert _parse_psnr_score(output) == 32.148004


def test_build_diffusers_command_uses_torchrun_and_runner_path(tmp_path: Path) -> None:
    runner_path = tmp_path / "run_wan22_i2v_diffusers_cp.py"
    command = _build_diffusers_command(
        runner_path=runner_path,
        output_path=tmp_path / "offline.mp4",
        metadata_path=tmp_path / "offline.json",
    )

    assert command[:5] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        "2",
    ]
    assert command[5] == str(runner_path)
    assert "--output" in command
    assert "--metadata-output" in command


REPO_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = REPO_ROOT.parent
RUNNER_PATH = Path(__file__).with_name("run_wan22_i2v_diffusers_cp.py")
VIDEO_TIMEOUT_SECONDS = 60 * 60
_SSIM_RE = re.compile(r"All:(?P<score>[0-9.]+)")
_PSNR_RE = re.compile(r"average:(?P<score>[0-9.]+)")

SERVER_CASES = [
    pytest.param(
        OmniServerParams(
            model=MODEL_NAME,
            server_args=[
                "--usp",
                "2",
                "--use-hsdp",
                "--hsdp-shard-size",
                "2",
            ],
            use_omni=True,
        ),
        id="wan22_i2v_usp2_hsdp2",
    )
]


def _parse_video_metadata(payload: dict[str, Any]) -> dict[str, int | float]:
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError(f"ffprobe payload did not include video streams: {payload}")

    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps = float(Fraction(stream["avg_frame_rate"]))
    frame_count_value = stream.get("nb_read_frames") or stream.get("nb_frames")
    if frame_count_value is None:
        raise ValueError(f"ffprobe payload did not include frame count: {payload}")
    frame_count = int(frame_count_value)
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
    }


def _parse_ssim_score(output: str) -> float:
    match = _SSIM_RE.search(output)
    if match is None:
        raise ValueError(f"Could not parse SSIM score from ffmpeg output:\n{output}")
    return float(match.group("score"))


def _parse_psnr_score(output: str) -> float:
    match = _PSNR_RE.search(output)
    if match is None:
        raise ValueError(f"Could not parse PSNR score from ffmpeg output:\n{output}")
    return float(match.group("score"))


def _probe_binary(binary: str) -> str:
    resolved = shutil.which(binary)
    if resolved is None:
        pytest.skip(f"{binary} is required for Wan2.2 video similarity e2e test.")
    return resolved


def _build_diffusers_command(
    *,
    runner_path: Path,
    output_path: Path,
    metadata_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        "2",
        str(runner_path),
        "--model",
        MODEL_NAME,
        "--image-url",
        RABBIT_IMAGE_URL,
        "--prompt",
        PROMPT,
        "--negative-prompt",
        NEGATIVE_PROMPT,
        "--size",
        SIZE,
        "--fps",
        str(FPS),
        "--num-frames",
        str(NUM_FRAMES),
        "--guidance-scale",
        str(GUIDANCE_SCALE),
        "--guidance-scale-2",
        str(GUIDANCE_SCALE_2),
        "--flow-shift",
        str(FLOW_SHIFT),
        "--num-inference-steps",
        str(NUM_INFERENCE_STEPS),
        "--seed",
        str(SEED),
        "--output",
        str(output_path),
        "--metadata-output",
        str(metadata_path),
    ]


def _runner_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [
        str(REPO_ROOT),
        str(WORKSPACE_ROOT / "diffusers" / "src"),
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return env


def _fetch_remote_image(url: str) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()


def _probe_video(path: Path) -> dict[str, int | float]:
    _probe_binary("ffprobe")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return _parse_video_metadata(json.loads(result.stdout))


def _run_ffmpeg_similarity(filter_name: str, first: Path, second: Path) -> str:
    _probe_binary("ffmpeg")
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(first),
            "-i",
            str(second),
            "-lavfi",
            f"[0:v][1:v]{filter_name}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stderr


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_wan22_i2v_serving_matches_diffusers_video_similarity(
    omni_server,
    openai_client,
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Wan2.2 I2V similarity e2e test requires >= 2 CUDA GPUs.")

    _probe_binary("ffmpeg")
    _probe_binary("ffprobe")
    if not RUNNER_PATH.exists():
        raise AssertionError(f"Offline diffusers runner does not exist: {RUNNER_PATH}")

    _fetch_remote_image(RABBIT_IMAGE_URL)

    online_path = tmp_path / "online.mp4"
    offline_path = tmp_path / "offline.mp4"
    offline_metadata_path = tmp_path / "offline_metadata.json"

    request_config = {
        "model": omni_server.model,
        "form_data": {
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "size": SIZE,
            "fps": FPS,
            "num_frames": NUM_FRAMES,
            "guidance_scale": GUIDANCE_SCALE,
            "guidance_scale_2": GUIDANCE_SCALE_2,
            "flow_shift": FLOW_SHIFT,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "seed": SEED,
        },
        "image_reference": RABBIT_IMAGE_URL,
    }
    online_response = openai_client.send_video_diffusion_request(request_config)[0]
    assert online_response.videos is not None and len(online_response.videos) == 1
    online_path.write_bytes(online_response.videos[0])

    command = _build_diffusers_command(
        runner_path=RUNNER_PATH,
        output_path=offline_path,
        metadata_path=offline_metadata_path,
    )
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_runner_env(),
        check=True,
        timeout=VIDEO_TIMEOUT_SECONDS,
    )

    assert online_path.exists(), f"Expected online video artifact at {online_path}"
    assert offline_path.exists(), f"Expected offline video artifact at {offline_path}"
    assert offline_metadata_path.exists(), f"Expected offline metadata artifact at {offline_metadata_path}"

    online_metadata = _probe_video(online_path)
    offline_metadata = _probe_video(offline_path)
    assert online_metadata == offline_metadata, (
        f"Video metadata mismatch:\n"
        f"online={online_metadata}\n"
        f"offline={offline_metadata}\n"
        f"online_path={online_path}\n"
        f"offline_path={offline_path}"
    )
    assert online_metadata["width"] == WIDTH
    assert online_metadata["height"] == HEIGHT
    assert online_metadata["fps"] == float(FPS)
    assert online_metadata["frame_count"] == NUM_FRAMES

    ssim_output = _run_ffmpeg_similarity("ssim", online_path, offline_path)
    psnr_output = _run_ffmpeg_similarity("psnr", online_path, offline_path)
    ssim_score = _parse_ssim_score(ssim_output)
    psnr_score = _parse_psnr_score(psnr_output)

    print(f"wan22_i2v_similarity: ssim={ssim_score:.6f}, psnr={psnr_score:.6f}")
    print(f"online_video={online_path}")
    print(f"offline_video={offline_path}")
    print(f"offline_metadata={offline_metadata_path}")

    assert ssim_score >= SSIM_THRESHOLD, (
        f"SSIM below threshold: got {ssim_score:.6f}, expected >= {SSIM_THRESHOLD:.6f}. "
        f"online={online_path} offline={offline_path}"
    )
    assert psnr_score >= PSNR_THRESHOLD, (
        f"PSNR below threshold: got {psnr_score:.6f}, expected >= {PSNR_THRESHOLD:.6f}. "
        f"online={online_path} offline={offline_path}"
    )
