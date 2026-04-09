# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OpenAI-compatible video API encoding helpers."""

import sys
import types

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.postprocess import rife_interpolator
from vllm_omni.entrypoints.openai import video_api_utils

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _install_fake_diffusers_export(monkeypatch, export_calls):
    diffusers_module = types.ModuleType("diffusers")
    utils_module = types.ModuleType("diffusers.utils")

    def _fake_export_to_video(frames, output_path, fps):
        export_calls.append({"frames": frames, "fps": fps})
        with open(output_path, "wb") as f:
            f.write(b"fake-video")

    utils_module.export_to_video = _fake_export_to_video
    monkeypatch.setitem(sys.modules, "diffusers", diffusers_module)
    monkeypatch.setitem(sys.modules, "diffusers.utils", utils_module)


def test_encode_video_bytes_interpolates_frames_and_scales_fps(monkeypatch):
    export_calls = []
    interpolate_calls = []
    _install_fake_diffusers_export(monkeypatch, export_calls)

    def _fake_interpolate_video_frames(frames, exp, scale, model_path):
        interpolate_calls.append(
            {
                "frames": frames,
                "exp": exp,
                "scale": scale,
                "model_path": model_path,
            }
        )
        return frames[:1] * 9, 4

    monkeypatch.setattr(
        video_api_utils,
        "interpolate_video_frames",
        _fake_interpolate_video_frames,
        raising=False,
    )

    frames = [np.full((2, 2, 3), fill_value=i / 5, dtype=np.float32) for i in range(5)]
    video_bytes = video_api_utils._encode_video_bytes(
        frames,
        fps=8,
        enable_frame_interpolation=True,
        frame_interpolation_exp=2,
        frame_interpolation_scale=0.5,
        frame_interpolation_model_path="local-rife",
    )

    assert video_bytes == b"fake-video"
    assert interpolate_calls
    assert interpolate_calls[0]["exp"] == 2
    assert interpolate_calls[0]["scale"] == 0.5
    assert interpolate_calls[0]["model_path"] == "local-rife"
    assert len(export_calls[0]["frames"]) == 9
    assert export_calls[0]["fps"] == 32


def test_encode_video_bytes_skips_interpolation_when_disabled(monkeypatch):
    export_calls = []
    _install_fake_diffusers_export(monkeypatch, export_calls)

    def _unexpected_interpolation(*args, **kwargs):
        raise AssertionError("interpolation should not run")

    monkeypatch.setattr(
        video_api_utils,
        "interpolate_video_frames",
        _unexpected_interpolation,
        raising=False,
    )

    frames = [np.zeros((2, 2, 3), dtype=np.float32) for _ in range(5)]
    video_bytes = video_api_utils._encode_video_bytes(frames, fps=8)

    assert video_bytes == b"fake-video"
    assert len(export_calls[0]["frames"]) == 5
    assert export_calls[0]["fps"] == 8


def test_rife_model_inference_runs_on_dummy_tensors():
    model = rife_interpolator.Model().eval()
    img0 = torch.rand(1, 3, 32, 32)
    img1 = torch.rand(1, 3, 32, 32)

    output = model.inference(img0, img1, scale=1.0)

    assert output.shape == (1, 3, 32, 32)
    assert torch.isfinite(output).all()


def test_frame_interpolator_runs_actual_torch_path(monkeypatch):
    model = rife_interpolator.Model().eval()
    interpolator = rife_interpolator.FrameInterpolator()
    monkeypatch.setattr(interpolator, "_ensure_model_loaded", lambda: model)

    frames = [
        np.zeros((32, 32, 3), dtype=np.uint8),
        np.full((32, 32, 3), 255, dtype=np.uint8),
    ]
    output_frames, multiplier = interpolator.interpolate(frames, exp=1, scale=1.0)

    assert multiplier == 2
    assert len(output_frames) == 3
    assert output_frames[1].shape == (32, 32, 3)
    assert output_frames[1].dtype == np.uint8
