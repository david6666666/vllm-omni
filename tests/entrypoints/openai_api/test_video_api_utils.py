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


def test_encode_video_bytes_exports_frames_without_interpolation(monkeypatch):
    export_calls = []
    _install_fake_diffusers_export(monkeypatch, export_calls)

    frames = [np.full((2, 2, 3), fill_value=i / 5, dtype=np.float32) for i in range(5)]
    video_bytes = video_api_utils._encode_video_bytes(
        frames,
        fps=8,
    )

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


def test_frame_interpolator_runs_actual_torch_tensor_path(monkeypatch):
    model = rife_interpolator.Model().eval()
    interpolator = rife_interpolator.FrameInterpolator()
    monkeypatch.setattr(interpolator, "_ensure_model_loaded", lambda: model)

    video = torch.zeros(1, 3, 2, 32, 32)
    output_video, multiplier = interpolator.interpolate_tensor(video, exp=1, scale=1.0)

    assert multiplier == 2
    assert output_video.shape == (1, 3, 3, 32, 32)
    assert torch.isfinite(output_video).all()
