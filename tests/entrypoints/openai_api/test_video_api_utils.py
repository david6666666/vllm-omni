# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OpenAI-compatible video API encoding helpers."""

import sys
import types

import numpy as np
import pytest

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


def test_frame_interpolation_auto_device_uses_cpu_on_non_cuda_platform(monkeypatch):
    class FakeNPUPlatform:
        @staticmethod
        def is_cuda():
            return False

        @staticmethod
        def is_rocm():
            return False

    monkeypatch.delenv("VLLM_OMNI_FRAME_INTERPOLATION_DEVICE", raising=False)
    monkeypatch.setattr(rife_interpolator.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        rife_interpolator,
        "_get_current_omni_platform",
        lambda: FakeNPUPlatform(),
    )

    assert rife_interpolator._select_torch_device().type == "cpu"


def test_frame_interpolation_device_env_overrides_auto(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FRAME_INTERPOLATION_DEVICE", "cpu")

    assert rife_interpolator._select_torch_device().type == "cpu"
