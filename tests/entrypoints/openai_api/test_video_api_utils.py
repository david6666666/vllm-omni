# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OpenAI-compatible video API encoding helpers."""

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.postprocess import rife_interpolator
from vllm_omni.entrypoints.openai import video_api_utils

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _install_fake_video_mux(monkeypatch, mux_calls):
    def _fake_mux_video_audio_bytes(frames, audio, fps, audio_sample_rate, video_codec_options=None):
        mux_calls.append(
            {
                "frames": frames,
                "audio": audio,
                "fps": fps,
                "audio_sample_rate": audio_sample_rate,
                "video_codec_options": video_codec_options,
            }
        )
        return b"fake-video"

    monkeypatch.setattr(
        "vllm_omni.diffusion.utils.media_utils.mux_video_audio_bytes",
        _fake_mux_video_audio_bytes,
    )


def test_encode_video_bytes_exports_frames_without_interpolation(monkeypatch):
    mux_calls = []
    _install_fake_video_mux(monkeypatch, mux_calls)

    frames = [np.full((2, 2, 3), fill_value=i / 5, dtype=np.float32) for i in range(5)]
    video_bytes = video_api_utils._encode_video_bytes(
        frames,
        fps=8,
    )

    assert video_bytes == b"fake-video"
    assert mux_calls[0]["frames"].shape == (5, 2, 2, 3)
    assert mux_calls[0]["frames"].dtype == np.uint8
    assert mux_calls[0]["fps"] == 8.0
    assert mux_calls[0]["audio"] is None


def test_encode_video_bytes_uses_direct_tensor_path(monkeypatch):
    mux_calls = []
    _install_fake_video_mux(monkeypatch, mux_calls)

    def _unexpected_stack(*args, **kwargs):
        raise AssertionError("tensor video path should not materialize a frame list with np.stack")

    monkeypatch.setattr(video_api_utils.np, "stack", _unexpected_stack)
    video = torch.linspace(0, 1, steps=3 * 5 * 2 * 2, dtype=torch.float32).reshape(3, 5, 2, 2)
    video_bytes = video_api_utils._encode_video_bytes(video, fps=8)

    assert video_bytes == b"fake-video"
    frames = mux_calls[0]["frames"]
    assert frames.shape == (5, 2, 2, 3)
    assert frames.dtype == np.uint8
    assert frames.flags.c_contiguous
    assert frames[0, 0, 0, 0] == 0
    assert frames[-1, -1, -1, -1] == 255


def test_encode_video_bytes_can_skip_tensor_range_probe(monkeypatch):
    mux_calls = []
    _install_fake_video_mux(monkeypatch, mux_calls)

    def _unexpected_item(*args, **kwargs):
        raise AssertionError("unit interval tensor path should not call Tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", _unexpected_item)
    video = torch.linspace(0, 1, steps=3 * 5 * 2 * 2, dtype=torch.float32).reshape(3, 5, 2, 2)
    video_bytes = video_api_utils._encode_video_bytes(
        video,
        fps=8,
        assume_unit_interval_tensor=True,
    )

    assert video_bytes == b"fake-video"
    frames = mux_calls[0]["frames"]
    assert frames.shape == (5, 2, 2, 3)
    assert frames.dtype == np.uint8
    assert frames[0, 0, 0, 0] == 0
    assert frames[-1, -1, -1, -1] == 255


def test_encode_video_bytes_pinned_transfer_option_preserves_cpu_tensor_output(monkeypatch):
    mux_calls = []
    _install_fake_video_mux(monkeypatch, mux_calls)

    video = torch.linspace(0, 1, steps=3 * 5 * 2 * 2, dtype=torch.float32).reshape(3, 5, 2, 2)
    video_bytes = video_api_utils._encode_video_bytes(
        video,
        fps=8,
        assume_unit_interval_tensor=True,
        use_pinned_host_transfer=True,
    )

    assert video_bytes == b"fake-video"
    frames = mux_calls[0]["frames"]
    assert frames.shape == (5, 2, 2, 3)
    assert frames.dtype == np.uint8
    assert frames.flags.c_contiguous
    assert frames[0, 0, 0, 0] == 0
    assert frames[-1, -1, -1, -1] == 255


def test_encode_video_bytes_direct_tensor_preserves_raw_minus_one_to_one_range(monkeypatch):
    mux_calls = []
    _install_fake_video_mux(monkeypatch, mux_calls)

    video = torch.tensor(
        [
            [[[-1.0]], [[1.0]]],
            [[[-1.0]], [[1.0]]],
            [[[-1.0]], [[1.0]]],
        ]
    )
    video_api_utils._encode_video_bytes(video, fps=8)

    frames = mux_calls[0]["frames"]
    assert frames.shape == (2, 1, 1, 3)
    assert frames[0, 0, 0].tolist() == [0, 0, 0]
    assert frames[1, 0, 0].tolist() == [255, 255, 255]


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
    monkeypatch.setattr(interpolator, "_ensure_model_loaded", lambda preferred_device=None: model)

    video = torch.zeros(1, 3, 2, 32, 32)
    output_video, multiplier = interpolator.interpolate_tensor(video, exp=1, scale=1.0)

    assert multiplier == 2
    assert output_video.shape == (1, 3, 3, 32, 32)
    assert torch.isfinite(output_video).all()


def test_frame_interpolator_uses_platform_device_when_tensor_is_cpu(monkeypatch):
    chosen_devices = []
    model = rife_interpolator.Model().eval()

    def _fake_ensure_model_loaded(*, preferred_device=None):
        chosen_devices.append(preferred_device)
        return model

    interpolator = rife_interpolator.FrameInterpolator()
    monkeypatch.setattr(interpolator, "_ensure_model_loaded", _fake_ensure_model_loaded)
    monkeypatch.setattr(model.flownet, "to", lambda device: model.flownet)
    monkeypatch.setattr(rife_interpolator, "_select_torch_device", lambda: torch.device("cuda"))

    video = torch.zeros(1, 3, 2, 32, 32)
    output_video, multiplier = interpolator.interpolate_tensor(video, exp=1, scale=1.0)

    assert chosen_devices == [torch.device("cuda")]
    assert multiplier == 2
    assert output_video.shape == (1, 3, 3, 32, 32)
