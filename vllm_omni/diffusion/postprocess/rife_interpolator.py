# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RIFE 4.22.lite frame interpolation for vLLM-Omni video generation.

RIFE model code is vendored and adapted from:
  - https://github.com/hzwer/ECCV2022-RIFE  (MIT License)
  - https://github.com/hzwer/Practical-RIFE  (MIT License)
  Copyright (c) 2021 Zhewei Huang

The FrameInterpolator wrapper and vLLM-Omni integration code are original work.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_RIFE_HF_REPO = "elfgum/RIFE-4.22.lite"
_MODEL_CACHE: dict[str, Model] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def warp(ten_input: torch.Tensor, ten_flow: torch.Tensor) -> torch.Tensor:
    """Warp input tensor by optical flow using grid_sample."""
    ten_horizontal = (
        torch.linspace(-1.0, 1.0, ten_flow.shape[3], device=ten_flow.device)
        .view(1, 1, 1, ten_flow.shape[3])
        .expand(ten_flow.shape[0], -1, ten_flow.shape[2], -1)
    )
    ten_vertical = (
        torch.linspace(-1.0, 1.0, ten_flow.shape[2], device=ten_flow.device)
        .view(1, 1, ten_flow.shape[2], 1)
        .expand(ten_flow.shape[0], -1, -1, ten_flow.shape[3])
    )
    ten_grid = torch.cat([ten_horizontal, ten_vertical], dim=1)

    ten_flow = torch.cat(
        [
            ten_flow[:, 0:1, :, :] / ((ten_input.shape[3] - 1.0) / 2.0),
            ten_flow[:, 1:2, :, :] / ((ten_input.shape[2] - 1.0) / 2.0),
        ],
        dim=1,
    )
    grid = (ten_grid + ten_flow).permute(0, 2, 3, 1)
    return F.grid_sample(
        input=ten_input,
        grid=grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def _conv(
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.2, True),
    )


class ResConv(nn.Module):
    """Residual convolution block with learnable beta scaling."""

    def __init__(self, c: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    """Single-scale optical flow, mask, and feature block."""

    def __init__(self, in_planes: int, c: int = 64):
        super().__init__()
        self.conv0 = nn.Sequential(
            _conv(in_planes, c // 2, 3, 2, 1),
            _conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(
        self,
        x: torch.Tensor,
        flow: torch.Tensor | None = None,
        scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = (
                F.interpolate(
                    flow,
                    scale_factor=1.0 / scale,
                    mode="bilinear",
                    align_corners=False,
                )
                * 1.0
                / scale
            )
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow, mask, feat


class Head(nn.Module):
    """Feature encoder producing four-channel features at full resolution."""

    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        return x3


class IFNet(nn.Module):
    """Four-scale IFNet optical flow network."""

    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=32)
        self.encode = Head()

    def forward(
        self,
        x: torch.Tensor,
        timestep: float = 0.5,
        scale_list: list[float] | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | torch.Tensor]]:
        if scale_list is None:
            scale_list = [8, 4, 2, 1]

        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]

        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])

        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])

        flow_list: list[torch.Tensor] = []
        merged: list[tuple[torch.Tensor, torch.Tensor] | torch.Tensor] = []
        mask_list: list[torch.Tensor] = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None

        for i, block in enumerate([self.block0, self.block1, self.block2, self.block3]):
            if flow is None:
                flow, mask, feat = block(
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1),
                    None,
                    scale=scale_list[i],
                )
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, m0, feat = block(
                    torch.cat(
                        (
                            warped_img0[:, :3],
                            warped_img1[:, :3],
                            wf0,
                            wf1,
                            timestep,
                            mask,
                            feat,
                        ),
                        1,
                    ),
                    flow,
                    scale=scale_list[i],
                )
                mask = m0
                flow = flow + fd

            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))

        mask = torch.sigmoid(mask)
        merged[3] = warped_img0 * mask + warped_img1 * (1 - mask)
        return flow_list, mask_list[3], merged


class Model:
    """Wraps IFNet and exposes RIFE-compatible load/inference helpers."""

    def __init__(self):
        self.flownet = IFNet()

    def eval(self) -> Model:
        self.flownet.eval()
        return self

    def device(self) -> torch.device:
        return next(self.flownet.parameters()).device

    def load_model(self, path: str) -> None:
        flownet_path = os.path.join(path, "flownet.pkl")
        if not os.path.isfile(flownet_path):
            raise FileNotFoundError(
                f"RIFE weight file not found: {flownet_path}. Expected layout: <model_path>/flownet.pkl"
            )

        state = torch.load(flownet_path, map_location="cpu", weights_only=False)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.flownet.load_state_dict(state, strict=False)
        logger.info("Loaded RIFE weights from %s", flownet_path)

    def inference(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        scale: float = 1.0,
        timestep: float = 0.5,
    ) -> torch.Tensor:
        _n, _c, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        pad = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, pad)
        img1 = F.pad(img1, pad)

        imgs = torch.cat((img0, img1), 1)
        scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        with torch.no_grad():
            _flow_list, _mask, merged = self.flownet(
                imgs,
                timestep=timestep,
                scale_list=scale_list,
            )
        return merged[3][:, :, :h, :w]


def _resolve_rife_model_path(model_path: str | None) -> str:
    model_path = model_path or _DEFAULT_RIFE_HF_REPO
    if os.path.isdir(model_path):
        return model_path
    from vllm_omni.model_executor.model_loader.weight_utils import (
        download_weights_from_hf_specific,
    )

    return download_weights_from_hf_specific(
        model_path,
        cache_dir=None,
        allow_patterns=["flownet.pkl"],
        require_all=True,
    )


def _select_torch_device() -> torch.device:
    try:
        from vllm_omni.platforms import current_omni_platform

        return current_omni_platform.get_torch_device()
    except Exception as exc:
        logger.warning("Failed to resolve current vLLM-Omni torch device: %s", exc)

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FrameInterpolator:
    """Lazy-loaded RIFE 4.22.lite frame interpolator."""

    def __init__(self, model_path: str | None = None):
        self._model_path = model_path
        self._resolved_path: str | None = None

    def _ensure_model_loaded(self) -> Model:
        resolved_path = _resolve_rife_model_path(self._model_path)
        self._resolved_path = resolved_path

        with _MODEL_CACHE_LOCK:
            if resolved_path in _MODEL_CACHE:
                return _MODEL_CACHE[resolved_path]

            device = _select_torch_device()
            model = Model()
            model.load_model(resolved_path)
            model.eval()
            model.flownet = model.flownet.to(device)
            _MODEL_CACHE[resolved_path] = model
            logger.info("RIFE model loaded on device: %s", device)
            return model

    @staticmethod
    def _frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return tensor.to(device)

    @staticmethod
    def _tensor_to_frame(tensor: torch.Tensor) -> np.ndarray:
        arr = tensor.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
        return (arr * 255.0).round().clip(0, 255).astype(np.uint8)

    def _make_inference(
        self,
        model: Model,
        img0: torch.Tensor,
        img1: torch.Tensor,
        n: int,
        scale: float,
    ) -> list[torch.Tensor]:
        if n == 1:
            return [model.inference(img0, img1, scale=scale)]
        mid = model.inference(img0, img1, scale=scale)
        return (
            self._make_inference(model, img0, mid, n // 2, scale)
            + [mid]
            + self._make_inference(model, mid, img1, n // 2, scale)
        )

    def interpolate(
        self,
        frames: list[np.ndarray],
        exp: int = 1,
        scale: float = 1.0,
    ) -> tuple[list[np.ndarray], int]:
        if exp < 1:
            raise ValueError(f"frame interpolation exp must be >= 1, got {exp}")
        if scale <= 0:
            raise ValueError(f"frame interpolation scale must be > 0, got {scale}")
        if len(frames) < 2:
            logger.warning("Frame interpolation requires at least 2 frames; returning input unchanged.")
            return frames, 1

        model = self._ensure_model_loaded()
        device = model.device()
        intermediates_per_pair = 2**exp // 2

        result: list[np.ndarray] = []
        for idx in range(len(frames) - 1):
            img0 = self._frame_to_tensor(frames[idx], device)
            img1 = self._frame_to_tensor(frames[idx + 1], device)
            result.append(frames[idx])
            for tensor in self._make_inference(model, img0, img1, intermediates_per_pair, scale):
                result.append(self._tensor_to_frame(tensor))
        result.append(frames[-1])
        return result, 2**exp


def interpolate_video_frames(
    frames: list[np.ndarray],
    exp: int = 1,
    scale: float = 1.0,
    model_path: str | None = None,
) -> tuple[list[np.ndarray], int]:
    """Interpolate a list of uint8 HWC frames and return the FPS multiplier."""
    interpolator = FrameInterpolator(model_path=model_path)
    return interpolator.interpolate(frames, exp=exp, scale=scale)
