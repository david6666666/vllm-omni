# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 remote-code VAE adapters and exact latent contracts."""

from __future__ import annotations

import importlib
import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedVaeMixin,
)
from vllm_omni.diffusion.distributed.parallel_state import get_dit_group

from .packed_tokens import minimax_h3_patchify_video_latent

MINIMAX_H3_KEYFRAME_ENCODE_SEED = 42
MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000
MINIMAX_H3_AUDIO_CHANNELS = 2


def _load_component_config(component_path: str) -> dict[str, Any]:
    config_path = Path(component_path) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    channels = int(config["latent_channels"])
    for key in ("latents_mean", "latents_std"):
        values = config.get(key)
        if not isinstance(values, list) or len(values) != channels:
            raise ValueError(f"{config_path}: {key} must contain {channels} values")
    return config


def _load_remote_component(
    component_path: str,
    config: dict[str, Any],
) -> nn.Module:
    auto_map = config.get("auto_map") or {}
    class_reference = auto_map.get("AutoModel")
    if not isinstance(class_reference, str):
        raise ValueError(f"{component_path}/config.json must define auto_map.AutoModel")
    component_cls = get_class_from_dynamic_module(
        class_reference,
        component_path,
    )
    return component_cls.from_pretrained(component_path)


def _install_fast_tiled_collective(model: nn.Module) -> None:
    """Use a fixed-shape gather when native VAE tiles have one shape.

    H3's configured 256-pixel tiles are equal-sized after alignment.  The
    bundled remote implementation uses a variable-shape gather for every
    encode/decode, which first all-gathers shape tensors and pads each rank.
    Keep that path as a fallback for unusual inputs, but avoid the metadata
    collective for the normal recipe.
    """
    original = getattr(model, "_all_gather_tiled_results", None)
    if original is None:
        return

    package = model.__class__.__module__.rsplit(".", 1)[0]
    parallel_module = importlib.import_module(f"{package}.parallel")

    def fast_all_gather_tiled_results(tasks, num_tiles):
        if not tasks:
            return original(tasks, num_tiles)
        shapes = {tuple(task.shape) for task in tasks}
        if len(shapes) != 1:
            return original(tasks, num_tiles)

        state = parallel_module.get_parallel_state()
        group = state["sp_process_group"]
        sp_size = int(state["sp_size"])
        sp_rank = int(state["sp_rank"])
        stacked = torch.stack(tasks, dim=0).contiguous()
        max_tasks = (int(num_tiles) + sp_size - 1) // sp_size
        if stacked.shape[0] < max_tasks:
            padding = torch.zeros(
                (max_tasks - stacked.shape[0], *stacked.shape[1:]),
                dtype=stacked.dtype,
                device=stacked.device,
            )
            stacked = torch.cat((stacked, padding), dim=0)
        gathered = [torch.empty_like(stacked) for _ in range(sp_size)]
        dist.all_gather(gathered, stacked, group=group)

        results = [None] * int(num_tiles)
        for rank, rank_tensors in enumerate(gathered):
            count = max(0, min(max_tasks, (int(num_tiles) - rank + sp_size - 1) // sp_size))
            for index in range(count):
                results[index * sp_size + rank] = rank_tensors[index]
        return results

    model._all_gather_tiled_results = fast_all_gather_tiled_results


class _AudioVAEDeterminismContext(AbstractContextManager):
    def __enter__(self):
        backends = torch.backends
        self._saved = (
            backends.cuda.matmul.allow_tf32,
            backends.cudnn.allow_tf32,
            backends.cudnn.benchmark,
            backends.cudnn.deterministic,
            backends.cudnn.enabled,
            backends.cuda.flash_sdp_enabled(),
            backends.cuda.mem_efficient_sdp_enabled(),
            backends.cuda.math_sdp_enabled(),
        )
        backends.cuda.matmul.allow_tf32 = False
        backends.cudnn.allow_tf32 = False
        backends.cudnn.benchmark = False
        backends.cudnn.deterministic = True
        backends.cudnn.enabled = False
        backends.cuda.enable_flash_sdp(False)
        backends.cuda.enable_mem_efficient_sdp(False)
        backends.cuda.enable_math_sdp(True)
        return self

    def __exit__(self, exc_type, exc, traceback):
        backends = torch.backends
        (
            backends.cuda.matmul.allow_tf32,
            backends.cudnn.allow_tf32,
            backends.cudnn.benchmark,
            backends.cudnn.deterministic,
            backends.cudnn.enabled,
            flash,
            memory_efficient,
            math_sdp,
        ) = self._saved
        backends.cuda.enable_flash_sdp(flash)
        backends.cuda.enable_mem_efficient_sdp(memory_efficient)
        backends.cuda.enable_math_sdp(math_sdp)
        return False


class MiniMaxH3VideoVAE(nn.Module, DistributedVaeMixin):
    """Adapter around the checkpoint's native parallel-tiled video VAE."""

    def __init__(
        self,
        component_path: str,
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.config_dict = _load_component_config(component_path)
        self.remote = _load_remote_component(
            component_path,
            self.config_dict,
        )
        # Match the reference loader contract: video VAE weights stay FP32.
        # Keyframe encoding is numerically sensitive to first casting the
        # checkpoint through FP16; decode still runs under FP16 autocast.
        self.remote.eval().to(device=device, dtype=torch.float32)
        self.model = self.remote.model
        _install_fast_tiled_collective(self.model)
        # Batch the independent local tiles so the decoder can keep its GEMMs
        # saturated instead of launching one kernel sequence per tile.
        self.model.stack_tiling = True
        self.use_tiling = True
        self.use_slicing = False
        self.parallel_size = 1

    def set_parallel_size(
        self,
        parallel_size: int,
        mode: str = "tile",
    ) -> None:
        if mode != "tile":
            raise ValueError(f"MiniMax H3 VAE supports its native tile parallel mode only, got {mode!r}")
        group = get_dit_group()
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        parallel_size = int(parallel_size)
        if parallel_size not in (1, world_size):
            raise ValueError(
                "MiniMax H3 native VAE patch parallelism currently requires "
                "vae_patch_parallel_size=1 or the full DiT group size "
                f"({world_size}), got {parallel_size}"
            )
        self.parallel_size = parallel_size
        enabled = parallel_size > 1

        package = self.remote.__class__.__module__.rsplit(".", 1)[0]
        parallel_module = importlib.import_module(f"{package}.parallel")
        state = parallel_module.get_parallel_state()
        state.clear()
        state.update(
            group_size=parallel_size,
            group_rank=rank if enabled else 0,
            local_process_group=group if enabled else None,
            sp_size=parallel_size,
            sp_rank=rank if enabled else 0,
            sp_enabled=enabled,
            sp_process_group=group if enabled else None,
            tp_size=1,
            tp_rank=0,
        )
        self.model.parallel_tiling = enabled

    def is_distributed_enabled(self) -> bool:
        return self.parallel_size > 1 and dist.is_initialized()

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        previous_parallel = self.model.parallel_tiling
        self.model.parallel_tiling = False
        parameter = next(self.parameters())
        previous_dtype = parameter.dtype
        if previous_dtype != torch.float32:
            self.to(torch.float32)
        devices = [parameter.device] if parameter.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                torch.default_generator.manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
                for device in devices:
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
                latent = self.model.encode_images(
                    image,
                    use_fp16_latent=True,
                )[0]
        finally:
            self.model.parallel_tiling = previous_parallel
            if previous_dtype != torch.float32:
                self.to(previous_dtype)

        # Match the reference contract exactly: normalization and patchify
        # happen on CPU in FP32 after the sampled encode. The condition noise
        # path is sensitive enough that doing these elementwise operations on
        # CUDA can noticeably change the final conditioned video.
        latent = latent.float().cpu()
        if latent.ndim == 4:
            latent = latent[None]
        channels = int(self.config_dict["latent_channels"])
        mean = torch.tensor(
            self.config_dict["latents_mean"],
        ).view(1, channels, 1, 1, 1)
        std = torch.tensor(
            self.config_dict["latents_std"],
        ).view(1, channels, 1, 1, 1)
        return minimax_h3_patchify_video_latent(
            (latent - mean) / std,
            patch_size=(1, 2, 2),
        ).float()

    @torch.inference_mode()
    def encode_video(
        self,
        frames: Any,
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        parameter = next(self.parameters())
        previous_dtype = parameter.dtype
        if previous_dtype != torch.float32:
            self.to(torch.float32)
        devices = [parameter.device] if parameter.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                torch.default_generator.manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
                for device in devices:
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
                latent = self.model.encode_videos(
                    frames,
                    use_fp16_latent=True,
                )[0]
        finally:
            if previous_dtype != torch.float32:
                self.to(previous_dtype)

        latent = latent.float().cpu()
        if latent.ndim == 4:
            latent = latent[None]
        channels = int(self.config_dict["latent_channels"])
        if latent.ndim != 5 or int(latent.shape[1]) != channels:
            raise ValueError(f"unexpected reference video latent shape {tuple(latent.shape)}")
        shape = (
            int(latent.shape[2]),
            int(latent.shape[3]),
            int(latent.shape[4]),
        )
        mean = torch.tensor(
            self.config_dict["latents_mean"],
        ).view(1, channels, 1, 1, 1)
        std = torch.tensor(
            self.config_dict["latents_std"],
        ).view(1, channels, 1, 1, 1)
        rows = minimax_h3_patchify_video_latent(
            (latent - mean) / std,
            patch_size=(1, 2, 2),
        ).float()
        return rows, shape

    @torch.inference_mode()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        channels = int(self.config_dict["latent_channels"])
        mean = torch.tensor(
            self.config_dict["latents_mean"],
            device=latent.device,
            dtype=latent.dtype,
        ).view(1, channels, 1, 1, 1)
        std = torch.tensor(
            self.config_dict["latents_std"],
            device=latent.device,
            dtype=latent.dtype,
        ).view(1, channels, 1, 1, 1)
        decoded = self.model.decode_base(latent * std + mean)
        frames = self.model.processor.revert_tensor(decoded)
        if frames.ndim == 4:
            frames = frames.unsqueeze(0).transpose(1, 2)
        if frames.ndim != 5:
            raise ValueError(f"unexpected decoded video shape {tuple(frames.shape)}")
        return frames.float()


class MiniMaxH3AudioVAE(nn.Module):
    def __init__(
        self,
        component_path: str,
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.config_dict = _load_component_config(component_path)
        self.remote = _load_remote_component(
            component_path,
            self.config_dict,
        )
        # The checkpoint's audio VAE contract is FP32 for both reference
        # encoding and waveform decoding.
        self.remote.eval().to(device=device, dtype=torch.float32)
        self.model = self.remote.model
        self.sample_rate = int(self.config_dict["sample_rate"])
        self._latent_stats_cpu = (
            torch.tensor(self.config_dict["latents_mean"], dtype=torch.float32).view(1, 1, -1),
            torch.tensor(self.config_dict["latents_std"], dtype=torch.float32).view(1, 1, -1),
        )
        self._latent_stats_device_cache: dict[
            tuple[str, int | None, torch.dtype], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._resamplers: dict[int, Any] = {}

    def _latent_stats(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device.type, device.index, dtype)
        stats = self._latent_stats_device_cache.get(key)
        if stats is None:
            stats = tuple(value.to(device=device, dtype=dtype) for value in self._latent_stats_cpu)
            self._latent_stats_device_cache[key] = stats
        return stats

    @torch.inference_mode()
    def encode_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> tuple[torch.Tensor, int]:
        import torchaudio

        waveform = waveform.float()
        if waveform.ndim == 1:
            waveform = waveform[None]
        if int(sample_rate) != MINIMAX_H3_AUDIO_SAMPLE_RATE:
            input_rate = int(sample_rate)
            resampler = self._resamplers.get(input_rate)
            if resampler is None:
                resampler = torchaudio.transforms.Resample(
                    input_rate,
                    MINIMAX_H3_AUDIO_SAMPLE_RATE,
                )
                self._resamplers[input_rate] = resampler
            waveform = resampler(waveform)
        if waveform.shape[0] < MINIMAX_H3_AUDIO_CHANNELS:
            waveform = waveform.repeat(
                MINIMAX_H3_AUDIO_CHANNELS,
                1,
            )
        waveform = waveform[:MINIMAX_H3_AUDIO_CHANNELS]
        device = next(self.model.parameters()).device
        waveform = waveform.to(device)

        with _AudioVAEDeterminismContext():
            audio = self.model.preprocess(
                waveform.unsqueeze(1),
                MINIMAX_H3_AUDIO_SAMPLE_RATE,
            )
            latent = self.model.encoder(audio)
            if bool(getattr(self.model, "attn_proj", False)):
                latent = self.model.pre_block(latent.transpose(1, 2)).transpose(1, 2)
            latent = self.model.mean_proj(latent).float().cpu()

        channels = int(self.config_dict["latent_channels"])
        if latent.shape[-1] != channels:
            if latent.shape[1] != channels:
                raise ValueError(f"cannot canonicalize audio latent {tuple(latent.shape)}")
            latent = latent.transpose(1, 2).contiguous()
        mean, std = self._latent_stats(device=latent.device, dtype=latent.dtype)
        rows = ((latent - mean) / std).reshape(-1, channels)
        return rows.float(), int(latent.shape[1])

    @torch.inference_mode()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        mean, std = self._latent_stats(device=latent.device, dtype=latent.dtype)
        mean = mean.transpose(1, 2)
        std = std.transpose(1, 2)
        waveform = self.remote.decode(latent * std + mean)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError(f"unexpected decoded audio shape {tuple(waveform.shape)}")
        return waveform.permute(1, 0, 2).contiguous().float()


__all__ = ["MiniMaxH3AudioVAE", "MiniMaxH3VideoVAE"]
