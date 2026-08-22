# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compatibility SVDQuant NVFP4 backend for datacenter Blackwell.

This backend deliberately uses vLLM's existing NVFP4 linear kernel. The
low-rank correction and optional per-output scale are ordinary PyTorch
operations, so basic checkpoint support does not depend on a new FlashInfer
operator.
"""

from __future__ import annotations

import functools

import torch
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability

logger = init_logger(__name__)

_SUPPORTED_CAPABILITIES = {(10, 3)}


def supports(capability: DeviceCapability | None) -> bool:
    return capability is not None and (capability.major, capability.minor) in _SUPPORTED_CAPABILITIES


def assert_supported() -> None:
    if not current_platform.is_cuda():
        raise RuntimeError("SVDQuant NVFP4 requires a CUDA device")
    capability = current_platform.get_device_capability()
    if not supports(capability):
        device = current_platform.device_name
        sm = capability.to_int() if capability is not None else "unknown"
        raise RuntimeError(f"Phase 1 SVDQuant NVFP4 is validated on SM103 only; got {device!r} (SM{sm})")


@functools.cache
def _nvfp4_kernel():
    return init_nvfp4_linear_kernel()


def prepare_weights(layer: torch.nn.Module) -> None:
    """Adapt the canonical row-major checkpoint to vLLM's NVFP4 kernel."""
    qweight = layer.qweight
    wscales = layer.wscales
    del layer.qweight
    del layer.wscales
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(
            qweight.detach().view(torch.uint8),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(
            wscales.detach().transpose(0, 1).contiguous(),
            requires_grad=False,
        ),
    )

    layer.register_parameter(
        "input_global_scale_inv",
        torch.nn.Parameter(
            torch.ones(
                1,
                dtype=torch.float32,
                device=layer.weight.device,
            ),
            requires_grad=False,
        ),
    )

    wtscale = layer.wtscale.detach().to(dtype=torch.float32)
    del layer.wtscale
    layer.register_parameter(
        "alpha",
        torch.nn.Parameter(wtscale, requires_grad=False),
    )

    channel_scale = layer.wcscales.detach()
    del layer.wcscales
    if torch.all(channel_scale == 1).item():
        layer.output_channel_scale = None
    else:
        layer.register_parameter(
            "output_channel_scale",
            torch.nn.Parameter(channel_scale, requires_grad=False),
        )

    _nvfp4_kernel().process_weights_after_loading(layer)
    logger.info_once(
        "SVDQuant NVFP4 is using vLLM's compatibility path; the rank correction is not fused into the GEMM."
    )


def apply(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute base NVFP4 GEMM plus the BF16 rank correction."""
    if x.dtype != torch.bfloat16:
        raise ValueError(f"SVDQuant NVFP4 requires BF16 activations; got {x.dtype}")

    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1]).contiguous()

    # The exported residual branch consumes the original activation, while
    # only the four-bit base GEMM consumes the smoothed activation.
    smoothed = x_2d / layer.smooth_factor
    out = _nvfp4_kernel().apply_weights(
        layer=layer,
        x=smoothed,
        bias=None,
    )

    channel_scale = getattr(layer, "output_channel_scale", None)
    if channel_scale is not None:
        # H3's fused QKV stores independent Q/K/V outer scales. Apply them
        # explicitly in Phase 1 rather than requiring a vector-alpha epilogue.
        out.mul_(channel_scale)

    correction_input = torch.mm(x_2d, layer.proj_down)
    out = torch.addmm(
        out,
        correction_input,
        layer.proj_up.transpose(0, 1),
    )
    if bias is not None:
        out.add_(bias)
    return out.reshape(
        *original_shape[:-1],
        layer.out_features_per_partition,
    )


__all__ = ["apply", "assert_supported", "prepare_weights", "supports"]
