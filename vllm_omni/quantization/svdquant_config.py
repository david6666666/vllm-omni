# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serialized SVDQuant NVFP4 support for diffusion transformers.

The checkpoint stores NVFP4 weights plus a rank-R correction for each
quantized linear. Phase 1 intentionally uses vLLM's existing NVFP4 linear
kernel and ordinary BF16 matrix multiplication for the correction. Native
SVDQuant fusion is a separate optimization and is not required to load or run
the checkpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Parameter
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods


class DiffusionSVDQuantConfig(QuantizationConfig):
    """Configuration for serialized NVFP4 W4A4 plus low-rank correction."""

    def __init__(
        self,
        rank: int = 32,
        precision: str = "nvfp4",
        act_unsigned: bool = False,
        modules_to_not_convert: list[str] | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"SVDQuant rank must be positive, got {rank}")
        if precision != "nvfp4":
            raise ValueError(
                f"Phase 1 SVDQuant supports serialized NVFP4 checkpoints only; got precision={precision!r}"
            )
        if act_unsigned:
            raise ValueError("Phase 1 SVDQuant does not support unsigned activations")
        self.rank = rank
        self.precision = precision
        self.act_unsigned = act_unsigned
        self.modules_to_not_convert = modules_to_not_convert or []

    def __repr__(self) -> str:
        return f"DiffusionSVDQuantConfig(rank={self.rank}, precision={self.precision!r})"

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "svdquant"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 103

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiffusionSVDQuantConfig:
        return cls(
            rank=config.get("rank", 32),
            precision=config.get("precision", "nvfp4"),
            act_unsigned=config.get("act_unsigned", False),
            modules_to_not_convert=config.get("modules_to_not_convert"),
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if is_layer_skipped(
            prefix,
            self.modules_to_not_convert,
            self.packed_modules_mapping,
            skip_with_substr=True,
        ):
            return UnquantizedLinearMethod()
        return DiffusionSVDQuantLinearMethod(self)


class DiffusionSVDQuantLinearMethod(LinearMethodBase):
    """Load the canonical checkpoint layout and call the Phase 1 backend."""

    def __init__(self, quant_config: DiffusionSVDQuantConfig) -> None:
        from . import svdquant_flashinfer

        svdquant_flashinfer.assert_supported()
        self.quant_config = quant_config
        self._backend = svdquant_flashinfer

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        del params_dtype
        weight_loader = extra_weight_attrs.pop(
            "weight_loader",
            default_weight_loader,
        )
        output_size_per_partition = sum(output_partition_sizes)
        rank = self.quant_config.rank

        qweight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        _set_attrs(
            qweight,
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        wscales = Parameter(
            torch.empty(
                input_size_per_partition // 16,
                output_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        _set_attrs(
            wscales,
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        proj_down = Parameter(
            torch.empty(
                input_size_per_partition,
                rank,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        _set_attrs(
            proj_down,
            input_dim=0,
            weight_loader=weight_loader,
        )

        proj_up = Parameter(
            torch.empty(
                output_size_per_partition,
                rank,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        _set_attrs(
            proj_up,
            output_dim=0,
            weight_loader=weight_loader,
        )

        smooth_factor = Parameter(
            torch.empty(
                input_size_per_partition,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        _set_attrs(
            smooth_factor,
            input_dim=0,
            weight_loader=weight_loader,
        )

        wcscales = Parameter(
            torch.ones(
                output_size_per_partition,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        _set_attrs(
            wcscales,
            output_dim=0,
            weight_loader=weight_loader,
        )

        wtscale = Parameter(
            torch.ones(1, dtype=torch.bfloat16),
            requires_grad=False,
        )
        _set_attrs(wtscale, weight_loader=default_weight_loader)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("wscales", wscales)
        layer.register_parameter("proj_down", proj_down)
        layer.register_parameter("proj_up", proj_up)
        layer.register_parameter("smooth_factor", smooth_factor)
        layer.register_parameter("wcscales", wcscales)
        layer.register_parameter("wtscale", wtscale)

        layer.in_features = input_size
        layer.out_features = output_size
        layer.output_size_per_partition = output_size_per_partition
        layer.out_features_per_partition = output_size_per_partition

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._backend.prepare_weights(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._backend.apply(layer, x, bias)


def _set_attrs(param: torch.nn.Parameter, **attrs: Any) -> None:
    for key, value in attrs.items():
        setattr(param, key, value)


__all__ = [
    "DiffusionSVDQuantConfig",
    "DiffusionSVDQuantLinearMethod",
]
