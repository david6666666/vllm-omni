# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


class OffloadStrategy(Enum):
    NONE = "none"
    MODEL_LEVEL = "model_level"  # Sequential offloading between DiT and encoders
    LAYER_WISE = "layer_wise"  # Block-level
    DISTRIBUTED_LAYER_WISE = "distributed_layer_wise"  # Sharded block-level offload


@dataclass
class OffloadConfig:
    strategy: OffloadStrategy
    pin_cpu_memory: bool = True
    use_hsdp: bool = False
    distributed_layerwise_offload_prefetch: bool = True
    data_parallel_size: int = 1
    sequence_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    cfg_parallel_size: int = 1
    cache_backend: str | None = "none"

    @classmethod
    def from_od_config(cls, od_config: OmniDiffusionConfig) -> "OffloadConfig":
        """Extract and validate offload settings from OmniDiffusionConfig.

        Selects one model-level, layer-wise, or distributed layer-wise strategy.
        Distributed layer-wise offload has the highest priority.

        Args:
            od_config: OmniDiffusionConfig with offload settings

        Returns:
            OffloadConfig with validated settings
        """
        enable_cpu_offload = getattr(od_config, "enable_cpu_offload", False)
        enable_layerwise_offload = getattr(od_config, "enable_layerwise_offload", False)
        enable_distributed_layerwise_offload = getattr(
            od_config,
            "enable_distributed_layerwise_offload",
            False,
        )
        pin_cpu_memory = getattr(od_config, "pin_cpu_memory", True)

        parallel_config = getattr(od_config, "parallel_config", None)
        use_hsdp = getattr(parallel_config, "use_hsdp", False) if parallel_config else False
        data_parallel_size = getattr(parallel_config, "data_parallel_size", 1) if parallel_config else 1
        sequence_parallel_size = getattr(parallel_config, "sequence_parallel_size", 1) if parallel_config else 1
        tensor_parallel_size = getattr(parallel_config, "tensor_parallel_size", 1) if parallel_config else 1
        pipeline_parallel_size = getattr(parallel_config, "pipeline_parallel_size", 1) if parallel_config else 1
        cfg_parallel_size = getattr(parallel_config, "cfg_parallel_size", 1) if parallel_config else 1

        # Determine strategy. Distributed layer-wise offload is explicit and
        # takes priority over the legacy single-rank strategies.
        if enable_distributed_layerwise_offload:
            strategy = OffloadStrategy.DISTRIBUTED_LAYER_WISE
            if enable_layerwise_offload or enable_cpu_offload:
                logger.info("Distributed layer-wise offloading takes priority, disabling other offloading strategies.")
        elif enable_layerwise_offload:
            strategy = OffloadStrategy.LAYER_WISE
            if enable_cpu_offload:
                logger.info(
                    "Both model-level and layer-wise offloading enabled. "
                    "Layer-wise takes priority, disabling model-level offloading."
                )
        elif enable_cpu_offload:
            strategy = OffloadStrategy.MODEL_LEVEL
        else:
            strategy = OffloadStrategy.NONE

        return cls(
            strategy=strategy,
            pin_cpu_memory=pin_cpu_memory,
            use_hsdp=use_hsdp,
            distributed_layerwise_offload_prefetch=getattr(
                od_config,
                "distributed_layerwise_offload_prefetch",
                True,
            ),
            data_parallel_size=data_parallel_size,
            sequence_parallel_size=sequence_parallel_size or 1,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            cfg_parallel_size=cfg_parallel_size,
            cache_backend=getattr(od_config, "cache_backend", "none"),
        )


class OffloadBackend(ABC):
    """Base class for CPU offload backends"""

    def __init__(self, config: OffloadConfig, device: torch.device):
        self.config = config
        self.device = device
        self.enabled = False

    @abstractmethod
    def enable(self, pipeline: nn.Module) -> None:
        """Enable offloading on the pipeline.

        Discovers modules, moves them to appropriate devices, and
        registers forward hooks for swapping/prefetching.

        Args:
            pipeline: Diffusion pipeline model (e.g., Wan22Pipeline)
        """
        raise NotImplementedError

    @abstractmethod
    def disable(self) -> None:
        """Disable offloading and cleanup resources.

        Removes all registered hooks. Does NOT move modules back to
        original devices (caller responsible for that).
        """
        raise NotImplementedError

    def is_enabled(self) -> bool:
        return self.enabled
