# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed layer-wise CPU offload with fixed device buffers.

Every rank retains only its contiguous shard of each offloaded block in host
memory.  A single controller shared by every DiT ring reconstructs blocks into
two persistent device slots.  H2D for the next block and its AllGather execute
on dedicated streams while the current block computes.

This backend intentionally supports only one weight-shard dimension at a time:
pure data parallelism or pure sequence parallelism.  HSDP/DTensor weights and
hybrid parallel meshes require placement-aware checkpoint loading and are
rejected rather than silently sharding an already-sharded tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from torch.profiler import record_function
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
from vllm_omni.diffusion.distributed.parallel_state import get_dp_group, get_sp_group
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.platforms import current_omni_platform

from .base import OffloadBackend, OffloadConfig
from .layerwise_backend import LayerWiseOffloadBackend
from .module_collector import ModuleDiscovery

logger = init_logger(__name__)


@dataclass
class _TensorBinding:
    target: torch.Tensor
    shape: torch.Size
    offset: int
    numel: int
    placeholder: torch.Tensor
    original: torch.Tensor | None


@dataclass
class _BlockState:
    block_id: int
    module: nn.Module
    bindings: dict[torch.dtype, list[_TensorBinding]]
    cpu_shards: dict[torch.dtype, torch.Tensor]
    shard_numel: dict[torch.dtype, int]
    next_block_id: int = -1
    slot: int | None = None


@dataclass
class _SlotState:
    full: dict[torch.dtype, torch.Tensor]
    staging: dict[torch.dtype, torch.Tensor]
    copy_done_event: Any
    ready_event: Any
    compute_done_event: Any
    owner: int | None = None
    generation: int = -1
    active: bool = False
    has_ready: bool = False
    has_compute_done: bool = False


class DistributedLayerwiseWeightController:
    """Own all CPU shards, the two device slots, and their stream lifecycle."""

    def __init__(
        self,
        rings: list[list[nn.Module]],
        *,
        device: torch.device,
        group: GroupCoordinator,
        pin_memory: bool,
        prefetch: bool,
        copy_stream: Any | None = None,
        comm_stream: Any | None = None,
    ) -> None:
        if not rings or any(not ring for ring in rings):
            raise ValueError("Distributed layerwise offload requires non-empty block rings.")
        if group.world_size < 2:
            raise ValueError("Distributed layerwise offload requires a process group with at least two ranks.")
        if not 0 <= group.rank_in_group < group.world_size:
            raise ValueError(f"Invalid group-local rank {group.rank_in_group} for world size {group.world_size}.")

        self.device = device
        self.group = group
        self.world_size = group.world_size
        self.rank_in_group = group.rank_in_group
        self.pin_memory = pin_memory
        self.prefetch_enabled = prefetch
        self.copy_stream = copy_stream or current_omni_platform.Stream()
        self.comm_stream = comm_stream or current_omni_platform.Stream()

        self.blocks: list[_BlockState] = []
        self.rings: list[list[int]] = []
        self._module_to_block_id: dict[int, int] = {}
        self._installed = False
        self._generation = 0
        self._active_block_id: int | None = None

        seen_modules: set[int] = set()
        seen_tensors: set[int] = set()
        for ring in rings:
            ring_ids: list[int] = []
            for module in ring:
                if id(module) in seen_modules:
                    raise ValueError("An offloaded block cannot appear in more than one ring.")
                seen_modules.add(id(module))
                block_id = len(self.blocks)
                state = self._capture_block(block_id, module, seen_tensors)
                self.blocks.append(state)
                self._module_to_block_id[id(module)] = block_id
                ring_ids.append(block_id)
            if len(ring_ids) < 2:
                raise ValueError("Each distributed layerwise offload ring must contain at least two blocks.")
            for index, block_id in enumerate(ring_ids):
                self.blocks[block_id].next_block_id = ring_ids[(index + 1) % len(ring_ids)]
            self.rings.append(ring_ids)

        max_full_numel: dict[torch.dtype, int] = {}
        max_shard_numel: dict[torch.dtype, int] = {}
        for block in self.blocks:
            for dtype, shard_numel in block.shard_numel.items():
                max_shard_numel[dtype] = max(max_shard_numel.get(dtype, 0), shard_numel)
                padded_full_numel = shard_numel * self.world_size
                max_full_numel[dtype] = max(max_full_numel.get(dtype, 0), padded_full_numel)

        # These are the only weight-bearing device allocations owned by the
        # controller: two full slots and two per-rank shard staging slots.
        full_slots = [
            {dtype: torch.empty(numel, dtype=dtype, device=self.device) for dtype, numel in max_full_numel.items()}
            for _ in range(2)
        ]
        staging_slots = [
            {dtype: torch.empty(numel, dtype=dtype, device=self.device) for dtype, numel in max_shard_numel.items()}
            for _ in range(2)
        ]
        self.slots = [
            _SlotState(
                full=full_slots[slot],
                staging=staging_slots[slot],
                copy_done_event=current_omni_platform.Event(),
                ready_event=current_omni_platform.Event(),
                compute_done_event=current_omni_platform.Event(),
            )
            for slot in range(2)
        ]

        logger.info(
            "Allocated distributed layerwise buffers for %d blocks across %d rings: "
            "2 full slots + 2 shard staging slots, group_size=%d, rank_in_group=%d",
            len(self.blocks),
            len(self.rings),
            self.world_size,
            self.rank_in_group,
        )

    def _capture_block(
        self,
        block_id: int,
        module: nn.Module,
        seen_tensors: set[int],
    ) -> _BlockState:
        grouped: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
        for name, tensor in chain(module.named_parameters(), module.named_buffers()):
            self._validate_tensor(name, tensor)
            if id(tensor) in seen_tensors:
                raise ValueError(f"Shared tensor {name!r} appears in multiple offloaded blocks.")
            seen_tensors.add(id(tensor))
            grouped.setdefault(tensor.dtype, []).append((name, tensor))

        if not grouped or not any(tensor.numel() for tensors in grouped.values() for _, tensor in tensors):
            raise ValueError(f"Offloaded block {module.__class__.__name__} has no tensor storage.")

        bindings: dict[torch.dtype, list[_TensorBinding]] = {}
        cpu_shards: dict[torch.dtype, torch.Tensor] = {}
        shard_numel: dict[torch.dtype, int] = {}

        for dtype, tensors in grouped.items():
            total_numel = sum(tensor.numel() for _, tensor in tensors)
            local_shard_numel = (total_numel + self.world_size - 1) // self.world_size
            shard_start = self.rank_in_group * local_shard_numel
            shard_end = min(shard_start + local_shard_numel, total_numel)
            shard = torch.zeros(
                local_shard_numel,
                dtype=dtype,
                device="cpu",
                pin_memory=self.pin_memory,
            )

            offset = 0
            dtype_bindings: list[_TensorBinding] = []
            for name, tensor in tensors:
                numel = tensor.numel()
                overlap_start = max(offset, shard_start)
                overlap_end = min(offset + numel, shard_end)
                if overlap_start < overlap_end:
                    source_start = overlap_start - offset
                    destination_start = overlap_start - shard_start
                    copy_numel = overlap_end - overlap_start
                    shard.narrow(0, destination_start, copy_numel).copy_(
                        tensor.detach().reshape(-1).narrow(0, source_start, copy_numel),
                        non_blocking=False,
                    )

                dtype_bindings.append(
                    _TensorBinding(
                        target=tensor,
                        shape=tensor.shape,
                        offset=offset,
                        numel=numel,
                        placeholder=torch.empty(0, dtype=dtype, device=self.device),
                        original=tensor.detach(),
                    )
                )
                offset += numel

            bindings[dtype] = dtype_bindings
            cpu_shards[dtype] = shard
            shard_numel[dtype] = local_shard_numel

        return _BlockState(
            block_id=block_id,
            module=module,
            bindings=bindings,
            cpu_shards=cpu_shards,
            shard_numel=shard_numel,
        )

    @staticmethod
    def _validate_tensor(name: str, tensor: torch.Tensor) -> None:
        if isinstance(tensor, DTensor):
            raise ValueError(f"Distributed layerwise offload only supports ordinary tensors; {name!r} is a DTensor.")
        # vLLM linear layers store ordinary dense weights in subclasses such
        # as ModelWeightParameter. Preserve that Parameter object (and its
        # loader metadata) while swapping only ``.data`` between placeholders
        # and full-slot views. Non-Parameter Tensor subclasses remain unsafe.
        is_parameter_subclass = isinstance(tensor, nn.Parameter) and issubclass(type(tensor), nn.Parameter)
        if type(tensor) is not torch.Tensor and not is_parameter_subclass:
            raise ValueError(
                "Distributed layerwise offload does not support tensor subclasses; "
                f"{name!r} has type {type(tensor).__name__}."
            )
        if tensor.layout is not torch.strided or tensor.is_quantized:
            raise ValueError(f"Distributed layerwise offload requires a dense, non-quantized tensor for {name!r}.")
        if not tensor.is_contiguous():
            raise ValueError(f"Distributed layerwise offload requires a contiguous tensor for {name!r}.")
        if tensor.is_meta:
            raise ValueError(f"Cannot shard meta tensor {name!r}.")

    def block_id_for(self, module: nn.Module) -> int:
        return self._module_to_block_id[id(module)]

    @property
    def full_slots(self) -> list[dict[torch.dtype, torch.Tensor]]:
        return [slot.full for slot in self.slots]

    @property
    def staging_slots(self) -> list[dict[torch.dtype, torch.Tensor]]:
        return [slot.staging for slot in self.slots]

    def install_placeholders(self) -> None:
        if self._installed:
            return
        for block in self.blocks:
            self._point_to_placeholder(block)
        self._installed = True

    def commit_originals(self) -> None:
        """Release full pre-shard tensors after enable has completed safely."""
        for block in self.blocks:
            for dtype_bindings in block.bindings.values():
                for binding in dtype_bindings:
                    binding.original = None

    def rollback_originals(self) -> None:
        """Restore original tensors if enable fails before it is committed."""
        if not self._installed:
            return
        current_omni_platform.synchronize()
        for block in self.blocks:
            for dtype_bindings in block.bindings.values():
                for binding in dtype_bindings:
                    if binding.original is not None:
                        binding.target.data = binding.original
            block.slot = None
        self._installed = False

    def quiesce(self) -> None:
        current_omni_platform.synchronize()
        for block in self.blocks:
            self._point_to_placeholder(block)
            block.slot = None
        for slot in self.slots:
            slot.owner = None
            slot.active = False
        self._active_block_id = None

    def _point_to_placeholder(self, block: _BlockState) -> None:
        for dtype_bindings in block.bindings.values():
            for binding in dtype_bindings:
                binding.target.data = binding.placeholder

    def _point_to_slot(self, block: _BlockState, slot_id: int) -> None:
        full = self.slots[slot_id].full
        for dtype, dtype_bindings in block.bindings.items():
            flat = full[dtype]
            for binding in dtype_bindings:
                binding.target.data = flat.narrow(0, binding.offset, binding.numel).view(binding.shape)

    def _select_slot(self, exclude: int | None) -> int:
        candidates = [slot_id for slot_id, slot in enumerate(self.slots) if slot_id != exclude and not slot.active]
        if not candidates:
            raise RuntimeError("No distributed layerwise buffer slot is safe to reuse.")
        empty = [slot_id for slot_id in candidates if self.slots[slot_id].owner is None]
        if empty:
            return empty[0]
        return min(candidates, key=lambda slot_id: self.slots[slot_id].generation)

    @torch.compiler.disable
    def _load_block(self, block_id: int, *, exclude: int | None = None) -> int:
        block = self.blocks[block_id]
        if block.slot is not None and self.slots[block.slot].owner == block_id:
            return block.slot

        slot_id = self._select_slot(exclude)
        slot = self.slots[slot_id]
        old_owner = slot.owner
        if old_owner is not None:
            old_block = self.blocks[old_owner]
            self._point_to_placeholder(old_block)
            old_block.slot = None

        # The staging slot may be reused once the previous AllGather has read
        # it.  The full slot may be overwritten only after its prior compute.
        if slot.has_ready:
            self.copy_stream.wait_event(slot.ready_event)
        with current_omni_platform.stream(self.copy_stream):
            with record_function(f"distributed_layerwise_offload.h2d.block_{block_id}"):
                for dtype, cpu_shard in block.cpu_shards.items():
                    destination = slot.staging[dtype].narrow(0, 0, block.shard_numel[dtype])
                    destination.copy_(
                        cpu_shard,
                        non_blocking=self.pin_memory and self.device.type != "cpu",
                    )
                slot.copy_done_event.record(self.copy_stream)

        self.comm_stream.wait_event(slot.copy_done_event)
        if slot.has_compute_done:
            self.comm_stream.wait_event(slot.compute_done_event)
        with current_omni_platform.stream(self.comm_stream):
            with record_function(f"distributed_layerwise_offload.all_gather.block_{block_id}"):
                for dtype, local_numel in block.shard_numel.items():
                    source = slot.staging[dtype].narrow(0, 0, local_numel)
                    destination = slot.full[dtype].narrow(0, 0, local_numel * self.world_size)
                    dist.all_gather_into_tensor(
                        destination,
                        source,
                        group=self.group.device_group,
                        async_op=False,
                    )
                slot.ready_event.record(self.comm_stream)

        self._generation += 1
        slot.owner = block_id
        slot.generation = self._generation
        slot.has_ready = True
        slot.has_compute_done = False
        block.slot = slot_id
        return slot_id

    @torch.compiler.disable
    def before_forward(self, block_id: int) -> int:
        if self._active_block_id is not None:
            raise RuntimeError(
                "Nested execution of distributed-offloaded blocks is unsupported; "
                f"block {self._active_block_id} is still active."
            )

        slot_id = self._load_block(block_id)
        slot = self.slots[slot_id]
        with record_function(f"distributed_layerwise_offload.wait.block_{block_id}"):
            current_omni_platform.current_stream().wait_event(slot.ready_event)
            if not self.prefetch_enabled:
                # Waiting on the compute stream alone is asynchronous from the
                # host.  In eager execution Python can otherwise submit the
                # following block's preparation while this block still runs,
                # accidentally retaining overlap in the advertised synchronous
                # ablation.  This host-visible ready wait, paired with the
                # compute-done wait in ``after_forward``, makes prefetch=False
                # a true prepare-then-compute baseline without synchronizing
                # unrelated device work.
                slot.ready_event.synchronize()
        self._point_to_slot(self.blocks[block_id], slot_id)
        slot.active = True
        self._active_block_id = block_id

        if self.prefetch_enabled:
            next_block_id = self.blocks[block_id].next_block_id
            try:
                self._load_block(next_block_id, exclude=slot_id)
            except Exception:
                slot.active = False
                self._point_to_placeholder(self.blocks[block_id])
                self._active_block_id = None
                raise
        return slot_id

    @torch.compiler.disable
    def after_forward(self, block_id: int, slot_id: int) -> None:
        if self._active_block_id != block_id:
            raise RuntimeError(
                f"Distributed layerwise block completion mismatch: active={self._active_block_id}, done={block_id}."
            )
        slot = self.slots[slot_id]
        slot.compute_done_event.record(current_omni_platform.current_stream())
        slot.has_compute_done = True
        if not self.prefetch_enabled:
            # Complete the serial ablation contract: the host must not reach
            # the following block and enqueue its preparation while this
            # block's compute is still running asynchronously.
            slot.compute_done_event.synchronize()
        slot.active = False
        self._point_to_placeholder(self.blocks[block_id])
        self._active_block_id = None

    def prefetch_first_block(self) -> None:
        self._load_block(self.rings[0][0])


class DistributedLayerwiseOffloadHook(ModelHook):
    """Execute one block through the shared distributed weight controller."""

    _HOOK_NAME = "distributed_layerwise_offload"

    def __init__(self, controller: DistributedLayerwiseWeightController, block_id: int) -> None:
        self.controller = controller
        self.block_id = block_id

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        if self.controller.blocks[self.block_id].module is not module:
            raise ValueError("Distributed layerwise hook was registered on the wrong block.")
        return super().initialize_hook(module)

    def new_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> Any:
        slot_id = self.controller.before_forward(self.block_id)
        try:
            with record_function(f"distributed_layerwise_offload.compute.block_{self.block_id}"):
                return self.fn_ref.original_forward(*args, **kwargs)
        finally:
            self.controller.after_forward(self.block_id, slot_id)


def apply_distributed_block_hook(
    module: nn.Module,
    controller: DistributedLayerwiseWeightController,
    block_id: int,
) -> DistributedLayerwiseOffloadHook:
    registry = HookRegistry.get_or_create(module)
    hook = DistributedLayerwiseOffloadHook(controller, block_id)
    registry.register_hook(DistributedLayerwiseOffloadHook._HOOK_NAME, hook)
    return hook


def remove_distributed_block_hook(module: nn.Module) -> None:
    registry: HookRegistry | None = getattr(module, "_hook_registry", None)
    if registry is not None:
        registry.remove_hook(DistributedLayerwiseOffloadHook._HOOK_NAME)


class DistributedLayerwiseOffloadBackend(OffloadBackend):
    """Shard DiT blocks over DP or SP ranks and reconstruct through two slots."""

    def __init__(self, config: OffloadConfig, device: torch.device):
        super().__init__(config, device)
        self.controller: DistributedLayerwiseWeightController | None = None
        self._blocks: list[list[nn.Module]] = []
        self._hooked_blocks: list[nn.Module] = []

    def _validate_and_get_group(self) -> GroupCoordinator:
        if self.config.use_hsdp:
            raise ValueError("Distributed layerwise offload cannot be combined with HSDP/DTensor sharding.")
        if self.config.tensor_parallel_size != 1:
            raise ValueError("Distributed layerwise offload does not support tensor parallelism.")
        if self.config.pipeline_parallel_size != 1:
            raise ValueError("Distributed layerwise offload does not support pipeline parallelism.")
        if self.config.cfg_parallel_size != 1:
            raise ValueError("Distributed layerwise offload does not support CFG parallelism.")
        if self.config.cache_backend not in (None, "none"):
            raise ValueError(
                "Distributed layerwise offload does not support cache backends because skipped blocks "
                "would make collective ordering diverge across ranks."
            )

        dp_size = self.config.data_parallel_size
        sp_size = self.config.sequence_parallel_size
        if dp_size > 1 and sp_size > 1:
            raise ValueError("Distributed layerwise offload does not support hybrid DP + SP.")
        if dp_size > 1:
            group = get_dp_group()
            expected_size = dp_size
            group_name = "DP"
        elif sp_size > 1:
            group = get_sp_group()
            expected_size = sp_size
            group_name = "SP"
        else:
            raise ValueError("Distributed layerwise offload requires either DP > 1 or SP > 1.")

        if group.world_size != expected_size:
            raise RuntimeError(
                f"{group_name} GroupCoordinator size {group.world_size} does not match config size {expected_size}."
            )
        if group.device_group is None:
            raise RuntimeError(f"{group_name} GroupCoordinator has no device process group.")
        return group

    def enable(self, pipeline: nn.Module) -> None:
        if self.enabled:
            logger.warning("DistributedLayerwiseOffloadBackend is already enabled.")
            return

        group = self._validate_and_get_group()
        modules = ModuleDiscovery.discover(pipeline)
        if not modules.dits:
            raise ValueError("No DiT/transformer modules were found for distributed layerwise offload.")

        rings: list[list[nn.Module]] = []
        dit_block_attrs: dict[int, set[str]] = {}
        for dit_name, dit_module in zip(modules.dit_names, modules.dits):
            block_attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(dit_module)
            if len(blocks) <= 1:
                logger.warning(
                    "Skipping distributed layerwise offload for %s: expected at least two blocks, got %d.",
                    dit_name,
                    len(blocks),
                )
                continue
            rings.append(blocks)
            dit_block_attrs[id(dit_module)] = set(block_attr_names)

        if not rings:
            raise ValueError("No DiT block ring with at least two blocks was found.")

        controller: DistributedLayerwiseWeightController | None = None
        registered: list[nn.Module] = []
        try:
            controller = DistributedLayerwiseWeightController(
                rings,
                device=self.device,
                group=group,
                pin_memory=self.config.pin_cpu_memory,
                prefetch=self.config.distributed_layerwise_offload_prefetch,
            )
            controller.install_placeholders()

            # Encoders, VAEs, and non-block DiT components remain resident.
            for module in [*modules.encoders, *modules.vaes, *modules.resident_modules]:
                module.to(self.device)

            discovered_dit_ids = {id(module) for module in modules.dits}
            for dit_module in modules.dits:
                block_attrs = dit_block_attrs.get(id(dit_module), set())
                for child_name, child in dit_module.named_children():
                    if child_name in block_attrs or id(child) in discovered_dit_ids:
                        continue
                    child.to(self.device)
                for parameter in dit_module._parameters.values():
                    if parameter is not None:
                        parameter.data = parameter.data.to(self.device, non_blocking=True)
                for buffer in dit_module._buffers.values():
                    if buffer is not None:
                        buffer.data = buffer.data.to(self.device, non_blocking=True)

            for ring in rings:
                for block in ring:
                    apply_distributed_block_hook(block, controller, controller.block_id_for(block))
                    registered.append(block)

            controller.prefetch_first_block()
            controller.commit_originals()
        except Exception:
            for block in registered:
                remove_distributed_block_hook(block)
            if controller is not None:
                controller.rollback_originals()
            raise

        self.controller = controller
        self._blocks = rings
        self._hooked_blocks = registered
        self.enabled = True
        logger.info(
            "Distributed layerwise offload enabled on %d blocks across %d rings using %s%d; prefetch=%s.",
            sum(len(ring) for ring in rings),
            len(rings),
            "DP" if self.config.data_parallel_size > 1 else "SP",
            group.world_size,
            self.config.distributed_layerwise_offload_prefetch,
        )

    def disable(self) -> None:
        if not self.enabled:
            return
        assert self.controller is not None
        self.controller.quiesce()
        for block in self._hooked_blocks:
            remove_distributed_block_hook(block)
        self._hooked_blocks.clear()
        self._blocks.clear()
        self.controller = None
        self.enabled = False
        logger.info("Distributed layerwise offload disabled; reload the pipeline before reusing it.")


__all__ = [
    "DistributedLayerwiseOffloadBackend",
    "DistributedLayerwiseOffloadHook",
    "DistributedLayerwiseWeightController",
    "apply_distributed_block_hook",
    "remove_distributed_block_hook",
]
