# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.parameter import ModelWeightParameter

import vllm_omni.diffusion.offloader.distributed_layerwise_backend as backend_module
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import (
    DistributedLayerwiseOffloadBackend,
    DistributedLayerwiseWeightController,
    apply_distributed_block_hook,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


class _DummyEvent:
    def __init__(self) -> None:
        self.recorded_on = None
        self.synchronize_calls = 0

    def record(self, stream) -> None:
        self.recorded_on = stream

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class _DummyStream:
    def __init__(self, name: str) -> None:
        self.name = name
        self.waited_events: list[_DummyEvent] = []

    def wait_event(self, event: _DummyEvent) -> None:
        self.waited_events.append(event)


@contextmanager
def _dummy_stream_context(_stream):
    yield


class _FakeGroup:
    def __init__(self, world_size: int, rank_in_group: int = 0) -> None:
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self.device_group = object()
        self.ranks = list(range(world_size))


class _ScaleBlock(nn.Module):
    def __init__(self, first: float) -> None:
        super().__init__()
        # Repeating each rank-0 half lets the fake collective reconstruct the
        # exact original tensor while still exercising a two-rank shard.
        self.weight = nn.Parameter(torch.tensor([first, first + 1, first, first + 1], dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.weight.sum()


class _FailingBlock(_ScaleBlock):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        del value
        raise RuntimeError("expected failure")


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch):
    compute_stream = _DummyStream("compute")
    stream_index = 0

    def make_stream():
        nonlocal stream_index
        stream_index += 1
        return _DummyStream(f"async-{stream_index}")

    monkeypatch.setattr(backend_module.current_omni_platform, "Stream", make_stream)
    monkeypatch.setattr(backend_module.current_omni_platform, "Event", _DummyEvent)
    monkeypatch.setattr(
        backend_module.current_omni_platform,
        "current_stream",
        lambda: compute_stream,
    )
    monkeypatch.setattr(
        backend_module.current_omni_platform,
        "stream",
        _dummy_stream_context,
    )
    monkeypatch.setattr(backend_module.current_omni_platform, "synchronize", lambda: None)

    gathered_outputs: list[torch.Tensor] = []

    def fake_all_gather_into_tensor(output, source, *, group, async_op=False):
        del group
        assert async_op is False
        gathered_outputs.append(output)
        output.copy_(source.repeat(output.numel() // source.numel()))
        return None

    monkeypatch.setattr(backend_module.dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    return compute_stream, gathered_outputs


class _TinyLanguageModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["layers"]

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ScaleBlock(1), _ScaleBlock(3), _ScaleBlock(5)])


class _TinyCosmosTransformer(nn.Module):
    _layerwise_offload_blocks_attrs = ["gen_layers"]

    def __init__(self) -> None:
        super().__init__()
        self.language_model = _TinyLanguageModel()
        self.gen_layers = nn.ModuleList([_ScaleBlock(7), _ScaleBlock(9)])


class _TinyCosmosPipeline(nn.Module):
    _dit_modules = ["transformer.language_model", "transformer"]
    _encoder_modules: list[str] = []
    _vae_modules: list[str] = []
    _resident_modules: list[str] = []

    def __init__(self) -> None:
        super().__init__()
        self.transformer = _TinyCosmosTransformer()


def _make_controller(
    rings: list[list[nn.Module]],
    *,
    world_size: int = 2,
    rank_in_group: int = 0,
    prefetch: bool = True,
) -> tuple[DistributedLayerwiseWeightController, _DummyStream, _DummyStream]:
    copy_stream = _DummyStream("copy")
    comm_stream = _DummyStream("comm")
    controller = DistributedLayerwiseWeightController(
        rings,
        device=torch.device("cpu"),
        group=_FakeGroup(world_size, rank_in_group),
        pin_memory=False,
        prefetch=prefetch,
        copy_stream=copy_stream,
        comm_stream=comm_stream,
    )
    return controller, copy_stream, comm_stream


def _install_hooks(controller: DistributedLayerwiseWeightController, rings: list[list[nn.Module]]) -> None:
    controller.install_placeholders()
    for ring in rings:
        for block in ring:
            apply_distributed_block_hook(block, controller, controller.block_id_for(block))
    controller.prefetch_first_block()
    controller.commit_originals()


def test_config_uses_explicit_distributed_strategy_and_prefetch_flag() -> None:
    od_config = SimpleNamespace(
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=True,
        distributed_layerwise_offload_prefetch=False,
        pin_cpu_memory=True,
        cache_backend="none",
        parallel_config=SimpleNamespace(
            use_hsdp=False,
            data_parallel_size=1,
            sequence_parallel_size=4,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            cfg_parallel_size=1,
        ),
    )

    config = OffloadConfig.from_od_config(od_config)

    assert config.strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE
    assert config.distributed_layerwise_offload_prefetch is False
    assert config.sequence_parallel_size == 4


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"use_hsdp": True}, "HSDP/DTensor"),
        ({"tensor_parallel_size": 2}, "tensor parallelism"),
        ({"pipeline_parallel_size": 2}, "pipeline parallelism"),
        ({"cfg_parallel_size": 2}, "CFG parallelism"),
        ({"data_parallel_size": 2, "sequence_parallel_size": 2}, r"hybrid DP \+ SP"),
        ({"cache_backend": "cache_dit"}, "cache backends"),
    ],
)
def test_backend_rejects_unsafe_parallel_or_cache_combinations(
    changes: dict,
    message: str,
) -> None:
    values = dict(
        strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
        pin_cpu_memory=False,
        use_hsdp=False,
        data_parallel_size=1,
        sequence_parallel_size=4,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        cfg_parallel_size=1,
        cache_backend="none",
    )
    values.update(changes)
    backend = DistributedLayerwiseOffloadBackend(OffloadConfig(**values), torch.device("cpu"))

    with pytest.raises(ValueError, match=message):
        backend._validate_and_get_group()


def test_backend_selects_real_group_and_group_local_size(monkeypatch: pytest.MonkeyPatch) -> None:
    dp_group = _FakeGroup(4, rank_in_group=3)
    sp_group = _FakeGroup(4, rank_in_group=2)
    monkeypatch.setattr(backend_module, "get_dp_group", lambda: dp_group)
    monkeypatch.setattr(backend_module, "get_sp_group", lambda: sp_group)

    dp_backend = DistributedLayerwiseOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
            data_parallel_size=4,
            sequence_parallel_size=1,
        ),
        torch.device("cpu"),
    )
    sp_backend = DistributedLayerwiseOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
            data_parallel_size=1,
            sequence_parallel_size=4,
        ),
        torch.device("cpu"),
    )

    assert dp_backend._validate_and_get_group() is dp_group
    assert sp_backend._validate_and_get_group() is sp_group


def test_backend_enable_keeps_nested_und_and_gen_as_two_rings(
    fake_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_runtime
    group = _FakeGroup(2)
    monkeypatch.setattr(backend_module, "get_sp_group", lambda: group)
    pipeline = _TinyCosmosPipeline()
    backend = DistributedLayerwiseOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
            pin_cpu_memory=False,
            data_parallel_size=1,
            sequence_parallel_size=2,
        ),
        torch.device("cpu"),
    )

    backend.enable(pipeline)

    assert backend.enabled
    assert backend.controller is not None
    assert len(backend.controller.rings) == 2
    und_ids, gen_ids = backend.controller.rings
    assert backend.controller.blocks[und_ids[-1]].next_block_id == und_ids[0]
    assert backend.controller.blocks[gen_ids[-1]].next_block_id == gen_ids[0]
    value = pipeline.transformer.language_model.layers[0](torch.tensor(0.0))
    value = pipeline.transformer.gen_layers[0](value)
    assert value.item() == pytest.approx(6.0 + 30.0)

    backend.disable()
    assert not backend.enabled


def test_dtensor_is_rejected_before_manual_sharding(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDTensor(torch.Tensor):
        pass

    value = torch.Tensor._make_subclass(FakeDTensor, torch.ones(2), False)
    monkeypatch.setattr(backend_module, "DTensor", FakeDTensor)

    with pytest.raises(ValueError, match="is a DTensor"):
        DistributedLayerwiseWeightController._validate_tensor("weight", value)


def test_vllm_model_weight_parameter_is_supported(fake_runtime) -> None:
    del fake_runtime

    class VllmLinearBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Bypass loader initialization: the offloader only needs the
            # already-loaded dense Parameter subclass used by vLLM linears.
            self.weight = ModelWeightParameter.__new__(
                ModelWeightParameter,
                torch.tensor([1.0, 2.0, 1.0, 2.0]),
            )

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + self.weight.sum()

    blocks = [VllmLinearBlock(), VllmLinearBlock()]
    controller, _, _ = _make_controller([blocks])
    _install_hooks(controller, [blocks])

    assert isinstance(blocks[0].weight, ModelWeightParameter)
    assert blocks[0](torch.tensor(0.0)).item() == pytest.approx(6.0)


def test_noncontiguous_parameter_is_rejected() -> None:
    parameter = nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3).t())

    with pytest.raises(ValueError, match="contiguous"):
        DistributedLayerwiseWeightController._validate_tensor("weight", parameter)


def test_parameter_marked_tensor_subclass_is_rejected() -> None:
    class WrapperTensor(torch.Tensor):
        pass

    wrapped = torch.Tensor._make_subclass(WrapperTensor, torch.ones(2), False)
    parameter = nn.Parameter(wrapped, requires_grad=False)
    assert isinstance(parameter, nn.Parameter)
    assert not issubclass(type(parameter), nn.Parameter)

    with pytest.raises(ValueError, match="tensor subclasses"):
        DistributedLayerwiseWeightController._validate_tensor("weight", parameter)


def test_cpu_storage_is_one_group_local_shard_and_padding_uses_local_rank(fake_runtime) -> None:
    del fake_runtime

    class VectorBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.arange(5, dtype=torch.float32))

    rings = [[VectorBlock(), VectorBlock()]]
    controller, _, _ = _make_controller(rings, world_size=4, rank_in_group=2)
    first = controller.blocks[0]

    # Five fp32 values -> ceil(5 / 4) == two values per rank. Group-local
    # rank 2 owns the final value plus one zero-padding element.
    assert first.cpu_shards[torch.float32].numel() == 2
    assert torch.equal(first.cpu_shards[torch.float32], torch.tensor([4.0, 0.0]))
    assert controller.full_slots[0][torch.float32].numel() == 8
    controller.install_placeholders()
    controller.commit_originals()
    assert rings[0][0].weight.numel() == 0
    assert all(
        binding.original is None
        for block in controller.blocks
        for bindings in block.bindings.values()
        for binding in bindings
    )


def test_fixed_buffers_drive_odd_rings_skips_and_cross_ring_transitions(
    fake_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compute_stream, gathered_outputs = fake_runtime
    und_ring = [_ScaleBlock(1), _ScaleBlock(3), _ScaleBlock(5)]
    gen_ring = [_ScaleBlock(7), _ScaleBlock(9), _ScaleBlock(11), _ScaleBlock(13), _ScaleBlock(15)]
    rings = [und_ring, gen_ring]
    expected_sum = {id(block): block.weight.sum().item() for ring in rings for block in ring}
    controller, copy_stream, comm_stream = _make_controller(rings, prefetch=True)
    _install_hooks(controller, rings)

    assert len(controller.full_slots) == 2
    assert len(controller.staging_slots) == 2
    full_ptrs = {tensor.data_ptr() for slot in controller.full_slots for tensor in slot.values()}
    staging_ptrs = {tensor.data_ptr() for slot in controller.staging_slots for tensor in slot.values()}
    assert len(full_ptrs) == 2
    assert len(staging_ptrs) == 2

    original_empty = torch.empty

    def fail_hot_path_empty(*args, **kwargs):
        del args, kwargs
        raise AssertionError("hot path allocated a new torch.empty tensor")

    monkeypatch.setattr(backend_module.torch, "empty", fail_hot_path_empty)
    # Jump over the prefetched UND successor, enter GEN at a non-zero block,
    # then return to UND. This exercises skip recovery and global slot ownership.
    execution = [und_ring[0], gen_ring[3], gen_ring[4], und_ring[2], und_ring[0]]
    value = torch.tensor(0.0)
    for block in execution:
        value = block(value)

    assert value.item() == pytest.approx(sum(expected_sum[id(block)] for block in execution))
    assert {tensor.data_ptr() for slot in controller.full_slots for tensor in slot.values()} == full_ptrs
    assert {tensor.data_ptr() for slot in controller.staging_slots for tensor in slot.values()} == staging_ptrs
    assert all(output.data_ptr() in full_ptrs for output in gathered_outputs)
    assert len(compute_stream.waited_events) == len(execution)
    assert any(slot.compute_done_event in comm_stream.waited_events for slot in controller.slots)
    assert any(slot.ready_event in copy_stream.waited_events for slot in controller.slots)

    monkeypatch.setattr(backend_module.torch, "empty", original_empty)


def test_prefetch_flag_controls_successor_collective(fake_runtime) -> None:
    _, gathered_outputs = fake_runtime
    blocks = [_ScaleBlock(1), _ScaleBlock(3), _ScaleBlock(5)]
    rings = [blocks]
    controller, _, _ = _make_controller(rings, prefetch=False)
    _install_hooks(controller, rings)

    assert len(gathered_outputs) == 1  # initial first-block materialization
    blocks[0](torch.tensor(0.0))
    assert len(gathered_outputs) == 1
    assert sum(slot.ready_event.synchronize_calls for slot in controller.slots) == 1
    assert sum(slot.compute_done_event.synchronize_calls for slot in controller.slots) == 1
    blocks[1](torch.tensor(0.0))
    assert len(gathered_outputs) == 2
    assert sum(slot.ready_event.synchronize_calls for slot in controller.slots) == 2
    assert sum(slot.compute_done_event.synchronize_calls for slot in controller.slots) == 2


def test_prefetch_enabled_never_host_synchronizes_ready_events(fake_runtime) -> None:
    del fake_runtime
    blocks = [_ScaleBlock(1), _ScaleBlock(3)]
    controller, _, _ = _make_controller([blocks], prefetch=True)
    _install_hooks(controller, [blocks])

    blocks[0](torch.tensor(0.0))

    assert all(slot.ready_event.synchronize_calls == 0 for slot in controller.slots)
    assert all(slot.compute_done_event.synchronize_calls == 0 for slot in controller.slots)


def test_failed_forward_records_compute_completion_and_releases_slot(fake_runtime) -> None:
    del fake_runtime
    failing = _FailingBlock(1)
    rings = [[failing, _ScaleBlock(3)]]
    controller, _, _ = _make_controller(rings)
    _install_hooks(controller, rings)

    with pytest.raises(RuntimeError, match="expected failure"):
        failing(torch.tensor(0.0))

    assert controller._active_block_id is None
    assert all(not slot.active for slot in controller.slots)
    assert failing.weight.numel() == 0
