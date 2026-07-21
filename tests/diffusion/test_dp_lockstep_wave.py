# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import ANY, Mock

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.ipc import DIFFUSION_RPC_RESULT_ENVELOPE
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.sched.request_scheduler import RequestScheduler
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker, WorkerProc
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def _config(*, dp: int = 4, max_wait_ms: float = 100.0, sp: int = 1, dlo: bool = True):
    return SimpleNamespace(
        max_num_seqs=32,
        request_batch_max_wait_ms=max_wait_ms,
        enable_distributed_layerwise_offload=dlo,
        parallel_config=SimpleNamespace(
            data_parallel_size=dp,
            tensor_parallel_size=1,
            sequence_parallel_size=sp,
            ulysses_degree=sp,
            ring_degree=1,
            cfg_parallel_size=1,
            pipeline_parallel_size=1,
        ),
    )


def _request(
    request_id: str,
    *,
    num_steps: int = 2,
    num_frames: int = 1,
    seed: int = 1,
    extra_args: dict | None = None,
    input_modality: str | None = None,
) -> OmniDiffusionRequest:
    prompt: str | dict = f"prompt-{request_id}"
    if input_modality is not None:
        prompt = {
            "prompt": f"prompt-{request_id}",
            "multi_modal_data": {input_modality: object()},
        }
    return OmniDiffusionRequest(
        prompt=prompt,
        request_id=request_id,
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=num_steps,
            num_frames=num_frames,
            seed=seed,
            extra_args=extra_args or {},
        ),
    )


class _ResultQueue:
    def __init__(self) -> None:
        self.queue: queue.Queue = queue.Queue()

    def dequeue(self, timeout=None):
        return self.queue.get(timeout=timeout)


class _BroadcastQueue:
    def __init__(self) -> None:
        self.queue: queue.Queue = queue.Queue()

    def enqueue(self, item) -> None:
        self.queue.put(item)


def _executor_with_rank_queues(num_gpus: int = 4):
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor.od_config = SimpleNamespace(num_gpus=num_gpus)
    executor._closed = False
    executor.is_failed = False
    executor._failure_callbacks_lock = threading.Lock()
    executor._failure_callbacks = []
    executor._failure_callbacks_notified = False
    executor._broadcast_mq = _BroadcastQueue()
    executor._result_mqs = [_ResultQueue() for _ in range(num_gpus)]
    executor._result_mq = executor._result_mqs[0]
    return executor


def test_dp_lockstep_activates_only_for_pure_dp() -> None:
    scheduler = RequestScheduler()
    scheduler.initialize(_config(dp=4, sp=1))
    assert scheduler.dp_lockstep_enabled is True
    assert scheduler.dp_wave_size == 4
    assert scheduler.max_num_running_reqs == 1

    scheduler.initialize(_config(dp=4, sp=2))
    assert scheduler.dp_lockstep_enabled is False

    scheduler.initialize(_config(dp=4, sp=1, dlo=False))
    assert scheduler.dp_lockstep_enabled is False
    assert scheduler.max_num_running_reqs == 32
    scheduler.add_request(_request("legacy"))
    assert scheduler.schedule().dp_wave_reqs is None


def test_one_logical_request_is_immediately_replicated_and_next_request_waits() -> None:
    scheduler = RequestScheduler()
    scheduler.initialize(_config(max_wait_ms=10_000.0))
    scheduler.add_request(_request("a", seed=1))
    scheduler.add_request(_request("b", seed=2))

    output = scheduler.schedule()

    assert output.scheduled_request_ids == ["a"]
    assert output.dp_wave_request_ids == ["a"] * 4
    assert output.num_running_reqs == 1
    assert output.num_waiting_reqs == 1


def test_dummy_single_request_is_immediately_replicated_to_full_wave() -> None:
    scheduler = RequestScheduler()
    scheduler.initialize(_config(max_wait_ms=10_000.0))
    scheduler.add_request(_request(DUMMY_DIFFUSION_REQUEST_ID))

    output = scheduler.schedule()

    assert output.scheduled_request_ids == [DUMMY_DIFFUSION_REQUEST_ID]
    assert output.dp_wave_request_ids == [DUMMY_DIFFUSION_REQUEST_ID] * 4


def test_different_modalities_are_never_assigned_to_different_shard_ranks() -> None:
    scheduler = RequestScheduler()
    scheduler.initialize(_config(max_wait_ms=0.0))
    scheduler.add_request(_request("image", num_steps=2, num_frames=1))
    scheduler.add_request(_request("video", num_steps=2, num_frames=9))

    output = scheduler.schedule()

    assert output.scheduled_request_ids == ["image"]
    assert output.dp_wave_request_ids == ["image"] * 4
    assert output.num_waiting_reqs == 1


def test_four_rank_replicated_wave_returns_only_rank_zero_media() -> None:
    scheduler = RequestScheduler()
    scheduler.initialize(_config(max_wait_ms=0.0))
    scheduler.add_request(_request("a", seed=1))
    scheduler_output = scheduler.schedule()
    assert scheduler_output.dp_wave_request_ids == ["a"] * 4

    executor = _executor_with_rank_queues()

    def _workers_reply_out_of_order() -> None:
        rpc = executor._broadcast_mq.queue.get(timeout=2)
        # Rank 3 completes first and rank 0 last. Every non-zero rank sends a
        # lightweight status ack; only rank zero transports generated media.
        for rank in (3, 2, 1, 0):
            request_id = rpc["wave_request_ids"][rank]
            executor._result_mqs[rank].queue.put(
                {
                    "type": DIFFUSION_RPC_RESULT_ENVELOPE,
                    "wave_rpc_id": rpc["wave_rpc_id"],
                    "rank": rank,
                    "request_id": request_id,
                    "ok": True,
                    "result": DiffusionOutput(output="a@rank0") if rank == 0 else None,
                }
            )
            time.sleep(0.005)

    worker_thread = threading.Thread(target=_workers_reply_out_of_order)
    worker_thread.start()
    output = executor.execute_request(scheduler_output)
    worker_thread.join(timeout=2)

    assert isinstance(output, BatchRunnerOutput)
    assert output.request_ids == ["a"]
    assert len(output.runner_outputs) == 1
    assert output.get_request_output("a").result.output == "a@rank0"
    assert all(result_queue.queue.empty() for result_queue in executor._result_mqs)


def test_rank_failure_is_detected_out_of_order_and_kills_dp_wave() -> None:
    executor = _executor_with_rank_queues()
    executor.shutdown = Mock()
    failure_callback = Mock()
    executor._failure_callbacks = [failure_callback]
    executor._failure_callbacks_notified = False
    executor._result_mqs[2].queue.put(
        {
            "type": DIFFUSION_RPC_RESULT_ENVELOPE,
            "wave_rpc_id": 1,
            "rank": 2,
            "request_id": "a",
            "ok": False,
            "result": None,
            "error": "failed before collective",
            "error_type": "RuntimeError",
        }
    )

    started = time.monotonic()
    with pytest.raises(EngineDeadError, match="rank 2 failed request"):
        executor.collective_rpc(
            "execute_model_dp_wave",
            timeout=1.0,
            exec_all_ranks=True,
            wave_request_ids=["a"] * 4,
        )

    assert time.monotonic() - started < 0.8
    assert executor.is_failed is True
    failure_callback.assert_called_once_with()
    executor.shutdown.assert_called_once_with()


def test_execute_request_propagates_fatal_dp_wave_failure() -> None:
    scheduler = RequestScheduler()
    scheduler.initialize(_config(max_wait_ms=0.0))
    scheduler.add_request(_request("a"))
    scheduler_output = scheduler.schedule()
    executor = _executor_with_rank_queues()
    executor.collective_rpc = Mock(side_effect=EngineDeadError("fatal DP wave"))

    with pytest.raises(EngineDeadError, match="fatal DP wave"):
        executor.execute_request(scheduler_output)


def test_worker_executes_the_same_replicated_request() -> None:
    worker = object.__new__(DiffusionWorker)
    worker.execute_model = Mock(return_value=DiffusionOutput(output="replicated"))
    req = _request("a")

    output = worker.execute_model_dp_wave([req] * 4, SimpleNamespace())

    assert output.request_id == "a"
    worker.execute_model.assert_called_once_with(req, ANY, None)


def test_worker_rejects_different_requests_across_shard_ranks() -> None:
    worker = object.__new__(DiffusionWorker)
    worker.execute_model = Mock()

    with pytest.raises(RuntimeError, match="one replicated logical request"):
        worker.execute_model_dp_wave([_request("a"), _request("b")], SimpleNamespace())

    worker.execute_model.assert_not_called()


def test_worker_wave_envelope_carries_rank_and_request_id() -> None:
    proc = object.__new__(WorkerProc)
    proc.gpu_id = 2
    proc.result_mq = object()
    proc.worker = SimpleNamespace(
        execute_method=Mock(
            return_value=RunnerOutput(
                request_id="a",
                finished=True,
                result=DiffusionOutput(output="result-a"),
            )
        )
    )

    response, should_reply = proc.execute_rpc(
        {
            "method": "execute_model_dp_wave",
            "args": (),
            "kwargs": {},
            "output_rank": None,
            "exec_all_ranks": True,
            "collect_rank_status": False,
            "wave_rpc_id": 7,
            "wave_request_ids": ["a"] * 4,
        }
    )

    assert should_reply is True
    assert response["wave_rpc_id"] == 7
    assert response["rank"] == 2
    assert response["request_id"] == "a"
    assert response["result"] is None


def test_rank_zero_wave_envelope_carries_media_result() -> None:
    proc = object.__new__(WorkerProc)
    proc.gpu_id = 0
    proc.result_mq = object()
    proc.worker = SimpleNamespace(
        execute_method=Mock(
            return_value=RunnerOutput(
                request_id="a",
                finished=True,
                result=DiffusionOutput(output="result-a"),
            )
        )
    )

    response, should_reply = proc.execute_rpc(
        {
            "method": "execute_model_dp_wave",
            "args": (),
            "kwargs": {},
            "output_rank": None,
            "exec_all_ranks": True,
            "collect_rank_status": False,
            "wave_rpc_id": 8,
            "wave_request_ids": ["a"] * 4,
        }
    )

    assert should_reply is True
    assert response["rank"] == 0
    assert response["result"].output == "result-a"


def test_executor_rejects_distinct_request_assignments() -> None:
    executor = _executor_with_rank_queues()

    with pytest.raises(ValueError, match="same logical request"):
        executor.collective_rpc(
            "execute_model_dp_wave",
            exec_all_ranks=True,
            wave_request_ids=["a", "b", "a", "b"],
        )

    assert executor._broadcast_mq.queue.empty()


def test_worker_normal_all_rank_rpc_replies_only_on_designated_rank() -> None:
    proc = object.__new__(WorkerProc)
    proc.gpu_id = 2
    proc.result_mq = object()
    proc.worker = SimpleNamespace(execute_method=Mock(return_value="rank-2-result"))

    result, should_reply = proc.execute_rpc(
        {
            "method": "ping",
            "args": (),
            "kwargs": {},
            "output_rank": 0,
            "exec_all_ranks": True,
            "collect_rank_status": False,
        }
    )

    assert result == "rank-2-result"
    assert should_reply is False


def test_normal_rpc_reads_only_the_designated_rank_queue() -> None:
    executor = _executor_with_rank_queues()
    executor._result_mqs[2].queue.put(DiffusionOutput(output="rank-2"))

    result = executor.collective_rpc("ping", unique_reply_rank=2, exec_all_ranks=True)
    rpc = executor._broadcast_mq.queue.get_nowait()

    assert result.output == "rank-2"
    assert rpc["output_rank"] == 2
    assert rpc["collect_rank_status"] is False
    assert executor._result_mqs[0].queue.empty()


def test_normal_rpc_discards_a_late_wave_reply() -> None:
    executor = _executor_with_rank_queues()
    executor._result_mqs[0].queue.put(
        {
            "type": DIFFUSION_RPC_RESULT_ENVELOPE,
            "wave_rpc_id": 1,
            "rank": 0,
            "request_id": "old",
            "ok": False,
            "error": "late error",
        }
    )
    executor._result_mqs[0].queue.put(DiffusionOutput(output="current"))

    result = executor.collective_rpc("ping", unique_reply_rank=0)

    assert result.output == "current"
