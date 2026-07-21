from __future__ import annotations

import multiprocessing as mp
import multiprocessing.connection
import queue
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import zmq
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.data import SHUTDOWN_MESSAGE, DiffusionOutput
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.ipc import DIFFUSION_RPC_RESULT_ENVELOPE, unpack_diffusion_output_shm
from vllm_omni.diffusion.worker import WorkerProc

if TYPE_CHECKING:
    from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput
    from vllm_omni.diffusion.worker.utils import BaseRunnerOutput

logger = init_logger(__name__)

_DEQUEUE_TIMEOUT_S = 5.0
_DP_WAVE_TIMEOUT_S = 30 * 60.0
_DP_WAVE_POLL_SLICE_S = 0.05


@dataclass
class BackgroundResources:
    """
    Used as a finalizer for clean shutdown.
    """

    broadcast_mq: MessageQueue | None = None
    result_mq: MessageQueue | None = None
    result_mqs: list[MessageQueue] | None = None
    num_workers: int = 0
    processes: list[mp.Process] | None = None

    def __call__(self):
        """Clean up background resources."""
        if hasattr(self, "wake_events") and self.wake_events:
            for ev in self.wake_events:
                ev.set()

        if self.broadcast_mq is not None:
            try:
                for _ in range(self.num_workers):
                    self.broadcast_mq.enqueue(SHUTDOWN_MESSAGE, timeout=1.0)

                self.broadcast_mq = None
                self.result_mq = None
                self.result_mqs = None
            except Exception as exc:
                logger.warning("Failed to send shutdown signal: %s", exc)

        if self.processes:
            for proc in self.processes:
                if not proc.is_alive():
                    continue
                proc.join(5)
                if proc.is_alive():
                    logger.warning("Terminating diffusion worker %s after timeout", proc.name)
                    proc.terminate()
                    proc.join(5)


class MultiprocDiffusionExecutor(DiffusionExecutor):
    uses_multiproc: bool = True

    def _init_executor(self) -> None:
        self._processes: list[mp.Process] = []
        self._closed = False
        self.is_failed = False
        self._failure_callbacks_lock = threading.Lock()
        self._failure_callbacks: list[Callable[[], None]] = []
        self._failure_callbacks_notified = False

        num_workers = self.od_config.num_gpus
        self.wake_events = [mp.Event() for _ in range(num_workers)]

        self._broadcast_mq = self._init_broadcast_queue(num_workers)
        broadcast_handle = self._broadcast_mq.export_handle()

        # Launch workers
        processes, result_handles = self._launch_workers(broadcast_handle, self.wake_events)
        self._result_mqs = self._init_result_queues(result_handles)
        # Retain the historical rank-zero alias for callers/tests that inspect
        # the ordinary single-reply transport.
        self._result_mq = self._result_mqs[0]
        self._processes = processes

        self.resources = BackgroundResources(
            broadcast_mq=self._broadcast_mq,
            result_mq=self._result_mq,
            result_mqs=self._result_mqs,
            num_workers=num_workers,
            processes=self._processes,
        )
        self._finalizer = weakref.finalize(self, self.resources)

        self.start_worker_monitor()

    def _init_broadcast_queue(self, num_workers: int) -> MessageQueue:
        return MessageQueue(
            n_reader=num_workers,
            n_local_reader=num_workers,
            local_reader_ranks=list(range(num_workers)),
        )

    def _init_result_queues(self, result_handles: list) -> list[MessageQueue]:
        result_mqs = []
        for rank, result_handle in enumerate(result_handles):
            if result_handle is None:
                raise RuntimeError(f"Failed to get result queue handle from worker rank {rank}")
            result_mqs.append(MessageQueue.create_from_handle(result_handle, 0))
        return result_mqs

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("DiffusionExecutor is closed.")
        if not getattr(self, "_result_mqs", None) and getattr(self, "_result_mq", None) is None:
            raise RuntimeError("Result queues not initialized")

    def _result_queue_for_rank(self, rank: int) -> MessageQueue:
        result_mqs = getattr(self, "_result_mqs", None)
        if result_mqs:
            if rank < 0 or rank >= len(result_mqs):
                raise RuntimeError(f"No result queue for worker rank {rank}")
            return result_mqs[rank]
        if rank == 0 and getattr(self, "_result_mq", None) is not None:
            return self._result_mq
        raise RuntimeError(f"No result queue for worker rank {rank}")

    def _dequeue_one_with_failure_polling(
        self,
        deadline: float | None,
        method: str,
        *,
        rank: int = 0,
        wave_rpc_id: int | None = None,
    ) -> Any:
        """Read one rank-local reply while ignoring timed-out wave leftovers."""
        result_mq = self._result_queue_for_rank(rank)
        while True:
            if deadline is None:
                chunk_timeout = _DEQUEUE_TIMEOUT_S
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"RPC call to {method} timed out.")
                chunk_timeout = min(_DEQUEUE_TIMEOUT_S, remaining)
            try:
                response = result_mq.dequeue(timeout=chunk_timeout)
            except (TimeoutError, zmq.error.Again, queue.Empty):
                if self.is_failed:
                    raise EngineDeadError()
                continue
            response_wave_id = response.get("wave_rpc_id") if isinstance(response, dict) else None
            if wave_rpc_id is None and response_wave_id is not None:
                logger.warning("Discarding stale DP wave reply %s from rank %d", response_wave_id, rank)
                try:
                    unpack_diffusion_output_shm(response)
                except Exception:
                    logger.debug("Failed to release stale DP wave SHM payload", exc_info=True)
                continue
            if wave_rpc_id is not None and response_wave_id != wave_rpc_id:
                logger.warning(
                    "Discarding stale reply from rank %d while waiting for DP wave %s (got %s)",
                    rank,
                    wave_rpc_id,
                    response_wave_id,
                )
                try:
                    unpack_diffusion_output_shm(response)
                except Exception:
                    logger.debug("Failed to release stale RPC SHM payload", exc_info=True)
                continue
            return response

    @staticmethod
    def _raise_for_rpc_error_dict(response: Any) -> None:
        if isinstance(response, dict) and response.get("status") == "error":
            raise RuntimeError(
                f"Worker failed with error '{response.get('error')}', "
                "please check the stack trace above for the root cause"
            )

    @staticmethod
    def _unwrap_rpc_result_envelope(response: Any) -> Any:
        if not (isinstance(response, dict) and response.get("type") == DIFFUSION_RPC_RESULT_ENVELOPE):
            return response

        rank_statuses = response.get("rank_statuses") or []
        failed = [status for status in rank_statuses if not status.get("ok", False)]
        if failed:
            details = "; ".join(
                f"rank {status.get('rank')}: {status.get('error_type') or 'Error'}: {status.get('error')}"
                for status in failed
            )
            tracebacks = "\n\n".join(
                f"rank {status.get('rank')} traceback:\n{status['traceback']}"
                for status in failed
                if status.get("traceback")
            )
            if tracebacks:
                details = f"{details}\n\n{tracebacks}"
            method = response.get("method", "<unknown>")
            raise RuntimeError(f"RPC '{method}' failed on worker rank(s): {details}")

        result = response.get("result")
        if isinstance(result, bool):
            # Only bool-returning RPCs participate in the all-rank AND.
            # Non-bool results leave bool_result unset and are ignored here.
            bool_results = [
                status.get("bool_result") for status in rank_statuses if status.get("bool_result") is not None
            ]
            if bool_results and not all(bool_results):
                return False
        return result

    @staticmethod
    def _handle_rpc_response(response: Any) -> Any:
        MultiprocDiffusionExecutor._raise_for_rpc_error_dict(response)
        response = MultiprocDiffusionExecutor._unwrap_rpc_result_envelope(response)
        # After unwrapping, a worker method result may itself be the same
        # {"status": "error"} shape produced by worker_busy_loop transport
        # failures. Preserve the pre-envelope error handling for that case.
        MultiprocDiffusionExecutor._raise_for_rpc_error_dict(response)
        return response

    def _launch_workers(self, broadcast_handle, wake_events):
        od_config = self.od_config
        logger.info("Starting server...")

        num_gpus = od_config.num_gpus
        mp.set_start_method("spawn", force=True)
        processes = []

        # Extract worker_extension_cls and custom_pipeline_args from od_config
        worker_extension_cls = od_config.worker_extension_cls
        custom_pipeline_args = getattr(od_config, "custom_pipeline_args", None)

        # Launch all worker processes
        scheduler_pipe_readers = []
        scheduler_pipe_writers = []

        for i in range(num_gpus):
            reader, writer = mp.Pipe(duplex=False)
            scheduler_pipe_writers.append(writer)
            process = mp.Process(
                target=WorkerProc.worker_main,
                args=(
                    i,  # rank
                    od_config,
                    writer,
                    broadcast_handle,
                    wake_events[i],
                    worker_extension_cls,
                    custom_pipeline_args,
                ),
                name=f"DiffusionWorker-{i}",
                daemon=True,
            )
            scheduler_pipe_readers.append(reader)
            process.start()
            processes.append(process)

        # Wait for all workers to be ready
        scheduler_infos = []
        result_handles = []
        for writer in scheduler_pipe_writers:
            writer.close()

        for i, reader in enumerate(scheduler_pipe_readers):
            try:
                data = reader.recv()
            except EOFError:
                logger.error(f"Rank {i} scheduler is dead. Please check if there are relevant logs.")
                processes[i].join()
                logger.error(f"Exit code: {processes[i].exitcode}")
                raise

            if data["status"] != "ready":
                raise RuntimeError("Initialization failed. Please see the error messages above.")

            result_handles.append(data.get("result_handle"))

            scheduler_infos.append(data)
            reader.close()

        logger.debug("All workers are ready")

        return processes, result_handles

    def start_worker_monitor(self) -> None:
        # Monitors worker process liveness. If any die unexpectedly,
        # logs an error, shuts down the executor and invokes the failure
        # callback to inform the engine.
        sentinels = [p.sentinel for p in self._processes]
        if not sentinels:
            return

        def _monitor() -> None:
            try:
                finished = multiprocessing.connection.wait(sentinels)
            except OSError:
                return

            if self._closed:
                return

            dead = [p for p in self._processes if p.sentinel in finished]
            if dead:
                details = []
                for p in dead:
                    code = p.exitcode
                    # Negative exitcode == killed by signal N (-9 = SIGKILL/OOM,
                    # -11 = SIGSEGV). Surface this so callers don't only see
                    # "died unexpectedly" with no root cause.
                    if code is not None and code < 0:
                        try:
                            import signal as _signal

                            sig = _signal.Signals(-code).name
                        except (ValueError, ImportError):
                            sig = f"signal {-code}"
                        details.append(f"{p.name}(exitcode={code}, {sig})")
                    else:
                        details.append(f"{p.name}(exitcode={code})")
                logger.error(
                    "Diffusion worker(s) died unexpectedly: %s",
                    details,
                )
                self.is_failed = True

            self.shutdown()

            self._notify_failure_callbacks()

        t = threading.Thread(target=_monitor, daemon=True, name="diffusion-worker-monitor")
        t.start()

    def register_failure_callback(
        self,
        callback: Callable[[], None],
    ) -> None:
        """Register a callback invoked when a worker process dies."""
        with self._failure_callbacks_lock:
            notify_now = self._failure_callbacks_notified
            if not notify_now:
                self._failure_callbacks.append(callback)

        if notify_now:
            callback()

    def _notify_failure_callbacks(self) -> None:
        """Notify engine owners exactly once that the executor is terminal."""
        with self._failure_callbacks_lock:
            if self._failure_callbacks_notified:
                return
            self._failure_callbacks_notified = True
            callbacks = list(self._failure_callbacks)

        for callback in callbacks:
            try:
                callback()
            except Exception:
                logger.exception("failure_callback raised")

    def execute_request(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        """Adapt request-mode scheduler output to worker execute_model RPCs.

        Returns a BatchRunnerOutput with one RunnerOutput per scheduled request.
        """
        from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

        self._ensure_open()
        runner_outputs: list[RunnerOutput] = []

        if getattr(scheduler_output, "dp_wave_reqs", None) is not None:
            wave_reqs = scheduler_output.dp_wave_reqs
            wave_request_ids = [new_req.request_id for new_req in wave_reqs]
            wave_outputs = self.collective_rpc(
                "execute_model_dp_wave",
                args=([new_req.req for new_req in wave_reqs], self.od_config, scheduler_output.kv_prefetch_jobs),
                exec_all_ranks=True,
                wave_request_ids=wave_request_ids,
            )
            outputs_by_request_id: dict[str, RunnerOutput] = {}
            for wave_output in wave_outputs:
                outputs_by_request_id.setdefault(wave_output.request_id, wave_output)
            for request_id in scheduler_output.scheduled_request_ids:
                output = outputs_by_request_id.get(request_id)
                if output is None:
                    raise RuntimeError(f"DP wave returned no result for request {request_id!r}")
                runner_outputs.append(output)
            return BatchRunnerOutput.from_list(runner_outputs)

        for new_req in scheduler_output.scheduled_new_reqs:
            req = new_req.req
            try:
                result = self.collective_rpc(
                    "execute_model",
                    args=(req, self.od_config, scheduler_output.kv_prefetch_jobs),
                    unique_reply_rank=0,
                    exec_all_ranks=True,
                )
                if not isinstance(result, DiffusionOutput):
                    raise RuntimeError(f"Unexpected response type: {type(result)!r}")
                runner_outputs.append(
                    RunnerOutput(
                        request_id=new_req.request_id,
                        step_index=None,
                        finished=True,
                        result=result,
                    )
                )
            except Exception as exc:
                runner_outputs.append(
                    RunnerOutput(
                        request_id=new_req.request_id,
                        step_index=None,
                        finished=True,
                        result=DiffusionOutput(error=str(exc)),
                    )
                )

        return BatchRunnerOutput.from_list(runner_outputs)

    def execute_batch(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        """Execute request-mode work through a single batched worker RPC.

        The worker builds DiffusionRequestBatch from scheduler output and returns
        BatchRunnerOutput with one RunnerOutput per scheduled request.
        """
        from vllm_omni.diffusion.worker.utils import BatchRunnerOutput

        self._ensure_open()
        result = self.collective_rpc(
            "execute_model_batch",
            args=(scheduler_output, self.od_config),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )
        if not isinstance(result, BatchRunnerOutput):
            raise RuntimeError(f"Unexpected response type for execute_batch: {type(result)!r}")
        return result

    def execute_step(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        """Forward step-mode scheduler output to worker execute_stepwise RPC."""
        from vllm_omni.diffusion.worker.utils import BaseRunnerOutput

        self._ensure_open()
        result = self.collective_rpc(
            "execute_stepwise",
            args=(scheduler_output,),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )

        if isinstance(result, BaseRunnerOutput):
            return result
        raise RuntimeError(f"Unexpected response type for execute_step: {type(result)!r}")

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
        exec_all_ranks: bool = False,
        wave_request_ids: list[str] | None = None,
    ) -> Any:
        self._ensure_open()

        effective_timeout = _DP_WAVE_TIMEOUT_S if wave_request_ids is not None and timeout is None else timeout
        deadline = None if effective_timeout is None else time.monotonic() + effective_timeout
        kwargs = kwargs or {}
        wave_rpc_id: int | None = None
        if wave_request_ids is not None:
            if unique_reply_rank is not None or not exec_all_ranks:
                raise ValueError("DP wave RPC requires exec_all_ranks=True and no unique_reply_rank")
            if len(wave_request_ids) != self.od_config.num_gpus:
                raise ValueError(
                    f"DP wave requires one request assignment per worker, got "
                    f"{len(wave_request_ids)} for {self.od_config.num_gpus} workers"
                )
            if len(set(wave_request_ids)) != 1:
                raise ValueError(
                    f"DLO DP lockstep requires the same logical request on every worker; got {wave_request_ids!r}"
                )
            wave_rpc_id = getattr(self, "_next_wave_rpc_id", 0) + 1
            self._next_wave_rpc_id = wave_rpc_id

        # A normal all-rank control RPC designates rank zero as its sole reply
        # queue and gathers detailed rank statuses there. A DP wave is the one
        # explicit exception: every rank replies on its own known queue.
        execute_all_ranks = unique_reply_rank is None or exec_all_ranks
        collect_rank_status = unique_reply_rank is None and wave_rpc_id is None
        rpc_request = {
            "type": "rpc",
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "output_rank": (
                None if wave_rpc_id is not None else unique_reply_rank if unique_reply_rank is not None else 0
            ),
            "exec_all_ranks": execute_all_ranks,
            "collect_rank_status": collect_rank_status,
            "wave_rpc_id": wave_rpc_id,
            "wave_request_ids": wave_request_ids,
        }

        try:
            # Broadcast RPC request to all workers via unified message queue
            self._broadcast_mq.enqueue(rpc_request)

            if wave_rpc_id is not None:
                from vllm_omni.diffusion.worker.utils import RunnerOutput

                rank_zero_response = None
                pending_ranks = set(range(len(wave_request_ids)))
                while pending_ranks:
                    if deadline is not None and time.monotonic() >= deadline:
                        pending = ", ".join(str(rank) for rank in sorted(pending_ranks))
                        raise TimeoutError(f"DP wave RPC call to {method} timed out waiting for rank(s) {pending}.")
                    for rank in sorted(tuple(pending_ranks)):
                        probe_deadline = time.monotonic() + _DP_WAVE_POLL_SLICE_S
                        if deadline is not None:
                            probe_deadline = min(probe_deadline, deadline)
                        try:
                            response = self._dequeue_one_with_failure_polling(
                                probe_deadline,
                                method,
                                rank=rank,
                                wave_rpc_id=wave_rpc_id,
                            )
                        except TimeoutError:
                            continue

                        try:
                            unpack_diffusion_output_shm(response)
                        except Exception as e:
                            logger.warning("SHM unpack failed (data may already be inline): %s", e)

                        request_id = wave_request_ids[rank]
                        if not (isinstance(response, dict) and response.get("type") == DIFFUSION_RPC_RESULT_ENVELOPE):
                            raise RuntimeError(f"DP wave rank {rank} returned an invalid response envelope")
                        if response.get("rank") != rank:
                            raise RuntimeError(
                                f"DP wave queue {rank} returned an envelope for rank {response.get('rank')}"
                            )
                        if response.get("request_id") != request_id:
                            raise RuntimeError(
                                f"DP wave rank {rank} returned request {response.get('request_id')!r}, "
                                f"expected {request_id!r}"
                            )
                        if not response.get("ok", False):
                            raise RuntimeError(
                                f"DP wave rank {rank} failed request {request_id!r}: "
                                f"{response.get('error_type') or 'Error'}: {response.get('error')}"
                            )
                        result = response.get("result")
                        if rank == 0:
                            if not isinstance(result, DiffusionOutput):
                                raise RuntimeError(
                                    f"DP wave rank zero returned unexpected result type {type(result)!r}"
                                )
                            rank_zero_response = RunnerOutput(
                                request_id=request_id,
                                step_index=None,
                                finished=True,
                                result=result,
                            )
                        elif result is not None:
                            raise RuntimeError(f"DP wave rank {rank} unexpectedly returned a media result")
                        pending_ranks.remove(rank)
                if rank_zero_response is None:
                    raise RuntimeError("DLO DP lockstep returned no rank-zero result")
                return [rank_zero_response]

            reply_rank = unique_reply_rank if unique_reply_rank is not None else 0
            response = self._dequeue_one_with_failure_polling(deadline, method, rank=reply_rank)

            try:
                unpack_diffusion_output_shm(response)
            except Exception as e:
                logger.warning("SHM unpack failed (data may already be inline): %s", e)

            response = MultiprocDiffusionExecutor._handle_rpc_response(response)

            return response if unique_reply_rank is not None else [response]
        except Exception as e:
            logger.error(f"RPC call failed: {e}")
            if wave_rpc_id is not None:
                # A rank-local failure may leave peers blocked inside a
                # collective. The process group is no longer reusable: tear
                # down every worker and surface an engine-fatal error.
                self.is_failed = True
                self._notify_failure_callbacks()
                try:
                    self.shutdown()
                except Exception:
                    logger.exception("Failed to shut down executor after fatal DP wave error")
                raise EngineDeadError(f"DLO DP lockstep RPC failed fatally: {e}") from e
            raise

    def check_health(self) -> None:
        if self.is_failed:
            raise EngineDeadError()
        self._ensure_open()
        for p in self._processes:
            if not p.is_alive():
                self.is_failed = True
                raise EngineDeadError(f"Worker process {p.name} is dead")

    def shutdown(self) -> None:
        self._closed = True
        try:
            self._finalizer()
        finally:
            self._broadcast_mq = None
            self._result_mqs = []
            self._result_mq = None
            self.resources = None
            self._processes = []
