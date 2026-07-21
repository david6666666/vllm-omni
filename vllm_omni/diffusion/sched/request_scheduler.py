# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_omni.diffusion.sched.base_scheduler import (
    _BaseScheduler,
    get_pure_data_parallel_size,
    get_request_batch_sampling_params_key,
)
from vllm_omni.diffusion.sched.interface import (
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    NewRequestData,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import RunnerOutput


class RequestScheduler(_BaseScheduler):
    """Diffusion scheduler with vLLM-style waiting/running queues."""

    def __init__(self) -> None:
        super().__init__()
        self.dp_lockstep_enabled = False
        self.dp_wave_size = 1

    def initialize(self, od_config) -> None:
        super().initialize(od_config)
        self.dp_wave_size = get_pure_data_parallel_size(od_config)
        self.dp_lockstep_enabled = self.dp_wave_size > 1
        if self.dp_lockstep_enabled:
            # The DP axis is the weight-shard axis under DLO, not an
            # independent request-replica axis. One logical request is
            # therefore replicated to every rank in the group.
            self.max_num_running_reqs = 1

    def _build_sampling_params_key(self, request):
        return get_request_batch_sampling_params_key(request)

    def schedule(self) -> DiffusionSchedulerOutput:
        scheduler_output = super().schedule()
        if not self.dp_lockstep_enabled or scheduler_output.is_empty:
            return scheduler_output

        if len(scheduler_output.scheduled_request_ids) != 1:
            raise RuntimeError("DLO DP lockstep must schedule exactly one logical request.")
        state = self._request_states.get(scheduler_output.scheduled_request_ids[0])
        if state is None:
            raise RuntimeError("DLO DP lockstep lost its scheduled request state.")
        rank_request = NewRequestData.from_state(state)
        scheduler_output.dp_wave_reqs = [rank_request] * self.dp_wave_size
        return scheduler_output

    def has_full_dp_wave(self) -> bool:
        """One queued logical request always fills the replicated rank wave."""
        return not self.dp_lockstep_enabled or bool(self._waiting or self._running)

    def dp_wave_wait_remaining_s(self) -> float:
        """Replicated DLO DP waves never wait for additional requests."""
        return 0.0

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: RunnerOutput) -> set[str]:
        scheduled_request_ids = sched_output.scheduled_request_ids
        if not scheduled_request_ids:
            return set()

        terminal_statuses: dict[str, DiffusionRequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}
        for request_id in scheduled_request_ids:
            state = self._request_states.get(request_id)
            if state is None or state.is_finished():
                continue
            req_output = output.get_request_output(request_id)
            result = req_output.result if req_output is not None else None
            if result is None:
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[request_id] = "No output result"
            elif result.aborted:
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_ABORTED
                terminal_errors[request_id] = None
            elif result.error:
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[request_id] = result.error
            else:
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_COMPLETED
                terminal_errors[request_id] = None

        return self._finalize_update_from_output(sched_output, terminal_statuses, terminal_errors)
