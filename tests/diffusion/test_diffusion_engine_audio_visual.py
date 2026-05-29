# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_request(req_id: str) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[f"prompt_{req_id}"],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_id=req_id,
    )


def test_step_preserves_video_from_audio_visual_output(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_step():
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine._closed = False
        engine._loop_started = True
        engine._init_lock = asyncio.Lock()
        engine.main_loop = asyncio.get_running_loop()
        engine.stop_event = threading.Event()
        engine.od_config = SimpleNamespace(
            model_class_name="audio_visual_model",
            enable_cpu_offload=False,
        )
        engine.pre_process_func = None
        engine.post_process_func = lambda output: output
        engine._post_process_accepts_sampling_params = False
        engine.async_add_req_and_wait_for_response = AsyncMock(
            return_value=DiffusionOutput(
                output={
                    "video": "video_frames",
                    "audio": "audio_samples",
                    "audio_sample_rate": 48000,
                }
            )
        )

        return await engine.step(_make_request("audio-visual"))

    monkeypatch.setattr(
        "vllm_omni.diffusion.diffusion_engine.supports_audio_output",
        lambda model_class_name: True,
    )
    outputs = asyncio.run(run_step())

    assert len(outputs) == 1
    output = outputs[0]
    assert output.final_output_type == "image"
    assert output.images == ["video_frames"]
    assert output.multimodal_output["audio"] == "audio_samples"
    assert output.multimodal_output["audio_sample_rate"] == 48000
