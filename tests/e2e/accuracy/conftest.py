from __future__ import annotations

import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from tests.conftest import OmniServer, OmniServerParams, _build_omni_server


def _env_path_or_skip(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"Missing required environment variable: {name}")
    path = Path(value)
    if not path.exists():
        pytest.skip(f"Dataset path does not exist: {path}")
    return path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


@dataclass
class AccuracyServerBundle:
    generate_server: OmniServer
    judge_server: OmniServer

    @property
    def generate_base_url(self) -> str:
        return f"http://{self.generate_server.host}:{self.generate_server.port}"

    @property
    def judge_base_url(self) -> str:
        return f"http://{self.judge_server.host}:{self.judge_server.port}"


@pytest.fixture(scope="session")
def gebench_dataset_root() -> Path:
    return _env_path_or_skip("VLLM_TEST_GEBENCH_ROOT")


@pytest.fixture(scope="session")
def gedit_dataset_root() -> Path:
    return _env_path_or_skip("VLLM_TEST_GEDIT_ROOT")


@pytest.fixture(scope="session")
def accuracy_workers() -> int:
    return _env_int("VLLM_TEST_ACCURACY_WORKERS", 1)


@pytest.fixture(scope="session")
def gebench_samples_per_type() -> int:
    return _env_int("VLLM_TEST_GEBENCH_SAMPLES_PER_TYPE", 10)


@pytest.fixture(scope="session")
def gedit_samples_per_group() -> int:
    return _env_int("VLLM_TEST_GEDIT_SAMPLES_PER_GROUP", 10)


@pytest.fixture
def accuracy_servers(
    request: pytest.FixtureRequest,
    run_level: str,
    model_prefix: str,
) -> AccuracyServerBundle:
    if torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 CUDA GPUs for accuracy benchmark smoke tests.")

    params = getattr(request, "param", {})
    generate_model = params["generate_model"]
    if not generate_model:
        pytest.skip("No generate model configured for accuracy benchmark test.")
    judge_model = params.get(
        "judge_model",
        "/workspace/models/QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ",
    )
    generate_gpu = str(params.get("generate_gpu", os.environ.get("VLLM_ACCURACY_GEN_GPU", "0")))
    judge_gpu = str(params.get("judge_gpu", os.environ.get("VLLM_ACCURACY_JUDGE_GPU", "1")))
    generate_server_args = ["--num-gpus", "1", *(params.get("generate_server_args") or [])]
    judge_server_args = params.get("judge_server_args") or [
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.8",
        "--limit-mm-per-prompt.image",
        "4",
    ]

    with ExitStack() as stack:
        generate_server = stack.enter_context(
            _build_omni_server(
                OmniServerParams(
                    model=generate_model,
                    port=params.get("generate_port"),
                    server_args=generate_server_args,
                    env_dict={"CUDA_VISIBLE_DEVICES": generate_gpu},
                    use_omni=True,
                ),
                run_level=run_level,
                model_prefix=model_prefix,
            )
        )
        judge_server = stack.enter_context(
            _build_omni_server(
                OmniServerParams(
                    model=judge_model,
                    port=params.get("judge_port"),
                    server_args=judge_server_args,
                    env_dict={
                        "CUDA_VISIBLE_DEVICES": judge_gpu,
                        "VLLM_LIMIT_MM_PER_PROMPT": os.environ.get("VLLM_LIMIT_MM_PER_PROMPT", ""),
                    }
                    if os.environ.get("VLLM_LIMIT_MM_PER_PROMPT")
                    else {"CUDA_VISIBLE_DEVICES": judge_gpu},
                    use_omni=False,
                ),
                run_level=run_level,
                model_prefix=model_prefix,
            )
        )
        yield AccuracyServerBundle(generate_server=generate_server, judge_server=judge_server)
