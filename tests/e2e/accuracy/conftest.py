from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import shutil

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
class AccuracyServerConfig:
    generate_params: OmniServerParams
    judge_params: OmniServerParams
    run_level: str
    model_prefix: str

    @contextmanager
    def generate_server(self):
        with _build_omni_server(
            self.generate_params,
            run_level=self.run_level,
            model_prefix=self.model_prefix,
        ) as server:
            yield server

    @contextmanager
    def judge_server(self):
        with _build_omni_server(
            self.judge_params,
            run_level=self.run_level,
            model_prefix=self.model_prefix,
        ) as server:
            yield server


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


@pytest.fixture(scope="session")
def accuracy_artifact_root() -> Path:
    root = Path(__file__).resolve().parent / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def reset_artifact_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def accuracy_servers(
    request: pytest.FixtureRequest,
    run_level: str,
    model_prefix: str,
) -> AccuracyServerConfig:
    if torch.cuda.device_count() < 1:
        pytest.skip("Need at least 1 CUDA GPU for accuracy benchmark smoke tests.")

    params = getattr(request, "param", {})
    generate_model = params["generate_model"]
    if not generate_model:
        pytest.skip("No generate model configured for accuracy benchmark test.")
    judge_model = params.get(
        "judge_model",
        "/workspace/models/QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ",
    )
    shared_gpu = str(
        params.get(
            "gpu",
            os.environ.get("VLLM_ACCURACY_GPU", os.environ.get("VLLM_ACCURACY_GEN_GPU", "0")),
        )
    )
    generate_server_args = ["--num-gpus", "1", *(params.get("generate_server_args") or [])]
    judge_server_args = params.get("judge_server_args") or [
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.8",
        "--limit-mm-per-prompt.image",
        "4",
    ]

    judge_env = {"CUDA_VISIBLE_DEVICES": shared_gpu}
    if os.environ.get("VLLM_LIMIT_MM_PER_PROMPT"):
        judge_env["VLLM_LIMIT_MM_PER_PROMPT"] = os.environ["VLLM_LIMIT_MM_PER_PROMPT"]

    return AccuracyServerConfig(
        generate_params=OmniServerParams(
            model=generate_model,
            port=params.get("generate_port"),
            server_args=generate_server_args,
            env_dict={"CUDA_VISIBLE_DEVICES": shared_gpu},
            use_omni=True,
        ),
        judge_params=OmniServerParams(
            model=judge_model,
            port=params.get("judge_port"),
            server_args=judge_server_args,
            env_dict=judge_env,
            use_omni=False,
        ),
        run_level=run_level,
        model_prefix=model_prefix,
    )
