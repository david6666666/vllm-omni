from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import torch

from tests.conftest import OmniServer


def pytest_addoption(parser):
    group = parser.getgroup("video-accuracy-e2e")
    group.addoption("--vbench-root", action="store", default=None, help="Local VBench repo root")
    group.addoption(
        "--vbench-t2v-model",
        action="store",
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="Text-to-video model for VBench smoke tests",
    )
    group.addoption(
        "--vbench-i2v-model",
        action="store",
        default="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        help="Image-to-video model for VBench-I2V smoke tests",
    )


@pytest.fixture(scope="session")
def vbench_root(request: pytest.FixtureRequest) -> Path:
    configured = request.config.getoption("vbench_root") or os.environ.get("VBENCH_REPO_ROOT")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(Path(__file__).resolve().parents[4] / "VBench")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    pytest.skip("VBench repo root not found. Pass --vbench-root or set VBENCH_REPO_ROOT.")


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


def infer_model_label(model: str) -> str:
    label = Path(model.rstrip("/\\")).name or "model"
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in label)


@pytest.fixture
def vbench_t2v_server(request: pytest.FixtureRequest):
    if torch.cuda.device_count() < 1:
        pytest.skip("Need at least 1 CUDA GPU for VBench smoke tests.")
    model = request.config.getoption("vbench_t2v_model")
    with OmniServer(
        model,
        [
            "--num-gpus",
            "1",
            "--boundary-ratio",
            "0.875",
            "--flow-shift",
            "5.0",
            "--disable-log-stats",
        ],
    ) as server:
        yield server


@pytest.fixture
def vbench_i2v_server(request: pytest.FixtureRequest):
    if torch.cuda.device_count() < 1:
        pytest.skip("Need at least 1 CUDA GPU for VBench-I2V smoke tests.")
    model = request.config.getoption("vbench_i2v_model")
    with OmniServer(
        model,
        [
            "--num-gpus",
            "1",
            "--boundary-ratio",
            "0.875",
            "--flow-shift",
            "5.0",
            "--disable-log-stats",
        ],
    ) as server:
        yield server

