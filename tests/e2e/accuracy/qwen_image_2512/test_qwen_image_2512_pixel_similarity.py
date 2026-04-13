from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from benchmarks.accuracy.common import load_json, write_json
from tests.conftest import OmniServer
from tests.e2e.accuracy.qwen_image_2512.test_qwen_image_2512_online_serving import (
    HEIGHT,
    MODEL_NAME,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    PROMPT,
    SEED,
    SERVER_CASES,
    TRUE_CFG_SCALE,
    WIDTH,
    generate_online_image,
)
from tests.utils import hardware_test

RUNNER_PATH = Path(__file__).with_name("run_qwen_image_2512_diffusers_baseline.py")
REPO_ROOT = Path(__file__).resolve().parents[4]
RESULT_ROOT = Path(__file__).parent / "result"
GENERATION_TIMEOUT_SECONDS = 60 * 60
MAX_ABS_DIFF_THRESHOLD = int(os.environ.get("QWEN_IMAGE_2512_MAX_ABS_DIFF_THRESHOLD", "64"))
MEAN_ABS_DIFF_THRESHOLD = float(os.environ.get("QWEN_IMAGE_2512_MEAN_ABS_DIFF_THRESHOLD", "8.0"))
P95_ABS_DIFF_THRESHOLD = float(os.environ.get("QWEN_IMAGE_2512_P95_ABS_DIFF_THRESHOLD", "24.0"))


def _sanitize_case_id(raw_case_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in raw_case_id)


def _resolve_case_id(request: pytest.FixtureRequest) -> str:
    callspec = getattr(request.node, "callspec", None)
    if callspec is not None:
        return _sanitize_case_id(callspec.id)
    return "default"


def _offline_artifact_paths(case_id: str) -> tuple[Path, Path]:
    return (
        RESULT_ROOT / f"qwen_image_2512_diffusers_{case_id}.png",
        RESULT_ROOT / f"qwen_image_2512_diffusers_{case_id}.json",
    )


def _comparison_path(case_id: str) -> Path:
    return RESULT_ROOT / f"qwen_image_2512_comparison_{case_id}.json"


def _run_diffusers_baseline(case_id: str) -> tuple[Path, Path]:
    output_path, metadata_path = _offline_artifact_paths(case_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(RUNNER_PATH),
        "--model",
        MODEL_NAME,
        "--prompt",
        PROMPT,
        "--negative-prompt",
        NEGATIVE_PROMPT,
        "--width",
        str(WIDTH),
        "--height",
        str(HEIGHT),
        "--num-inference-steps",
        str(NUM_INFERENCE_STEPS),
        "--true-cfg-scale",
        str(TRUE_CFG_SCALE),
        "--seed",
        str(SEED),
        "--output",
        str(output_path),
        "--metadata-output",
        str(metadata_path),
    ]
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        check=True,
        timeout=GENERATION_TIMEOUT_SECONDS,
    )
    return output_path, metadata_path


def _load_rgb_array(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32)


def _compare_images(offline_path: Path, online_path: Path) -> dict[str, float | bool | list[int]]:
    offline = _load_rgb_array(offline_path)
    online = _load_rgb_array(online_path)
    if offline.shape != online.shape:
        return {
            "shape_match": False,
            "offline_shape": list(offline.shape),
            "online_shape": list(online.shape),
        }

    diff = np.abs(offline - online)
    flat = diff.reshape(-1)
    return {
        "shape_match": True,
        "offline_shape": list(offline.shape),
        "online_shape": list(online.shape),
        "max_abs_diff": int(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "p95_abs_diff": float(np.percentile(flat, 95)),
        "rmse": float(np.sqrt(np.mean((offline - online) ** 2))),
        "exact_match_ratio": float(np.mean(flat == 0)),
    }


@pytest.mark.advanced_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_qwen_image_2512_pixel_similarity(
    omni_server: OmniServer,
    request: pytest.FixtureRequest,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Qwen-Image-2512 pixel similarity test requires CUDA.")

    case_id = _resolve_case_id(request)
    offline_path, offline_metadata_path = _run_diffusers_baseline(case_id)
    online_path, online_metadata_path, _, _ = generate_online_image(omni_server, case_id=case_id)
    metrics = _compare_images(offline_path, online_path)

    write_json(
        _comparison_path(case_id),
        {
            "thresholds": {
                "max_abs_diff": MAX_ABS_DIFF_THRESHOLD,
                "mean_abs_diff": MEAN_ABS_DIFF_THRESHOLD,
                "p95_abs_diff": P95_ABS_DIFF_THRESHOLD,
            },
            "offline": load_json(offline_metadata_path),
            "online": load_json(online_metadata_path),
            "comparison": metrics,
        },
    )

    assert metrics["shape_match"], (
        f"Image shape mismatch: offline={metrics['offline_shape']} online={metrics['online_shape']}"
    )
    assert metrics["max_abs_diff"] <= MAX_ABS_DIFF_THRESHOLD, metrics
    assert metrics["mean_abs_diff"] <= MEAN_ABS_DIFF_THRESHOLD, metrics
    assert metrics["p95_abs_diff"] <= P95_ABS_DIFF_THRESHOLD, metrics
