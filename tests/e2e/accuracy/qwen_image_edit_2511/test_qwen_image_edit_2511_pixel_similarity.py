from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tests.conftest import OmniServer
from tests.e2e.accuracy.qwen_image_edit_2511 import (
    HEIGHT,
    MEAN_ABS_DIFF_THRESHOLD,
    P99_ABS_DIFF_THRESHOLD,
    WIDTH,
    artifact_paths,
    resolve_image_sources,
    write_json,
)
from tests.e2e.accuracy.qwen_image_edit_2511.test_qwen_image_edit_2511_online_serving import (
    SERVER_CASES,
    generate_online_image,
)
from tests.utils import hardware_test

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = REPO_ROOT.parent
RUNNER_PATH = Path(__file__).with_name("run_qwen_image_edit_2511_diffusers_baseline.py")
IMAGE_TIMEOUT_SECONDS = 60 * 60


def _runner_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [
        str(REPO_ROOT),
        str(WORKSPACE_ROOT / "diffusers" / "src"),
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return env


def _build_diffusers_command(
    *,
    runner_path: Path,
    image_sources: list[str],
    output_path: Path,
    metadata_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(runner_path),
        "--output",
        str(output_path),
        "--metadata-output",
        str(metadata_path),
    ]
    for image_source in image_sources:
        command += ["--image-source", image_source]
    return command


def _generate_offline_image(*, image_sources: list[str], accuracy_artifact_root: Path) -> Path:
    paths = artifact_paths(accuracy_artifact_root, image_sources)
    command = _build_diffusers_command(
        runner_path=RUNNER_PATH,
        image_sources=image_sources,
        output_path=paths["offline"],
        metadata_path=paths["offline_metadata"],
    )
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_runner_env(),
        check=True,
        timeout=IMAGE_TIMEOUT_SECONDS,
    )
    return paths["offline"]


def _diff_metrics(a: Image.Image, b: Image.Image) -> dict[str, float]:
    ta = np.asarray(a.convert("RGB"), dtype=np.float32) / 255.0
    tb = np.asarray(b.convert("RGB"), dtype=np.float32) / 255.0
    assert ta.shape == tb.shape, f"Image shapes differ: {ta.shape} vs {tb.shape}"
    abs_diff = np.abs(ta - tb)
    return {
        "mean_abs_diff": float(abs_diff.mean()),
        "p99_abs_diff": float(np.quantile(abs_diff.reshape(-1), 0.99)),
        "max_abs_diff": float(abs_diff.max()),
        "exact_match_ratio": float(np.mean(abs_diff == 0.0)),
    }


def _build_diff_image(a: Image.Image, b: Image.Image) -> Image.Image:
    a_uint8 = np.asarray(a.convert("RGB"), dtype=np.int16)
    b_uint8 = np.asarray(b.convert("RGB"), dtype=np.int16)
    diff = np.abs(a_uint8 - b_uint8).clip(0, 255).astype(np.uint8)
    return Image.fromarray(diff, mode="RGB")


@pytest.mark.advanced_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_qwen_image_edit_2511_serving_matches_diffusers_pixel_similarity(
    omni_server: OmniServer,
    accuracy_artifact_root: Path,
    qwen_image_edit_2511_image_sources: list[str] | None,
) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Qwen-Image-Edit-2511 pixel similarity test requires >= 2 CUDA GPUs.")
    if not RUNNER_PATH.exists():
        raise AssertionError(f"Diffusers baseline runner does not exist: {RUNNER_PATH}")

    image_sources = resolve_image_sources(qwen_image_edit_2511_image_sources)
    paths = artifact_paths(accuracy_artifact_root, image_sources)

    online_path = generate_online_image(
        omni_server=omni_server,
        image_sources=image_sources,
        accuracy_artifact_root=accuracy_artifact_root,
    )
    offline_path = _generate_offline_image(
        image_sources=image_sources,
        accuracy_artifact_root=accuracy_artifact_root,
    )

    online_image = Image.open(online_path)
    online_image.load()
    offline_image = Image.open(offline_path)
    offline_image.load()

    assert online_image.size == (WIDTH, HEIGHT)
    assert offline_image.size == (WIDTH, HEIGHT)

    metrics = _diff_metrics(online_image, offline_image)
    diff_image = _build_diff_image(online_image, offline_image)
    diff_image.save(paths["diff"])
    write_json(
        paths["compare_summary"],
        {
            "image_sources": image_sources,
            "online_path": str(online_path),
            "offline_path": str(offline_path),
            "diff_path": str(paths["diff"]),
            "mean_abs_diff_threshold": MEAN_ABS_DIFF_THRESHOLD,
            "p99_abs_diff_threshold": P99_ABS_DIFF_THRESHOLD,
            **metrics,
        },
    )

    print(
        "qwen_image_edit_2511 pixel diff stats: "
        f"mean_abs_diff={metrics['mean_abs_diff']:.6e}, "
        f"p99_abs_diff={metrics['p99_abs_diff']:.6e}, "
        f"max_abs_diff={metrics['max_abs_diff']:.6e}, "
        f"exact_match_ratio={metrics['exact_match_ratio']:.6e}; "
        f"thresholds: mean<={MEAN_ABS_DIFF_THRESHOLD:.6e}, "
        f"p99<={P99_ABS_DIFF_THRESHOLD:.6e}; "
        f"online={online_path}, offline={offline_path}, diff={paths['diff']}"
    )

    assert (
        metrics["mean_abs_diff"] <= MEAN_ABS_DIFF_THRESHOLD
        and metrics["p99_abs_diff"] <= P99_ABS_DIFF_THRESHOLD
    ), (
        f"Image diff exceeded threshold: mean_abs_diff={metrics['mean_abs_diff']:.6e}, "
        f"p99_abs_diff={metrics['p99_abs_diff']:.6e} "
        f"(thresholds: mean<={MEAN_ABS_DIFF_THRESHOLD:.6e}, "
        f"p99<={P99_ABS_DIFF_THRESHOLD:.6e})"
    )
