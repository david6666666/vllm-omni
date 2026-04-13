from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageChops

from tests.e2e.accuracy.qwen_image_layered.test_qwen_image_layered_01_diffusers_baseline import (
    LAYERS,
    artifact_dir,
    load_saved_layers,
    offline_layer_paths,
    resolve_image_source,
    run_diffusers_baseline,
    validate_image_source,
)
from tests.e2e.accuracy.qwen_image_layered.test_qwen_image_layered_02_online_serving import (
    SERVER_CASES,
    generate_online_layers,
    load_online_layers,
    online_layer_paths,
)
from tests.utils import hardware_test


def _pixel_metrics(expected: Image.Image, actual: Image.Image) -> dict[str, float | int]:
    expected_rgba = expected.convert("RGBA")
    actual_rgba = actual.convert("RGBA")
    expected_array = np.asarray(expected_rgba, dtype=np.int16)
    actual_array = np.asarray(actual_rgba, dtype=np.int16)
    abs_diff = np.abs(expected_array - actual_array)
    different_pixels = int(np.count_nonzero(np.any(abs_diff != 0, axis=-1)))
    return {
        "different_pixels": different_pixels,
        "max_abs_diff": int(abs_diff.max(initial=0)),
        "mean_abs_diff": float(abs_diff.mean()),
    }


def _diff_image_path(image_source: str, index: int) -> Path:
    return artifact_dir(image_source) / "compare" / f"layer_{index}_diff.png"


def _save_diff_image(image_source: str, index: int, expected: Image.Image, actual: Image.Image) -> Path:
    diff_path = _diff_image_path(image_source)
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    ImageChops.difference(expected.convert("RGBA"), actual.convert("RGBA")).save(diff_path)
    return diff_path


@pytest.mark.advanced_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_qwen_image_layered_online_matches_diffusers_pixelwise(
    omni_server,
    qwen_image_layered_image_source: str | None,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Qwen-Image-Layered pixel-compare test requires CUDA.")

    image_source = resolve_image_source(qwen_image_layered_image_source)
    validate_image_source(image_source)

    offline_paths = offline_layer_paths(image_source)
    if not all(path.exists() for path in offline_paths):
        run_diffusers_baseline(image_source)
    online_paths = online_layer_paths(image_source)
    if not all(path.exists() for path in online_paths):
        generate_online_layers(omni_server=omni_server, image_source=image_source)

    offline_images = load_saved_layers(offline_paths)
    online_images = load_online_layers(image_source)

    assert len(offline_images) == LAYERS, f"Expected {LAYERS} offline layers, got {len(offline_images)}"
    assert len(online_images) == LAYERS, f"Expected {LAYERS} online layers, got {len(online_images)}"

    for index, (offline_image, online_image) in enumerate(zip(offline_images, online_images, strict=True)):
        assert offline_image.size == online_image.size, (
            f"Layer {index} size mismatch: offline={offline_image.size}, online={online_image.size}"
        )
        metrics = _pixel_metrics(offline_image, online_image)
        print(
            "qwen_image_layered pixel compare:",
            f"layer={index}",
            f"different_pixels={metrics['different_pixels']}",
            f"max_abs_diff={metrics['max_abs_diff']}",
            f"mean_abs_diff={metrics['mean_abs_diff']:.6f}",
        )
        if metrics["different_pixels"] != 0:
            diff_path = _save_diff_image(image_source, index, offline_image, online_image)
            pytest.fail(
                "Qwen-Image-Layered pixel mismatch detected for "
                f"layer {index}: {metrics}. diff_image={diff_path}"
            )
