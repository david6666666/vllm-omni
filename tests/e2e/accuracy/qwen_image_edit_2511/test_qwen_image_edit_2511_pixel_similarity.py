from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from tests.conftest import OmniServer
from tests.e2e.accuracy.qwen_image_edit_2511.common import (
    HEIGHT,
    MEAN_ABS_DIFF_THRESHOLD,
    P99_ABS_DIFF_THRESHOLD,
    WIDTH,
    artifact_paths,
    build_diffusers_baseline_command,
    resolve_configured_image_sources,
)
from tests.e2e.accuracy.qwen_image_edit_2511.test_qwen_image_edit_2511_online_serving import (
    SERVER_CASES,
    generate_online_image,
)
from tests.e2e.accuracy.utils import (
    build_abs_diff_image,
    build_pythonpath_env,
    compute_image_diff_metrics,
    load_rgb_image,
    write_json,
)
from tests.utils import hardware_test

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = REPO_ROOT.parent
RUNNER_PATH = Path(__file__).with_name("run_qwen_image_edit_2511_diffusers_baseline.py")
IMAGE_TIMEOUT_SECONDS = 60 * 60


def _runner_env() -> dict[str, str]:
    return build_pythonpath_env(REPO_ROOT, WORKSPACE_ROOT / "diffusers" / "src")


def _build_diffusers_command(
    *,
    runner_path: Path,
    image_sources: list[str],
    output_path: Path,
    metadata_path: Path,
) -> list[str]:
    return build_diffusers_baseline_command(runner_path, image_sources, output_path, metadata_path)


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


@pytest.mark.advanced_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_qwen_image_edit_2511_serving_matches_diffusers_pixel_similarity(
    omni_server: OmniServer,
    accuracy_artifact_root: Path,
    qwen_image_edit_2511_image_sources: list[str] | None,
) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("Qwen-Image-Edit-2511 pixel similarity test requires >= 1 CUDA GPU.")
    if not RUNNER_PATH.exists():
        raise AssertionError(f"Diffusers baseline runner does not exist: {RUNNER_PATH}")

    image_sources = resolve_configured_image_sources(qwen_image_edit_2511_image_sources)
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

    online_image = load_rgb_image(online_path)
    offline_image = load_rgb_image(offline_path)

    assert online_image.size == (WIDTH, HEIGHT)
    if offline_image.size != (WIDTH, HEIGHT):
        pytest.skip(
            "Diffusers baseline returned an unexpected output size: "
            f"{offline_image.size}, expected {(WIDTH, HEIGHT)}"
        )

    metrics = compute_image_diff_metrics(online_image, offline_image)
    diff_image = build_abs_diff_image(online_image, offline_image)
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
