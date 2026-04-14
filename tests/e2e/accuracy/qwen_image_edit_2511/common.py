from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

from tests.e2e.accuracy.utils import build_image_artifact_paths, resolve_image_sources

MODEL_NAME = os.environ.get("QWEN_IMAGE_EDIT_2511_MODEL", "Qwen/Qwen-Image-Edit-2511")
PROMPT = "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'"
NEGATIVE_PROMPT = " "
TRUE_CFG_SCALE = 4.0
GUIDANCE_SCALE = 1.0
NUM_INFERENCE_STEPS = 40
NUM_IMAGES_PER_PROMPT = 1
SEED = 2918787680
# Qwen-Image-Edit-2511 aligns generated images to 16-pixel granularity.
WIDTH = 1184
HEIGHT = 2032
SIZE = f"{WIDTH}x{HEIGHT}"
OUTPUT_FORMAT = "png"
ONLINE_TIMEOUT_SECONDS = 1200
ARTIFACT_SUITE = "qwen_image_edit_2511"
DEFAULT_IMAGE_SOURCES = (
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/qwen-bear.png",
)
MEAN_ABS_DIFF_THRESHOLD = 3e-2
P99_ABS_DIFF_THRESHOLD = 2.5e-1


def resolve_configured_image_sources(configured: Sequence[str] | None) -> list[str]:
    return resolve_image_sources(configured, DEFAULT_IMAGE_SOURCES)


def artifact_paths(accuracy_artifact_root: Path, image_sources: Sequence[str]) -> dict[str, Path]:
    return build_image_artifact_paths(accuracy_artifact_root, ARTIFACT_SUITE, image_sources)


def build_diffusers_baseline_command(
    runner_path: Path,
    image_sources: Sequence[str],
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
