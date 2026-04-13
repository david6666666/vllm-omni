from __future__ import annotations

import base64
import json
import re
import sys
from hashlib import sha1
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = REPO_ROOT.parent
DIFFUSERS_SRC = WORKSPACE_ROOT / "diffusers" / "src"
if str(DIFFUSERS_SRC) not in sys.path:
    sys.path.insert(0, str(DIFFUSERS_SRC))

from diffusers import QwenImageLayeredPipeline

from tests.utils import hardware_test

MODEL_NAME = "Qwen/Qwen-Image-Layered"
DEFAULT_IMAGE_SOURCE = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/qwen-bear.png"
NEGATIVE_PROMPT = " "
TRUE_CFG_SCALE = 4.0
NUM_INFERENCE_STEPS = 50
NUM_IMAGES_PER_PROMPT = 1
LAYERS = 3
RESOLUTION = 640
CFG_NORMALIZE = False
USE_EN_PROMPT = False
SEED = 777
RESULT_ROOT = Path(__file__).parent / "result"


def resolve_image_source(configured: str | None) -> str:
    configured = configured or DEFAULT_IMAGE_SOURCE
    candidate = Path(configured)
    if candidate.exists():
        return str(candidate.resolve())
    return configured


def is_remote_image_source(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def load_input_image(source: str) -> Image.Image:
    if source.startswith("data:image"):
        _, encoded = source.split(",", 1)
        image = Image.open(BytesIO(base64.b64decode(encoded)))
        image.load()
        return image.convert("RGBA")

    source_path = Path(source)
    if source_path.exists():
        image = Image.open(source_path)
        image.load()
        return image.convert("RGBA")

    response = requests.get(source, timeout=60)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    image.load()
    return image.convert("RGBA")


def validate_image_source(source: str) -> None:
    if is_remote_image_source(source):
        response = requests.get(source, timeout=60)
        response.raise_for_status()
        return

    if source.startswith("data:image"):
        return

    image_path = Path(source)
    if not image_path.exists():
        raise FileNotFoundError(f"Local image source does not exist: {image_path}")


def artifact_dir(image_source: str) -> Path:
    if is_remote_image_source(image_source):
        source_name = Path(urlparse(image_source).path).stem or "remote"
    elif image_source.startswith("data:image"):
        source_name = "data_url"
    else:
        source_name = Path(image_source).stem or "local"

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_name)
    digest = sha1(image_source.encode("utf-8")).hexdigest()[:8]
    return RESULT_ROOT / f"{safe_name}-{digest}"


def offline_layer_paths(image_source: str) -> list[Path]:
    base_dir = artifact_dir(image_source) / "offline"
    return [base_dir / f"layer_{index}.png" for index in range(LAYERS)]


def offline_metadata_path(image_source: str) -> Path:
    return artifact_dir(image_source) / "offline" / "metadata.json"


def load_saved_layers(paths: list[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        image = Image.open(path)
        image.load()
        images.append(image.convert("RGBA"))
    return images


def _write_offline_metadata(image_source: str, images: list[Image.Image]) -> Path:
    metadata_path = offline_metadata_path(image_source)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": MODEL_NAME,
        "image_source": image_source,
        "negative_prompt": NEGATIVE_PROMPT,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "num_images_per_prompt": NUM_IMAGES_PER_PROMPT,
        "layers": LAYERS,
        "resolution": RESOLUTION,
        "cfg_normalize": CFG_NORMALIZE,
        "use_en_prompt": USE_EN_PROMPT,
        "seed": SEED,
        "layer_sizes": [list(image.size) for image in images],
        "layer_modes": [image.mode for image in images],
        "layer_paths": [str(path) for path in offline_layer_paths(image_source)],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path


def run_diffusers_baseline(image_source: str) -> list[Image.Image]:
    input_image = load_input_image(image_source)
    pipeline = QwenImageLayeredPipeline.from_pretrained(MODEL_NAME)
    pipeline = pipeline.to("cuda", torch.bfloat16)
    pipeline.set_progress_bar_config(disable=None)

    inputs = {
        "image": input_image,
        "generator": torch.Generator(device="cuda").manual_seed(SEED),
        "true_cfg_scale": TRUE_CFG_SCALE,
        "negative_prompt": NEGATIVE_PROMPT,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "num_images_per_prompt": NUM_IMAGES_PER_PROMPT,
        "layers": LAYERS,
        "resolution": RESOLUTION,
        "cfg_normalize": CFG_NORMALIZE,
        "use_en_prompt": USE_EN_PROMPT,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)

    layered_images = output.images[0]
    if not isinstance(layered_images, list):
        raise AssertionError(f"Expected layered output to be a list, got {type(layered_images)}")

    paths = offline_layer_paths(image_source)
    for path, image in zip(paths, layered_images, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)

    _write_offline_metadata(image_source, layered_images)

    if hasattr(pipeline, "maybe_free_model_hooks"):
        pipeline.maybe_free_model_hooks()

    return layered_images


def test_resolve_image_source_prefers_existing_local_path(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGBA", (8, 8), color=(255, 128, 64, 255)).save(image_path)

    assert resolve_image_source(str(image_path)) == str(image_path.resolve())


@pytest.mark.advanced_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_qwen_image_layered_diffusers_baseline_generates_layers(
    qwen_image_layered_image_source: str | None,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Qwen-Image-Layered diffusers baseline requires CUDA.")

    image_source = resolve_image_source(qwen_image_layered_image_source)
    validate_image_source(image_source)
    images = run_diffusers_baseline(image_source)
    paths = offline_layer_paths(image_source)
    metadata_path = offline_metadata_path(image_source)

    assert len(images) == LAYERS, f"Expected {LAYERS} offline layers, got {len(images)}"
    assert len(paths) == len(images)
    for path, image in zip(paths, images, strict=True):
        assert path.exists(), f"Missing offline layer artifact: {path}"
        assert image.size[0] > 0 and image.size[1] > 0, f"Invalid offline layer size: {image.size}"
    assert metadata_path.exists(), f"Missing offline metadata artifact: {metadata_path}"
