from __future__ import annotations

import time
from pathlib import Path

import pytest
import requests
import torch
from PIL import Image

from tests.conftest import OmniServer, OmniServerParams
from tests.e2e.accuracy.qwen_image_edit_2511 import (
    GUIDANCE_SCALE,
    HEIGHT,
    MODEL_NAME,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    ONLINE_TIMEOUT_SECONDS,
    OUTPUT_FORMAT,
    PROMPT,
    SEED,
    SIZE,
    TRUE_CFG_SCALE,
    WIDTH,
    artifact_paths,
    build_online_image_reference,
    decode_base64_image,
    resolve_image_sources,
    write_json,
)
from tests.utils import hardware_test

SERVER_CASES = [
    pytest.param(
        OmniServerParams(
            model=MODEL_NAME,
            server_args=[
                "--cfg-parallel-size",
                "2",
                "--default-sampling-params",
                '{"0": {"num_inference_steps": 4, "guidance_scale": 1.0}}',
                "--vae-use-tiling",
                "--vae-use-slicing",
            ],
            env_dict={"VLLM_IMAGE_FETCH_TIMEOUT": "60"},
            stage_init_timeout=300,
        ),
        id="qwen_image_edit_2511_cfg_parallel",
    )
]


def generate_online_image(
    *,
    omni_server: OmniServer,
    image_sources: list[str],
    accuracy_artifact_root: Path,
    timeout_seconds: int = ONLINE_TIMEOUT_SECONDS,
) -> Path:
    paths = artifact_paths(accuracy_artifact_root, image_sources)
    url = f"http://{omni_server.host}:{omni_server.port}/v1/images/edits"
    data = [
        ("model", omni_server.model),
        ("prompt", PROMPT),
        ("negative_prompt", NEGATIVE_PROMPT),
        ("size", SIZE),
        ("output_format", OUTPUT_FORMAT),
        ("num_inference_steps", str(NUM_INFERENCE_STEPS)),
        ("guidance_scale", str(GUIDANCE_SCALE)),
        ("true_cfg_scale", str(TRUE_CFG_SCALE)),
        ("seed", str(SEED)),
    ]
    for image_source in image_sources:
        data.append(("url", build_online_image_reference(image_source)))

    start_time = time.perf_counter()
    response = requests.post(
        url,
        data=data,
        headers={"Accept": "application/json"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    output_image = decode_base64_image(payload["data"][0]["b64_json"])
    output_image.save(paths["online"], format=OUTPUT_FORMAT.upper())
    write_json(
        paths["online_metadata"],
        {
            "model": omni_server.model,
            "image_sources": image_sources,
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "size": SIZE,
            "width": output_image.width,
            "height": output_image.height,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "true_cfg_scale": TRUE_CFG_SCALE,
            "seed": SEED,
            "timeout_seconds": timeout_seconds,
            "e2e_latency_seconds": time.perf_counter() - start_time,
            "output_path": str(paths["online"]),
        },
    )
    return paths["online"]


@pytest.mark.advanced_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_qwen_image_edit_2511_online_serving_generates_image(
    omni_server: OmniServer,
    accuracy_artifact_root: Path,
    qwen_image_edit_2511_image_sources: list[str] | None,
) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Qwen-Image-Edit-2511 online accuracy test requires >= 2 CUDA GPUs.")

    image_sources = resolve_image_sources(qwen_image_edit_2511_image_sources)
    online_path = generate_online_image(
        omni_server=omni_server,
        image_sources=image_sources,
        accuracy_artifact_root=accuracy_artifact_root,
    )
    assert online_path.exists(), f"Expected online artifact at {online_path}"

    output_image = Image.open(online_path)
    output_image.load()
    assert output_image.size == (WIDTH, HEIGHT)
