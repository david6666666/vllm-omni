from __future__ import annotations

import time
from pathlib import Path

import pytest
import requests
import torch

from tests.conftest import OmniServer, OmniServerParams
from tests.e2e.accuracy.qwen_image_edit_2511.common import (
    GUIDANCE_SCALE,
    MODEL_NAME,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    ONLINE_TIMEOUT_SECONDS,
    OUTPUT_FORMAT,
    PROMPT,
    SEED,
    SIZE,
    TRUE_CFG_SCALE,
    artifact_paths,
    resolve_configured_image_sources,
)
from tests.e2e.accuracy.utils import (
    build_online_image_reference,
    decode_base64_image,
    load_rgb_image,
    write_json,
)
from tests.utils import hardware_test

SERVER_CASES = [
    pytest.param(
        OmniServerParams(
            model=MODEL_NAME,
            server_args=[
                "--default-sampling-params",
                '{"0": {"num_inference_steps": 4, "guidance_scale": 1.0}}',
                "--vae-use-tiling",
                "--vae-use-slicing",
            ],
            env_dict={"VLLM_IMAGE_FETCH_TIMEOUT": "60"},
            stage_init_timeout=300,
        ),
        id="qwen_image_edit_2511_single_gpu",
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
    with requests.Session() as session:
        # Bypass shell-configured HTTP proxies for loopback traffic.
        session.trust_env = False
        response = session.post(
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
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_qwen_image_edit_2511_online_serving_generates_image(
    omni_server: OmniServer,
    accuracy_artifact_root: Path,
    qwen_image_edit_2511_image_sources: list[str] | None,
) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("Qwen-Image-Edit-2511 online accuracy test requires >= 1 CUDA GPU.")

    image_sources = resolve_configured_image_sources(qwen_image_edit_2511_image_sources)
    paths = artifact_paths(accuracy_artifact_root, image_sources)
    online_path = generate_online_image(
        omni_server=omni_server,
        image_sources=image_sources,
        accuracy_artifact_root=accuracy_artifact_root,
    )
    assert online_path.exists(), f"Expected online artifact at {online_path}"
    assert paths["online_metadata"].exists(), f"Expected online metadata at {paths['online_metadata']}"
    output_image = load_rgb_image(online_path)
    assert output_image.size[0] > 0 and output_image.size[1] > 0
