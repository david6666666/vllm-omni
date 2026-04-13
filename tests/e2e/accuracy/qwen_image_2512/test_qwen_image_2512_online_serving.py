from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import requests
import torch
from PIL import Image

from benchmarks.accuracy.common import decode_openai_image_response_item, ensure_dir, write_json
from tests.conftest import OmniServer, OmniServerParams
from tests.utils import hardware_test

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL_NAME = "Qwen/Qwen-Image-2512"
PROMPT = (
    "A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes, "
    "expressive and lively, with a cheerful or subtly smiling expression. Her naturally wavy long "
    "hair is either loose or tied in twin ponytails. She has fair skin and light makeup accentuating "
    "her youthful freshness. She wears a modern, cute dress or relaxed outfit in bright, soft colors "
    "with lightweight fabric and a minimalist cut. She stands indoors at an anime convention, "
    "surrounded by banners, posters, or stalls. Lighting is typical indoor illumination with no staged "
    "lighting, and the image resembles a casual iPhone snapshot: unpretentious composition, yet "
    "brimming with vivid, fresh, youthful charm."
)
NEGATIVE_PROMPT = (
    "low resolution, low quality, deformed limbs, malformed fingers, oversaturated image, waxy look, "
    "face without details, overly smooth skin, obvious AI artifacts, chaotic composition, blurry text, "
    "distorted text"
)
WIDTH = 1664
HEIGHT = 928
NUM_INFERENCE_STEPS = 50
TRUE_CFG_SCALE = 4.0
SEED = 42
REQUEST_TIMEOUT_SECONDS = 60 * 60
DEFAULT_SAMPLING_PARAMS = json.dumps({"0": {"num_inference_steps": 4, "guidance_scale": 1.0}})
RESULT_ROOT = Path(__file__).parent / "result"

SERVER_CASES = [
    pytest.param(
        OmniServerParams(
            model=MODEL_NAME,
            server_args=["--default-sampling-params", DEFAULT_SAMPLING_PARAMS],
            init_timeout=1800,
        ),
        id="images_api_default_sampling",
    )
]


def _sanitize_case_id(raw_case_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in raw_case_id)


def _resolve_case_id(request: pytest.FixtureRequest) -> str:
    callspec = getattr(request.node, "callspec", None)
    if callspec is not None:
        return _sanitize_case_id(callspec.id)
    return "default"


def _build_request_payload(model: str = MODEL_NAME) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "n": 1,
        "size": f"{WIDTH}x{HEIGHT}",
        "response_format": "b64_json",
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "seed": SEED,
    }


def build_curl_command(base_url: str) -> str:
    payload = json.dumps(_build_request_payload(), ensure_ascii=False)
    return (
        f"curl -X POST {base_url.rstrip('/')}/v1/images/generations "
        f"-H \"Content-Type: application/json\" "
        f"-H \"Authorization: Bearer EMPTY\" "
        f"-d '{payload}' | jq -r '.data[0].b64_json' | base64 -d > qwen_image_2512_online.png"
    )


def _artifact_paths(case_id: str) -> tuple[Path, Path]:
    ensure_dir(RESULT_ROOT)
    return (
        RESULT_ROOT / f"qwen_image_2512_online_{case_id}.png",
        RESULT_ROOT / f"qwen_image_2512_online_{case_id}.json",
    )


def generate_online_image(
    omni_server: OmniServer,
    *,
    case_id: str,
) -> tuple[Path, Path, Image.Image, dict[str, Any]]:
    output_path, metadata_path = _artifact_paths(case_id)
    response = requests.post(
        f"http://{omni_server.host}:{omni_server.port}/v1/images/generations",
        json=_build_request_payload(model=omni_server.model),
        headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    image = decode_openai_image_response_item(payload["data"][0], timeout=REQUEST_TIMEOUT_SECONDS)
    image.save(output_path)

    write_json(
        metadata_path,
        {
            "model": omni_server.model,
            "server_host": omni_server.host,
            "server_port": omni_server.port,
            "server_args": omni_server.serve_args,
            "request_payload": _build_request_payload(model=omni_server.model),
            "response_keys": sorted(payload["data"][0].keys()),
            "output_path": str(output_path),
            "curl_command": build_curl_command(f"http://{omni_server.host}:{omni_server.port}"),
        },
    )
    return output_path, metadata_path, image, payload


@pytest.mark.advanced_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_qwen_image_2512_online_serving_generates_image(
    omni_server: OmniServer,
    request: pytest.FixtureRequest,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Qwen-Image-2512 online serving test requires CUDA.")

    _, metadata_path, image, _ = generate_online_image(omni_server, case_id=_resolve_case_id(request))

    assert image.size == (WIDTH, HEIGHT)
    assert metadata_path.exists()
