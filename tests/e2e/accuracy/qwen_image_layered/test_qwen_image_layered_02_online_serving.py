from __future__ import annotations

import base64
import json
import mimetypes
import shlex
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
import requests
import torch
from PIL import Image

from tests.conftest import OmniServerParams
from tests.e2e.accuracy.qwen_image_layered.test_qwen_image_layered_01_diffusers_baseline import (
    CFG_NORMALIZE,
    LAYERS,
    MODEL_NAME,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    RESOLUTION,
    RESULT_ROOT,
    SEED,
    TRUE_CFG_SCALE,
    artifact_dir,
    is_remote_image_source,
    resolve_image_source,
    validate_image_source,
)
from tests.utils import hardware_test

SERVER_CASES = [
    pytest.param(
        OmniServerParams(
            model=MODEL_NAME,
            server_args=["--num-gpus", "1"],
            use_omni=True,
        ),
        id="qwen_image_layered_1gpu",
    )
]
REQUEST_TIMEOUT_SECONDS = 1200
PROMPT = ""
OUTPUT_FORMAT = "png"


def build_vllm_serve_command(*, port: int) -> list[str]:
    return [
        "vllm",
        "serve",
        MODEL_NAME,
        "--omni",
        "--port",
        str(port),
        "--num-gpus",
        "1",
    ]


def build_curl_command(*, server_url: str, image_source: str, output_path: Path) -> str:
    form_fields = [
        "-F",
        f"model={MODEL_NAME}",
        "-F",
        "prompt=",
        "-F",
        "n=1",
        "-F",
        "size=auto",
        "-F",
        "response_format=b64_json",
        "-F",
        f"num_inference_steps={NUM_INFERENCE_STEPS}",
        "-F",
        f"true_cfg_scale={TRUE_CFG_SCALE}",
        "-F",
        "negative_prompt= ",
        "-F",
        f"seed={SEED}",
        "-F",
        f"layers={LAYERS}",
        "-F",
        f"resolution={RESOLUTION}",
        "-F",
        f"cfg_normalize={str(CFG_NORMALIZE).lower()}",
        "-F",
        "use_en_prompt=false",
        "-F",
        f"output_format={OUTPUT_FORMAT}",
    ]
    if is_remote_image_source(image_source) or image_source.startswith("data:image"):
        form_fields.extend(["-F", f"url={image_source}"])
    else:
        form_fields.extend(["-F", f"image=@{image_source}"])

    quoted_fields = " ".join(shlex.quote(item) for item in form_fields)
    output_dir = shlex.quote(str(output_path.parent))
    return (
        f"curl -sS {shlex.quote(server_url.rstrip('/') + '/v1/images/edits')} {quoted_fields} "
        f"| jq -r '.data[].b64_json' | nl -ba -w1 -s: | while IFS=: read -r idx b64; do "
        f"mkdir -p {output_dir}; echo \"$b64\" | base64 -d > {output_dir}/layer_${{idx}}.png; done"
    )


def online_layer_paths(image_source: str) -> list[Path]:
    base_dir = artifact_dir(image_source) / "online"
    return [base_dir / f"layer_{index}.png" for index in range(LAYERS)]


def online_response_path(image_source: str) -> Path:
    return artifact_dir(image_source) / "online" / "response.json"


def load_online_layers(image_source: str) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in online_layer_paths(image_source):
        image = Image.open(path)
        image.load()
        images.append(image.convert("RGBA"))
    return images


def _multipart_request_components(image_source: str) -> tuple[dict[str, str], list[tuple[str, tuple[str, bytes, str]]]]:
    data = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "n": "1",
        "size": "auto",
        "response_format": "b64_json",
        "num_inference_steps": str(NUM_INFERENCE_STEPS),
        "true_cfg_scale": str(TRUE_CFG_SCALE),
        "negative_prompt": NEGATIVE_PROMPT,
        "seed": str(SEED),
        "layers": str(LAYERS),
        "resolution": str(RESOLUTION),
        "cfg_normalize": str(CFG_NORMALIZE).lower(),
        "use_en_prompt": "false",
        "output_format": OUTPUT_FORMAT,
    }
    files: list[tuple[str, tuple[str, bytes, str]]] = []

    if is_remote_image_source(image_source) or image_source.startswith("data:image"):
        data["url"] = image_source
        return data, files

    image_path = Path(image_source)
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    files.append(("image", (image_path.name, image_path.read_bytes(), mime_type)))
    return data, files


def _decode_images(response_payload: dict[str, Any]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for item in response_payload.get("data", []):
        encoded = item.get("b64_json")
        if not encoded:
            continue
        image = Image.open(BytesIO(base64.b64decode(encoded)))
        image.load()
        images.append(image.convert("RGBA"))
    return images


def _write_online_response(image_source: str, response_payload: dict[str, Any]) -> Path:
    response_path = online_response_path(image_source)
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(json.dumps(response_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return response_path


def generate_online_layers(*, omni_server, image_source: str) -> list[Image.Image]:
    request_data, request_files = _multipart_request_components(image_source)
    url = f"http://{omni_server.host}:{omni_server.port}/v1/images/edits"
    response = requests.post(
        url,
        data=request_data,
        files=request_files,
        headers={"Accept": "application/json", "Authorization": "Bearer EMPTY"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    response_payload = response.json()
    images = _decode_images(response_payload)
    if len(images) != LAYERS:
        raise AssertionError(f"Expected {LAYERS} online layers, got {len(images)}. payload={response_payload}")

    for path, image in zip(online_layer_paths(image_source), images, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)

    _write_online_response(image_source, response_payload)
    return images


def test_build_curl_command_uses_url_field_for_remote_source(tmp_path: Path) -> None:
    command = build_curl_command(
        server_url="http://127.0.0.1:8093",
        image_source="https://example.com/input.png",
        output_path=tmp_path / "output.png",
    )

    assert "/v1/images/edits" in command
    assert "url=https://example.com/input.png" in command
    assert "image=@" not in command


def test_multipart_request_components_upload_local_file(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGBA", (8, 8), color=(255, 0, 0, 255)).save(image_path)

    data, files = _multipart_request_components(str(image_path))

    assert "url" not in data
    assert files[0][0] == "image"
    assert files[0][1][0] == "input.png"


@pytest.mark.advanced_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_qwen_image_layered_online_serving_generates_layers(
    omni_server,
    qwen_image_layered_image_source: str | None,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Qwen-Image-Layered online serving test requires CUDA.")

    image_source = resolve_image_source(qwen_image_layered_image_source)
    validate_image_source(image_source)

    serve_command = build_vllm_serve_command(port=omni_server.port)
    curl_command = build_curl_command(
        server_url=f"http://{omni_server.host}:{omni_server.port}",
        image_source=image_source,
        output_path=RESULT_ROOT / "curl-output" / "layer_0.png",
    )
    print("qwen_image_layered serve command:")
    print(" ".join(serve_command))
    print("qwen_image_layered curl command:")
    print(curl_command)

    images = generate_online_layers(omni_server=omni_server, image_source=image_source)
    response_path = online_response_path(image_source)

    assert len(images) == LAYERS
    for path, image in zip(online_layer_paths(image_source), images, strict=True):
        assert path.exists(), f"Missing online layer artifact: {path}"
        assert image.size[0] > 0 and image.size[1] > 0, f"Invalid online layer size: {image.size}"
    assert response_path.exists(), f"Missing online response artifact: {response_path}"
