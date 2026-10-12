# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import io
import json
from unittest.mock import Mock

import pytest
from PIL import Image

from benchmarks.diffusion import image_e2e

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_warmup_uses_fresh_seeds_and_first_request_is_preserved(tmp_path, monkeypatch):
    requests = tmp_path / "requests.jsonl"
    requests.write_text('{"prompt":"A","seed":42}\n{"prompt":"B","seed":43}\n')
    seen = []

    def request(endpoint, body, destination):
        seen.append(body)
        destination.write_bytes(b"saved image")
        return {"latency_ms": 100, "path": str(destination), "sha256": "fixture"}

    monkeypatch.setattr(image_e2e, "request_image", request)
    args = argparse.Namespace(
        requests=requests,
        output=tmp_path / "trial" / "result.json",
        base_url="http://localhost:8098",
        warmup_rounds=2,
        warmup_seed_offset=10000,
        repetitions=2,
    )
    result = image_e2e.run(args)
    assert [x["seed"] for x in seen] == [10042, 10043, 11042, 11043, 42, 43, 42, 43]
    assert result["first_request"]["seed"] == 10042
    assert result["groups_ms"] == [100, 100]
    assert len(result["artifacts"]) == 4
    assert (args.output.parent / "media" / "first-request.png").is_file()
    with pytest.raises(FileExistsError):
        image_e2e.run(args)


def test_download_finishes_before_decoding_and_saving(tmp_path, monkeypatch):
    import binascii

    buffer = io.BytesIO()
    Image.new("RGBA", (4, 3)).save(buffer, format="PNG")
    response = Mock(status=200)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = json.dumps(
        {"data": [{"b64_json": binascii.b2a_base64(buffer.getvalue()).decode()}]}
    ).encode()
    calls = []
    monkeypatch.setattr(image_e2e.urllib.request, "urlopen", lambda *args, **kwargs: response)

    def clock():
        calls.append("clock")
        return len(calls) / 4

    monkeypatch.setattr(image_e2e.time, "perf_counter", clock)
    destination = tmp_path / "output.png"
    result = image_e2e.request_image("http://localhost/v1/images/generations", {"seed": 42}, destination)
    assert result["latency_ms"] == 250
    assert calls == ["clock", "clock"]
    assert destination.read_bytes() == buffer.getvalue()
    response.read.return_value = b'{"data":[]}'
    with pytest.raises(ValueError, match="complete image"):
        image_e2e.request_image("http://localhost/v1/images/generations", {}, destination)
