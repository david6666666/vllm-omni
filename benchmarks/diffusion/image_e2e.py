# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serial PNG image API benchmark with saved first-request and timed outputs.

Requests are JSONL OpenAI image-generation bodies. Time covers POST through
the complete JSON/base64 response download; parsing and saving are outside the
timer. The caller owns server startup, source/environment pins and exclusivity.
"""

import argparse
import binascii
import hashlib
import io
import json
import statistics
import time
import urllib.request
from pathlib import Path

from PIL import Image


def request_image(endpoint: str, body: dict, destination: Path) -> dict:
    request = urllib.request.Request(endpoint, json.dumps(body).encode(), {"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        if response.status != 200:
            raise ValueError("Image generation did not complete successfully")
        payload = response.read()
    latency_ms = (time.perf_counter() - started) * 1000
    images = json.loads(payload)["data"]
    if len(images) != 1:
        raise ValueError("Expected one complete image per request")
    image = binascii.a2b_base64(images[0]["b64_json"])
    if not image.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("This benchmark requires PNG output")
    with Image.open(io.BytesIO(image)) as decoded:
        decoded.verify()
    destination.write_bytes(image)
    return {"latency_ms": latency_ms, "path": str(destination), "sha256": hashlib.sha256(image).hexdigest()}


def run(args: argparse.Namespace) -> dict:
    requests = [json.loads(line) for line in args.requests.read_text().splitlines() if line.strip()]
    if not requests or any(not isinstance(body, dict) for body in requests):
        raise ValueError("At least one JSON object request is required")
    if any(isinstance(body.get("seed"), bool) or not isinstance(body.get("seed"), int) for body in requests):
        raise ValueError("Each request must contain an integer seed")
    if args.output.exists():
        raise FileExistsError("Preserve the previous trial result; use a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    media = args.output.parent / "media"
    media.mkdir()
    endpoint = args.base_url.rstrip("/") + "/v1/images/generations"
    record = {
        "status": "running",
        "requests": requests,
        "request_file_sha256": hashlib.sha256(args.requests.read_bytes()).hexdigest(),
        "timing": "POST through full JSON/base64 download; parsing and file writes outside timer",
        "warmup_rounds": args.warmup_rounds,
        "warmup_seed_offset": args.warmup_seed_offset,
        "warmup_samples": [],
        "samples_ms": [],
        "groups_ms": [],
        "artifacts": [],
    }
    try:
        for round_index in range(args.warmup_rounds):
            for index, body in enumerate(requests):
                warm = {**body, "seed": body["seed"] + args.warmup_seed_offset + 1000 * round_index}
                path = media / "warmup.png"
                sample = request_image(endpoint, warm, path)
                record["warmup_samples"].append(
                    {"prompt_index": index, "seed": warm["seed"], "latency_ms": sample["latency_ms"]}
                )
                if "first_request" not in record:
                    first = media / "first-request.png"
                    path.rename(first)
                    record["first_request"] = {
                        **sample,
                        "path": str(first),
                        "seed": warm["seed"],
                        "prompt_index": index,
                    }
                else:
                    path.unlink()
        for group in range(1, args.repetitions + 1):
            values = []
            for index, body in enumerate(requests):
                path = media / f"group-{group:03d}-prompt-{index:03d}-seed-{body['seed']}.png"
                sample = request_image(endpoint, body, path)
                identity = {"group": group, "prompt_index": index, "seed": body["seed"]}
                record["samples_ms"].append({**identity, "latency_ms": sample["latency_ms"]})
                record["artifacts"].append({**identity, "path": sample["path"], "sha256": sample["sha256"]})
                values.append(sample["latency_ms"])
            record["groups_ms"].append(statistics.mean(values))
        record.update(status="completed", metrics={"latency_ms": statistics.mean(record["groups_ms"])})
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8098")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--warmup-seed-offset", type=int, default=10000)
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmup_rounds < 1:
        parser.error("repetitions and warmup-rounds must be positive")
    result = run(args)
    print(json.dumps({"status": result["status"], "metrics": result["metrics"]}))


if __name__ == "__main__":
    main()
