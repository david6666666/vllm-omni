from __future__ import annotations

import base64
import io
import json
import os
import re
from hashlib import sha1
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image


def resolve_image_sources(configured: Sequence[str] | None, default_sources: Sequence[str]) -> list[str]:
    if configured:
        return [str(item) for item in configured]
    return [str(item) for item in default_sources]


def is_remote_image_source(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def load_input_image(source: str) -> Image.Image:
    if source.startswith("data:image"):
        _, encoded = source.split(",", 1)
        image = Image.open(io.BytesIO(base64.b64decode(encoded)))
        image.load()
        return image.convert("RGB")

    source_path = Path(source)
    if source_path.exists():
        image = Image.open(source_path)
        image.load()
        return image.convert("RGB")

    response = requests.get(source, timeout=60)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content))
    image.load()
    return image.convert("RGB")


def build_online_image_reference(source: str) -> str:
    if is_remote_image_source(source) or source.startswith("data:image"):
        return source

    source_path = Path(source)
    mime_type = guess_type(source_path.name)[0] or "image/png"
    encoded = base64.b64encode(source_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def decode_base64_image(encoded: str) -> Image.Image:
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    image.load()
    return image.convert("RGB")


def infer_artifact_stem(image_sources: Sequence[str]) -> str:
    labels: list[str] = []
    for source in image_sources:
        if is_remote_image_source(source):
            labels.append(Path(urlparse(source).path).stem or "remote")
        elif source.startswith("data:image"):
            labels.append("data")
        else:
            labels.append(Path(source).stem or "local")

    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(labels) or "image")
    digest = sha1("||".join(image_sources).encode("utf-8")).hexdigest()[:8]
    return f"{safe_label[:80]}_{digest}"


def build_image_artifact_paths(
    accuracy_artifact_root: Path,
    suite_name: str,
    image_sources: Sequence[str],
) -> dict[str, Path]:
    root = accuracy_artifact_root / suite_name
    root.mkdir(parents=True, exist_ok=True)
    stem = infer_artifact_stem(image_sources)
    return {
        "root": root,
        "online": root / f"{stem}_online.png",
        "online_metadata": root / f"{stem}_online_metadata.json",
        "offline": root / f"{stem}_offline.png",
        "offline_metadata": root / f"{stem}_offline_metadata.json",
        "diff": root / f"{stem}_diff.png",
        "compare_summary": root / f"{stem}_compare_summary.json",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_pythonpath_env(repo_root: Path, *extra_paths: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(repo_root), *(str(path) for path in extra_paths)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return env


def load_rgb_image(path: Path) -> Image.Image:
    image = Image.open(path)
    image.load()
    return image.convert("RGB")


def compute_image_diff_metrics(a: Image.Image, b: Image.Image) -> dict[str, float]:
    ta = np.asarray(a.convert("RGB"), dtype=np.float32) / 255.0
    tb = np.asarray(b.convert("RGB"), dtype=np.float32) / 255.0
    assert ta.shape == tb.shape, f"Image shapes differ: {ta.shape} vs {tb.shape}"
    abs_diff = np.abs(ta - tb)
    return {
        "mean_abs_diff": float(abs_diff.mean()),
        "p99_abs_diff": float(np.quantile(abs_diff.reshape(-1), 0.99)),
        "max_abs_diff": float(abs_diff.max()),
        "exact_match_ratio": float(np.mean(abs_diff == 0.0)),
    }


def build_abs_diff_image(a: Image.Image, b: Image.Image) -> Image.Image:
    a_uint8 = np.asarray(a.convert("RGB"), dtype=np.int16)
    b_uint8 = np.asarray(b.convert("RGB"), dtype=np.int16)
    diff = np.abs(a_uint8 - b_uint8).clip(0, 255).astype(np.uint8)
    return Image.fromarray(diff, mode="RGB")
