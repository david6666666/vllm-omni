from __future__ import annotations

import contextlib
import json
import math
import mimetypes
import os
import sys
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import requests

VBENCH_DIMENSION_DISPLAY = {
    "subject_consistency": "subject consistency",
    "background_consistency": "background consistency",
    "temporal_flickering": "temporal flickering",
    "motion_smoothness": "motion smoothness",
    "dynamic_degree": "dynamic degree",
    "aesthetic_quality": "aesthetic quality",
    "imaging_quality": "imaging quality",
    "object_class": "object class",
    "multiple_objects": "multiple objects",
    "human_action": "human action",
    "color": "color",
    "spatial_relationship": "spatial relationship",
    "scene": "scene",
    "appearance_style": "appearance style",
    "temporal_style": "temporal style",
    "overall_consistency": "overall consistency",
}

VBENCH_DIMENSION_WEIGHT = {
    "subject consistency": 1.0,
    "background consistency": 1.0,
    "temporal flickering": 1.0,
    "motion smoothness": 1.0,
    "dynamic degree": 0.5,
    "aesthetic quality": 1.0,
    "imaging quality": 1.0,
    "object class": 1.0,
    "multiple objects": 1.0,
    "human action": 1.0,
    "color": 1.0,
    "spatial relationship": 1.0,
    "scene": 1.0,
    "appearance style": 1.0,
    "temporal style": 1.0,
    "overall consistency": 1.0,
}

VBENCH_NORMALIZE = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    "object class": {"Min": 0.0, "Max": 1.0},
    "multiple objects": {"Min": 0.0, "Max": 1.0},
    "human action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222},
    "appearance style": {"Min": 0.0009, "Max": 0.2855},
    "temporal style": {"Min": 0.0, "Max": 0.364},
    "overall consistency": {"Min": 0.0, "Max": 0.364},
}

VBENCH_QUALITY_LIST = [
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "dynamic degree",
    "aesthetic quality",
    "imaging quality",
]
VBENCH_SEMANTIC_LIST = [
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency",
]
VBENCH_TOTAL_WEIGHTS = {"quality": 4.0, "semantic": 1.0}

VBENCH_I2V_DIMENSION_DISPLAY = {
    "camera_motion": "Video-Text Camera Motion",
    "i2v_subject": "Video-Image Subject Consistency",
    "i2v_background": "Video-Image Background Consistency",
    "subject_consistency": "Subject Consistency",
    "background_consistency": "Background Consistency",
    "motion_smoothness": "Motion Smoothness",
    "dynamic_degree": "Dynamic Degree",
    "aesthetic_quality": "Aesthetic Quality",
    "imaging_quality": "Imaging Quality",
}

VBENCH_I2V_DIMENSION_WEIGHT = {
    "Video-Text Camera Motion": 0.1,
    "Video-Image Subject Consistency": 1.0,
    "Video-Image Background Consistency": 1.0,
    "Subject Consistency": 1.0,
    "Background Consistency": 1.0,
    "Motion Smoothness": 1.0,
    "Dynamic Degree": 0.5,
    "Aesthetic Quality": 1.0,
    "Imaging Quality": 1.0,
}

VBENCH_I2V_NORMALIZE = {
    "Video-Text Camera Motion": {"Min": 0.0, "Max": 1.0},
    "Video-Image Subject Consistency": {"Min": 0.1462, "Max": 1.0},
    "Video-Image Background Consistency": {"Min": 0.2615, "Max": 1.0},
    "Subject Consistency": {"Min": 0.1462, "Max": 1.0},
    "Background Consistency": {"Min": 0.2615, "Max": 1.0},
    "Motion Smoothness": {"Min": 0.7060, "Max": 0.9975},
    "Dynamic Degree": {"Min": 0.0, "Max": 1.0},
    "Aesthetic Quality": {"Min": 0.0, "Max": 1.0},
    "Imaging Quality": {"Min": 0.0, "Max": 1.0},
}

VBENCH_I2V_LIST = [
    "Video-Text Camera Motion",
    "Video-Image Subject Consistency",
    "Video-Image Background Consistency",
]
VBENCH_I2V_QUALITY_LIST = [
    "Subject Consistency",
    "Background Consistency",
    "Motion Smoothness",
    "Dynamic Degree",
    "Aesthetic Quality",
    "Imaging Quality",
]
VBENCH_I2V_TOTAL_WEIGHTS = {"quality": 1.0, "i2v": 1.0}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def stringify_form_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_t2v_video_request(
    *,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    num_inference_steps: int,
    guidance_scale: float | None = None,
    guidance_scale_2: float | None = None,
    boundary_ratio: float | None = None,
    flow_shift: float | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
) -> dict[str, str]:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
        "num_inference_steps": num_inference_steps,
    }
    payload.update(
        {
            key: value
            for key, value in {
                "guidance_scale": guidance_scale,
                "guidance_scale_2": guidance_scale_2,
                "boundary_ratio": boundary_ratio,
                "flow_shift": flow_shift,
                "seed": seed,
                "negative_prompt": negative_prompt,
            }.items()
            if value is not None
        }
    )
    return {key: stringify_form_value(value) for key, value in payload.items()}


def build_i2v_video_request(
    *,
    prompt: str,
    input_reference: Path,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    num_inference_steps: int,
    guidance_scale: float | None = None,
    guidance_scale_2: float | None = None,
    boundary_ratio: float | None = None,
    flow_shift: float | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
) -> tuple[dict[str, str], tuple[str, tuple[str, bytes, str]]]:
    form_fields = build_t2v_video_request(
        prompt=prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        guidance_scale_2=guidance_scale_2,
        boundary_ratio=boundary_ratio,
        flow_shift=flow_shift,
        seed=seed,
        negative_prompt=negative_prompt,
    )
    content_type = mimetypes.guess_type(input_reference.name)[0] or "application/octet-stream"
    file_payload = ("input_reference", (input_reference.name, input_reference.read_bytes(), content_type))
    return form_fields, file_payload


class VllmOmniVideoClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "EMPTY",
        timeout: int = 900,
        poll_interval_s: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.poll_interval_s = poll_interval_s

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _video_url(self, suffix: str = "") -> str:
        return f"{self.base_url}/v1/videos{suffix}"

    def create_video_job(
        self,
        *,
        form_fields: dict[str, str],
        input_reference: Path | None = None,
    ) -> dict[str, Any]:
        files = None
        if input_reference is not None:
            _, file_payload = build_i2v_video_request(
                prompt=form_fields["prompt"],
                input_reference=input_reference,
                width=int(form_fields["width"]),
                height=int(form_fields["height"]),
                num_frames=int(form_fields["num_frames"]),
                fps=int(form_fields["fps"]),
                num_inference_steps=int(form_fields["num_inference_steps"]),
                guidance_scale=float(form_fields["guidance_scale"]) if "guidance_scale" in form_fields else None,
                guidance_scale_2=(
                    float(form_fields["guidance_scale_2"]) if "guidance_scale_2" in form_fields else None
                ),
                boundary_ratio=float(form_fields["boundary_ratio"]) if "boundary_ratio" in form_fields else None,
                flow_shift=float(form_fields["flow_shift"]) if "flow_shift" in form_fields else None,
                seed=int(form_fields["seed"]) if "seed" in form_fields else None,
                negative_prompt=form_fields.get("negative_prompt"),
            )
            files = {file_payload[0]: file_payload[1]}

        response = requests.post(
            self._video_url(),
            data=form_fields,
            files=files,
            headers=self._headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def wait_for_video_completion(self, video_id: str) -> dict[str, Any]:
        deadline = time.time() + self.timeout
        last_payload: dict[str, Any] | None = None
        while time.time() < deadline:
            response = requests.get(
                self._video_url(f"/{video_id}"),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            last_payload = response.json()
            status = last_payload.get("status")
            if status == "completed":
                return last_payload
            if status == "failed":
                raise RuntimeError(f"Video job {video_id} failed: {last_payload}")
            time.sleep(self.poll_interval_s)
        raise TimeoutError(f"Timed out waiting for video job {video_id}. Last payload: {last_payload}")

    def download_video(self, video_id: str, output_path: Path) -> Path:
        response = requests.get(
            self._video_url(f"/{video_id}/content"),
            headers=self._headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        ensure_dir(output_path.parent)
        output_path.write_bytes(response.content)
        return output_path

    def delete_video_job(self, video_id: str) -> None:
        with contextlib.suppress(Exception):
            requests.delete(
                self._video_url(f"/{video_id}"),
                headers=self._headers,
                timeout=self.timeout,
            )

    def generate_video(
        self,
        *,
        output_path: Path,
        form_fields: dict[str, str],
        input_reference: Path | None = None,
    ) -> dict[str, Any]:
        created = self.create_video_job(form_fields=form_fields, input_reference=input_reference)
        video_id = created["id"]
        try:
            completed = self.wait_for_video_completion(video_id)
            self.download_video(video_id, output_path)
            return completed
        finally:
            self.delete_video_job(video_id)


def _repo_root_from_env() -> Path | None:
    value = os.environ.get("VBENCH_REPO_ROOT")
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    return path if path.exists() else None


@contextlib.contextmanager
def vbench_import_context(vbench_root: Path | str | None = None) -> Iterator[None]:
    added_path: str | None = None
    replaced_modules: dict[str, Any] = {}
    if vbench_root is not None:
        root = Path(vbench_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"VBench repo root does not exist: {root}")
        added_path = str(root)
    else:
        env_root = _repo_root_from_env()
        if env_root is not None:
            added_path = str(env_root)

    if added_path is not None and added_path not in sys.path:
        sys.path.insert(0, added_path)
        for module_name in ("vbench", "vbench2_beta_i2v"):
            if module_name in sys.modules:
                replaced_modules[module_name] = sys.modules.pop(module_name)
        try:
            yield
        finally:
            for module_name in ("vbench", "vbench2_beta_i2v"):
                sys.modules.pop(module_name, None)
            sys.modules.update(replaced_modules)
            with contextlib.suppress(ValueError):
                sys.path.remove(added_path)
    else:
        yield


def _normalize_score(raw_score: float, *, minimum: float, maximum: float) -> float:
    if not math.isfinite(raw_score):
        raise ValueError(f"Expected a finite score, got {raw_score!r}")
    if maximum <= minimum:
        raise ValueError(f"Invalid normalization range: min={minimum}, max={maximum}")
    return (raw_score - minimum) / (maximum - minimum)


def _weighted_average(scores: dict[str, float], weights: dict[str, float], keys: Sequence[str]) -> float | None:
    present_keys = [key for key in keys if key in scores]
    if not present_keys:
        return None
    weight_sum = sum(weights[key] for key in present_keys)
    if weight_sum == 0:
        return None
    return sum(scores[key] * weights[key] for key in present_keys) / weight_sum


def _extract_unique_sample_count(raw_results: dict[str, Any]) -> int:
    unique_paths: set[str] = set()
    for result in raw_results.values():
        if not isinstance(result, list) or len(result) < 2 or not isinstance(result[1], list):
            continue
        for item in result[1]:
            if isinstance(item, dict) and item.get("video_path"):
                unique_paths.add(str(item["video_path"]))
    return len(unique_paths)


def _summarize_scores(
    *,
    raw_results: dict[str, Any],
    dimension_display: dict[str, str],
    normalize_config: dict[str, dict[str, float]],
    weight_config: dict[str, float],
    primary_group_keys: Sequence[str],
    secondary_group_keys: Sequence[str],
    primary_group_name: str,
    secondary_group_name: str,
    total_weight_config: dict[str, float],
) -> dict[str, Any]:
    raw_scores: dict[str, float] = {}
    normalized_scores: dict[str, float] = {}
    weighted_scores: dict[str, float] = {}
    evaluated_dimensions: list[str] = []

    for dimension_name, display_name in dimension_display.items():
        result = raw_results.get(dimension_name)
        if not isinstance(result, list) or not result:
            continue
        raw_score = float(result[0])
        if not math.isfinite(raw_score):
            continue
        raw_scores[display_name] = raw_score
        normalized_score = _normalize_score(
            raw_score,
            minimum=normalize_config[display_name]["Min"],
            maximum=normalize_config[display_name]["Max"],
        )
        normalized_scores[display_name] = normalized_score
        weighted_scores[display_name] = normalized_score * weight_config[display_name]
        evaluated_dimensions.append(dimension_name)

    primary_score = _weighted_average(normalized_scores, weight_config, primary_group_keys)
    secondary_score = _weighted_average(normalized_scores, weight_config, secondary_group_keys)

    missing_dimensions = [dimension for dimension in dimension_display if dimension not in evaluated_dimensions]
    partial_benchmark = bool(missing_dimensions)

    summary: dict[str, Any] = {
        "partial_benchmark": partial_benchmark,
        "evaluated_dimensions": evaluated_dimensions,
        "missing_dimensions": missing_dimensions,
        "sample_count": _extract_unique_sample_count(raw_results),
        "raw_scores": raw_scores,
        "normalized_scores": normalized_scores,
        "weighted_scores": weighted_scores,
    }
    if primary_score is not None:
        summary[primary_group_name] = primary_score
    if secondary_score is not None:
        summary[secondary_group_name] = secondary_score

    if not partial_benchmark and primary_score is not None and secondary_score is not None:
        summary["total_score"] = (
            primary_score * total_weight_config["quality"] + secondary_score * total_weight_config["semantic"]
        ) / (total_weight_config["quality"] + total_weight_config["semantic"])

    return summary


def summarize_vbench_results(raw_results: dict[str, Any]) -> dict[str, Any]:
    return _summarize_scores(
        raw_results=raw_results,
        dimension_display=VBENCH_DIMENSION_DISPLAY,
        normalize_config=VBENCH_NORMALIZE,
        weight_config=VBENCH_DIMENSION_WEIGHT,
        primary_group_keys=VBENCH_QUALITY_LIST,
        secondary_group_keys=VBENCH_SEMANTIC_LIST,
        primary_group_name="quality_score",
        secondary_group_name="semantic_score",
        total_weight_config=VBENCH_TOTAL_WEIGHTS,
    )


def summarize_vbench_i2v_results(raw_results: dict[str, Any]) -> dict[str, Any]:
    return _summarize_scores(
        raw_results=raw_results,
        dimension_display=VBENCH_I2V_DIMENSION_DISPLAY,
        normalize_config=VBENCH_I2V_NORMALIZE,
        weight_config=VBENCH_I2V_DIMENSION_WEIGHT,
        primary_group_keys=VBENCH_I2V_QUALITY_LIST,
        secondary_group_keys=VBENCH_I2V_LIST,
        primary_group_name="quality_score",
        secondary_group_name="i2v_score",
        total_weight_config={"quality": VBENCH_I2V_TOTAL_WEIGHTS["quality"], "semantic": VBENCH_I2V_TOTAL_WEIGHTS["i2v"]},
    )


def _entry_matches_dimension(entry: dict[str, Any], dimension: str) -> bool:
    return dimension in entry.get("dimension", [])


def select_balanced_vbench_entries(
    full_info: Sequence[dict[str, Any]],
    *,
    dimensions: Sequence[str],
    prompts_per_dimension: int | None,
) -> list[dict[str, Any]]:
    if prompts_per_dimension is None:
        selected: list[dict[str, Any]] = []
        seen_prompts: set[str] = set()
        for entry in full_info:
            prompt = str(entry["prompt_en"])
            if prompt in seen_prompts:
                continue
            if set(entry.get("dimension", [])) & set(dimensions):
                selected.append(dict(entry))
                seen_prompts.add(prompt)
        return selected

    selected: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    for dimension in dimensions:
        chosen = 0
        for entry in full_info:
            prompt = str(entry["prompt_en"])
            if prompt in seen_prompts or not _entry_matches_dimension(entry, dimension):
                continue
            selected.append(dict(entry))
            seen_prompts.add(prompt)
            chosen += 1
            if chosen >= prompts_per_dimension:
                break
    return selected


def _i2v_case_key(entry: dict[str, Any]) -> tuple[str, str]:
    image_ref = str(entry.get("image_name") or entry.get("custom_image_path") or "")
    return str(entry["prompt_en"]), image_ref


def select_balanced_vbench_i2v_entries(
    full_info: Sequence[dict[str, Any]],
    *,
    dimensions: Sequence[str],
    cases_per_dimension: int | None,
) -> list[dict[str, Any]]:
    if cases_per_dimension is None:
        selected: list[dict[str, Any]] = []
        seen_cases: set[tuple[str, str]] = set()
        for entry in full_info:
            key = _i2v_case_key(entry)
            if key in seen_cases:
                continue
            if set(entry.get("dimension", [])) & set(dimensions):
                selected.append(dict(entry))
                seen_cases.add(key)
        return selected

    selected: list[dict[str, Any]] = []
    seen_cases: set[tuple[str, str]] = set()
    for dimension in dimensions:
        chosen = 0
        for entry in full_info:
            key = _i2v_case_key(entry)
            if key in seen_cases or not _entry_matches_dimension(entry, dimension):
                continue
            selected.append(dict(entry))
            seen_cases.add(key)
            chosen += 1
            if chosen >= cases_per_dimension:
                break
    return selected


def resolve_i2v_reference_path(dataset_root: Path, image_name: str, *, resolution: str) -> Path:
    candidates = [
        dataset_root / "crop" / resolution / image_name,
        dataset_root / resolution / image_name,
        dataset_root / "origin" / image_name,
        dataset_root / image_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve image '{image_name}' under dataset root {dataset_root} for resolution {resolution}."
    )


def build_vbench_generation_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_dimension: dict[str, int] = defaultdict(int)
    prompts: set[str] = set()
    for record in records:
        prompts.add(str(record["prompt"]))
        for dimension in record.get("dimensions", []):
            by_dimension[str(dimension)] += 1
    return {
        "count": len(records),
        "prompt_count": len(prompts),
        "by_dimension": dict(sorted(by_dimension.items())),
    }
