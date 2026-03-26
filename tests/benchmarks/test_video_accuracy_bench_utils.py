# ruff: noqa: E402, I001
import math
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.accuracy.common import (
    build_i2v_video_request,
    build_t2v_video_request,
    resolve_i2v_reference_path,
    select_balanced_vbench_entries,
    select_balanced_vbench_i2v_entries,
    summarize_vbench_i2v_results,
    summarize_vbench_results,
    vbench_import_context,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_build_t2v_video_request_omits_input_reference():
    payload = build_t2v_video_request(
        prompt="A lighthouse in fog.",
        width=640,
        height=480,
        num_frames=5,
        fps=8,
        num_inference_steps=4,
        guidance_scale=1.0,
        seed=7,
    )

    assert payload["prompt"] == "A lighthouse in fog."
    assert payload["width"] == "640"
    assert payload["height"] == "480"
    assert payload["num_frames"] == "5"
    assert payload["fps"] == "8"
    assert "input_reference" not in payload


def test_build_i2v_video_request_attaches_input_reference(tmp_path: Path):
    image_path = tmp_path / "ref.png"
    Image.new("RGB", (4, 3), color="white").save(image_path)

    form_fields, file_payload = build_i2v_video_request(
        prompt="A fox runs through snow.",
        input_reference=image_path,
        width=640,
        height=480,
        num_frames=5,
        fps=8,
        num_inference_steps=4,
    )

    assert form_fields["prompt"] == "A fox runs through snow."
    assert form_fields["width"] == "640"
    assert file_payload is not None
    assert file_payload[0] == "input_reference"
    assert file_payload[1][0] == "ref.png"
    assert file_payload[1][2] == "image/png"
    assert file_payload[1][1]


def test_select_balanced_vbench_entries_limits_each_dimension_with_unique_prompts():
    full_info = [
        {"prompt_en": "prompt_a", "dimension": ["subject_consistency", "motion_smoothness"]},
        {"prompt_en": "prompt_b", "dimension": ["subject_consistency"]},
        {"prompt_en": "prompt_c", "dimension": ["motion_smoothness"]},
        {"prompt_en": "prompt_d", "dimension": ["aesthetic_quality"]},
    ]

    selected = select_balanced_vbench_entries(
        full_info,
        dimensions=["subject_consistency", "motion_smoothness", "aesthetic_quality"],
        prompts_per_dimension=1,
    )

    assert [item["prompt_en"] for item in selected] == ["prompt_a", "prompt_c", "prompt_d"]


def test_select_balanced_vbench_i2v_entries_deduplicates_by_prompt_and_image():
    full_info = [
        {"prompt_en": "p1", "image_name": "a.png", "dimension": ["i2v_subject"]},
        {"prompt_en": "p1", "image_name": "a.png", "dimension": ["subject_consistency"]},
        {"prompt_en": "p2", "image_name": "b.png", "dimension": ["camera_motion"]},
        {"prompt_en": "p3", "image_name": "c.png", "dimension": ["camera_motion"]},
    ]

    selected = select_balanced_vbench_i2v_entries(
        full_info,
        dimensions=["i2v_subject", "camera_motion"],
        cases_per_dimension=1,
    )

    assert [(item["prompt_en"], item["image_name"]) for item in selected] == [("p1", "a.png"), ("p2", "b.png")]


def test_resolve_i2v_reference_path_prefers_crop_resolution(tmp_path: Path):
    crop_path = tmp_path / "crop" / "1-1"
    crop_path.mkdir(parents=True)
    image_path = crop_path / "demo.png"
    Image.new("RGB", (2, 2), color="black").save(image_path)

    resolved = resolve_i2v_reference_path(tmp_path, "demo.png", resolution="1-1")

    assert resolved == image_path


def test_summarize_vbench_results_partial_omits_total_score():
    raw_results = {
        "subject_consistency": [0.5731, [{"video_path": "/tmp/a.mp4", "video_results": 0.5731}]],
        "motion_smoothness": [0.85175, [{"video_path": "/tmp/b.mp4", "video_results": 0.85175}]],
        "aesthetic_quality": [0.5, [{"video_path": "/tmp/c.mp4", "video_results": 0.5}]],
        "object_class": [0.75, [{"video_path": "/tmp/d.mp4", "video_results": 0.75}]],
    }

    summary = summarize_vbench_results(raw_results)

    assert summary["partial_benchmark"] is True
    assert summary["sample_count"] == 4
    assert set(summary["evaluated_dimensions"]) == {
        "subject_consistency",
        "motion_smoothness",
        "aesthetic_quality",
        "object_class",
    }
    assert "background_consistency" in summary["missing_dimensions"]
    assert math.isclose(summary["raw_scores"]["subject consistency"], 0.5731)
    assert math.isclose(summary["normalized_scores"]["subject consistency"], 0.5)
    assert math.isclose(summary["normalized_scores"]["motion smoothness"], 0.5)
    assert math.isclose(summary["quality_score"], 0.5)
    assert math.isclose(summary["semantic_score"], 0.75)
    assert "total_score" not in summary


def test_summarize_vbench_i2v_results_partial_omits_total_score():
    raw_results = {
        "i2v_subject": [0.5731, [{"image_path": "/tmp/a.png", "video_path": "/tmp/a.mp4", "video_results": 0.5731}]],
        "i2v_background": [
            0.63075,
            [{"image_path": "/tmp/b.png", "video_path": "/tmp/b.mp4", "video_results": 0.63075}],
        ],
        "camera_motion": [0.5, [{"image_path": "/tmp/c.png", "video_path": "/tmp/c.mp4", "video_results": 0.5}]],
        "subject_consistency": [
            0.5731,
            [{"image_path": "/tmp/a.png", "video_path": "/tmp/a.mp4", "video_results": 0.5731}],
        ],
        "motion_smoothness": [
            0.85175,
            [{"image_path": "/tmp/d.png", "video_path": "/tmp/d.mp4", "video_results": 0.85175}],
        ],
    }

    summary = summarize_vbench_i2v_results(raw_results)

    assert summary["partial_benchmark"] is True
    assert summary["sample_count"] == 4
    assert set(summary["evaluated_dimensions"]) == {
        "i2v_subject",
        "i2v_background",
        "camera_motion",
        "subject_consistency",
        "motion_smoothness",
    }
    assert "dynamic_degree" in summary["missing_dimensions"]
    assert math.isclose(summary["normalized_scores"]["Video-Image Subject Consistency"], 0.5)
    assert math.isclose(summary["normalized_scores"]["Video-Image Background Consistency"], 0.5)
    assert math.isclose(summary["normalized_scores"]["Motion Smoothness"], 0.5)
    assert math.isclose(summary["quality_score"], 0.5)
    assert math.isclose(summary["i2v_score"], 0.5)
    assert "total_score" not in summary


def test_vbench_import_context_uses_explicit_repo_root(tmp_path: Path):
    fake_root = tmp_path / "fake_vbench"
    (fake_root / "vbench").mkdir(parents=True)
    (fake_root / "vbench2_beta_i2v").mkdir(parents=True)
    (fake_root / "vbench" / "__init__.py").write_text("MARKER = 'vbench'\n", encoding="utf-8")
    (fake_root / "vbench2_beta_i2v" / "__init__.py").write_text("MARKER = 'i2v'\n", encoding="utf-8")

    for module_name in ("vbench", "vbench2_beta_i2v"):
        sys.modules.pop(module_name, None)

    with vbench_import_context(fake_root):
        import vbench
        import vbench2_beta_i2v

        assert vbench.MARKER == "vbench"
        assert vbench2_beta_i2v.MARKER == "i2v"
