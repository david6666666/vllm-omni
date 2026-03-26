from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.accuracy.image_to_video.run_vbench_i2v import main as run_vbench_i2v_main
from tests.e2e.accuracy.conftest import infer_model_label, reset_artifact_dir
from tests.utils import hardware_test


@pytest.mark.advanced_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_vbench_i2v_l4_smoke(
    vbench_i2v_server,
    vbench_root: Path,
    accuracy_artifact_root: Path,
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "custom_input"
    image_root.mkdir(parents=True)

    cases = [
        {
            "prompt": "A red kite sways in the wind",
            "dimension": ["i2v_subject"],
            "image_path": str(image_root / "kite.png"),
        },
        {
            "prompt": "A calm lake at dawn",
            "dimension": ["i2v_background"],
            "image_path": str(image_root / "lake.png"),
        },
        {
            "prompt": "A toy train circles the track, camera pans left",
            "dimension": ["camera_motion"],
            "image_path": str(image_root / "train.png"),
        },
    ]
    for case in cases:
        Image.new("RGB", (512, 512), color="white").save(case["image_path"])

    custom_input_json = tmp_path / "custom_input.json"
    custom_input_json.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    model_label = infer_model_label(vbench_i2v_server.model).lower()
    output_root = reset_artifact_dir(accuracy_artifact_root / f"vbench_i2v_{model_label}")

    assert (
        run_vbench_i2v_main(
            [
                "generate",
                "--output-root",
                str(output_root),
                "--base-url",
                f"http://{vbench_i2v_server.host}:{vbench_i2v_server.port}",
                "--model",
                vbench_i2v_server.model,
                "--vbench-root",
                str(vbench_root),
                "--custom-input-json",
                str(custom_input_json),
                "--samples-per-case",
                "1",
                "--width",
                "640",
                "--height",
                "480",
                "--num-frames",
                "5",
                "--fps",
                "8",
                "--num-inference-steps",
                "2",
                "--guidance-scale",
                "1.0",
                "--seed",
                "42",
            ]
        )
        == 0
    )

    assert run_vbench_i2v_main(["evaluate", "--output-root", str(output_root), "--vbench-root", str(vbench_root)]) == 0
    assert run_vbench_i2v_main(["summarize", "--output-root", str(output_root)]) == 0

    assert (output_root / "generation_manifest.json").exists()
    assert (output_root / "evaluation_manifest.json").exists()
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["evaluation"]["partial_benchmark"] is True
    assert summary["evaluation"]["sample_count"] == 3
    assert set(summary["evaluation"]["evaluated_dimensions"]) == {
        "i2v_subject",
        "i2v_background",
        "camera_motion",
    }
    assert summary["evaluation"]["raw_scores"]
    assert summary["evaluation"]["i2v_score"] >= 0.0
    assert "total_score" not in summary["evaluation"]
