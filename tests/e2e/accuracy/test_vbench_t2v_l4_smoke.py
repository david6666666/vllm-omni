from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.accuracy.text_to_video.run_vbench import main as run_vbench_main
from tests.e2e.accuracy.conftest import infer_model_label, reset_artifact_dir
from tests.utils import hardware_test


@pytest.mark.advanced_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_vbench_t2v_l4_smoke(
    vbench_t2v_server,
    vbench_root: Path,
    accuracy_artifact_root: Path,
) -> None:
    model_label = infer_model_label(vbench_t2v_server.model).lower()
    output_root = reset_artifact_dir(accuracy_artifact_root / f"vbench_t2v_{model_label}")

    assert (
        run_vbench_main(
            [
                "generate",
                "--output-root",
                str(output_root),
                "--base-url",
                f"http://{vbench_t2v_server.host}:{vbench_t2v_server.port}",
                "--model",
                vbench_t2v_server.model,
                "--vbench-root",
                str(vbench_root),
                "--dimension",
                "subject_consistency",
                "motion_smoothness",
                "aesthetic_quality",
                "object_class",
                "--prompts-per-dimension",
                "1",
                "--samples-per-prompt",
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

    assert run_vbench_main(["evaluate", "--output-root", str(output_root), "--vbench-root", str(vbench_root)]) == 0
    assert run_vbench_main(["summarize", "--output-root", str(output_root)]) == 0

    assert (output_root / "generation_manifest.json").exists()
    assert (output_root / "evaluation_manifest.json").exists()
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["evaluation"]["partial_benchmark"] is True
    assert summary["evaluation"]["sample_count"] == 4
    assert set(summary["evaluation"]["evaluated_dimensions"]) == {
        "subject_consistency",
        "motion_smoothness",
        "aesthetic_quality",
        "object_class",
    }
    assert summary["evaluation"]["raw_scores"]
    assert summary["evaluation"]["quality_score"] >= 0.0
    assert summary["evaluation"]["semantic_score"] >= 0.0
    assert "total_score" not in summary["evaluation"]
