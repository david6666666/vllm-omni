from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks.accuracy.image_to_image.gedit_bench import GROUPS, main as gedit_main
from tests.utils import hardware_test


@pytest.mark.advanced_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize(
    "accuracy_servers",
    [
        pytest.param(
            {
                "generate_model": os.environ.get("VLLM_TEST_GEDIT_MODEL", "Qwen/Qwen-Image-Edit"),
                "judge_model": os.environ.get("VLLM_TEST_ACCURACY_JUDGE_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
            },
            id="gedit-bench",
        )
    ],
    indirect=True,
)
def test_gedit_bench_h100_smoke(
    accuracy_servers,
    gedit_dataset_root: Path,
    gedit_samples_per_group: int,
    accuracy_workers: int,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "gedit_results"
    score_root = tmp_path / "gedit_scores"
    model_name = "smoke_qwen_image_edit"

    assert gedit_main(
        [
            "generate",
            "--dataset-ref",
            str(gedit_dataset_root),
            "--output-root",
            str(output_root),
            "--base-url",
            accuracy_servers.generate_base_url,
            "--model",
            accuracy_servers.generate_server.model,
            "--model-name",
            model_name,
            "--task-type",
            "all",
            "--instruction-language",
            "all",
            "--samples-per-group",
            str(gedit_samples_per_group),
            "--workers",
            str(accuracy_workers),
        ]
    ) == 0

    assert gedit_main(
        [
            "evaluate",
            "--dataset-ref",
            str(gedit_dataset_root),
            "--output-root",
            str(output_root),
            "--model-name",
            model_name,
            "--save-dir",
            str(score_root),
            "--task-type",
            "all",
            "--instruction-language",
            "all",
            "--judge-base-url",
            accuracy_servers.judge_base_url,
            "--judge-model",
            accuracy_servers.judge_server.model,
            "--judge-api-key",
            "EMPTY",
            "--samples-per-group",
            str(gedit_samples_per_group),
            "--workers",
            str(accuracy_workers),
        ]
    ) == 0

    csv_path = score_root / f"{model_name}_all_all_vie_score.csv"
    assert gedit_main(["summarize", "--csv-path", str(csv_path), "--language", "all"]) == 0

    summary_path = score_root / f"{model_name}_all_all_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["overall"]["avg_semantics"] is not None
    assert summary["overall"]["avg_quality"] is not None
    assert summary["overall"]["avg_overall"] is not None

    for group in GROUPS:
        group_summary = summary["by_group"][group]
        assert group_summary["avg_overall"] is not None
