from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks.accuracy.text_to_image.gbench import TYPE_TO_FOLDER, main as gbench_main
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
                "generate_model": os.environ.get("VLLM_TEST_GEBENCH_MODEL"),
                "judge_model": os.environ.get("VLLM_TEST_ACCURACY_JUDGE_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
            },
            id="gebench",
        )
    ],
    indirect=True,
)
def test_gebench_h100_smoke(
    accuracy_servers,
    gebench_dataset_root: Path,
    gebench_samples_per_type: int,
    accuracy_workers: int,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "gebench"

    assert gbench_main(
        [
            "generate",
            "--dataset-root",
            str(gebench_dataset_root),
            "--output-root",
            str(output_root),
            "--base-url",
            accuracy_servers.generate_base_url,
            "--model",
            accuracy_servers.generate_server.model,
            "--data-type",
            "all",
            "--samples-per-type",
            str(gebench_samples_per_type),
            "--workers",
            str(accuracy_workers),
        ]
    ) == 0

    assert gbench_main(
        [
            "evaluate",
            "--dataset-root",
            str(gebench_dataset_root),
            "--output-root",
            str(output_root),
            "--data-type",
            "all",
            "--judge-base-url",
            accuracy_servers.judge_base_url,
            "--judge-model",
            accuracy_servers.judge_server.model,
            "--judge-api-key",
            "EMPTY",
            "--samples-per-type",
            str(gebench_samples_per_type),
            "--workers",
            str(accuracy_workers),
        ]
    ) == 0

    assert gbench_main(["summarize", "--output-root", str(output_root)]) == 0

    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert "generation" in summary
    assert "evaluation" in summary

    for data_type in TYPE_TO_FOLDER:
        assert data_type in summary["generation"]["by_type"]
        assert summary["generation"]["by_type"][data_type]["count"] <= gebench_samples_per_type
        assert summary["generation"]["by_type"][data_type]["count"] > 0
        assert data_type in summary["evaluation"]["by_type"]
        assert summary["evaluation"]["by_type"][data_type]["count"] <= gebench_samples_per_type
        assert summary["evaluation"]["by_type"][data_type]["count"] > 0

    assert 0.0 <= summary["evaluation"]["overall_mean"] <= 1.0
