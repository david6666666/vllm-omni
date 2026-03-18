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
                "generate_model": os.environ.get("VLLM_TEST_GEDIT_MODEL", "/workspace/models/Qwen/Qwen-Image-Edit"),
                "judge_model": os.environ.get(
                    "VLLM_TEST_ACCURACY_JUDGE_MODEL",
                    "/workspace/models/QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ",
                ),
                "generate_gpu": os.environ.get("VLLM_ACCURACY_GEN_GPU", "0"),
                "judge_gpu": os.environ.get("VLLM_ACCURACY_JUDGE_GPU", "1"),
                "generate_port": int(os.environ.get("VLLM_TEST_GEDIT_PORT", "8093")),
                "judge_port": int(os.environ.get("VLLM_TEST_ACCURACY_JUDGE_PORT", "8094")),
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
    assert set(summary["languages"]) == {"en", "cn"}

    for language in ["en", "cn"]:
        language_summary = summary["languages"][language]
        assert language_summary["overall"]["Q_SC"] is not None
        assert language_summary["overall"]["Q_PQ"] is not None
        assert language_summary["overall"]["Q_O"] is not None

        for group in GROUPS:
            group_summary = language_summary["by_group"][group]
            assert set(group_summary) == {"Q_SC", "Q_PQ", "Q_O"}
