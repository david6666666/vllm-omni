from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks.accuracy.text_to_image.gbench import main as gbench_main
from tests.utils import hardware_test


@pytest.mark.advanced_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize(
    "accuracy_servers",
    [
        pytest.param(
            {
                "generate_model": os.environ.get("VLLM_TEST_GEBENCH_MODEL", "/workspace/models/Qwen/Qwen-Image-2512"),
                "judge_model": os.environ.get(
                    "VLLM_TEST_ACCURACY_JUDGE_MODEL",
                    "/workspace/models/QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ",
                ),
                "gpu": os.environ.get("VLLM_ACCURACY_GPU", os.environ.get("VLLM_ACCURACY_GEN_GPU", "0")),
                "generate_port": int(os.environ.get("VLLM_TEST_GEBENCH_PORT", "8093")),
                "judge_port": int(os.environ.get("VLLM_TEST_ACCURACY_JUDGE_PORT", "8094")),
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

    with accuracy_servers.generate_server() as generate_server:
        assert gbench_main(
            [
                "generate",
                "--dataset-root",
                str(gebench_dataset_root),
                "--output-root",
                str(output_root),
                "--base-url",
                f"http://{generate_server.host}:{generate_server.port}",
                "--model",
                generate_server.model,
                "--data-type",
                "type3",
                "--samples-per-type",
                str(gebench_samples_per_type),
                "--workers",
                str(accuracy_workers),
            ]
        ) == 0

    with accuracy_servers.judge_server() as judge_server:
        assert gbench_main(
            [
                "evaluate",
                "--dataset-root",
                str(gebench_dataset_root),
                "--output-root",
                str(output_root),
                "--data-type",
                "type3",
                "--judge-base-url",
                f"http://{judge_server.host}:{judge_server.port}",
                "--judge-model",
                judge_server.model,
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

    assert "type3" in summary["generation"]["by_type"]
    assert summary["generation"]["by_type"]["type3"]["count"] <= gebench_samples_per_type
    assert summary["generation"]["by_type"]["type3"]["count"] > 0
    assert "type3" in summary["evaluation"]["by_type"]
    assert summary["evaluation"]["by_type"]["type3"]["count"] <= gebench_samples_per_type
    assert summary["evaluation"]["by_type"]["type3"]["count"] > 0

    assert 0.0 <= summary["evaluation"]["overall_mean"] <= 1.0
