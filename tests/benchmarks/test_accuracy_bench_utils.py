import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.accuracy.image_to_image.gedit_bench import (
    GROUPS as GEDIT_GROUPS,
    parse_score_payload,
    summarize_gedit_rows,
)
from benchmarks.accuracy.text_to_image.gbench import summarize_gebench_results


def test_summarize_gebench_results_computes_type_and_global_means():
    results = [
        {"data_type": "type1", "overall": 0.8, "scores": {"goal": 5, "logic": 4}},
        {"data_type": "type1", "overall": 0.6, "scores": {"goal": 3, "logic": 4}},
        {"data_type": "type2", "overall": 0.5, "scores": {"goal": 2, "logic": 3}},
    ]

    summary = summarize_gebench_results(results)

    assert math.isclose(summary["overall_mean"], (0.8 + 0.6 + 0.5) / 3)
    assert math.isclose(summary["by_type"]["type1"]["overall_mean"], 0.7)
    assert math.isclose(summary["by_type"]["type2"]["overall_mean"], 0.5)
    assert math.isclose(summary["by_type"]["type1"]["score_means"]["goal"], 4.0)


def test_parse_score_payload_handles_raw_json_and_delimited_json():
    raw = '{"score": [7, 8], "reasoning": "ok"}'
    wrapped = 'prefix ||V^=^V|| {"score": [6], "reasoning": "fine"} ||V^=^V|| suffix'

    assert parse_score_payload(raw)["score"] == [7, 8]
    assert parse_score_payload(wrapped)["score"] == [6]


def test_summarize_gedit_rows_computes_group_and_intersection_means():
    rows = []
    for group in GEDIT_GROUPS:
        rows.append(
            {
                "task_type": group,
                "instruction_language": "en",
                "sementics_score": 8.0,
                "quality_score": 9.0,
                "intersection_exist": True,
            }
        )
        rows.append(
            {
                "task_type": group,
                "instruction_language": "en",
                "sementics_score": 6.0,
                "quality_score": 4.0,
                "intersection_exist": False,
            }
        )

    summary = summarize_gedit_rows(rows, language="en")

    expected_overall = (math.sqrt(8.0 * 9.0) + math.sqrt(6.0 * 4.0)) / 2
    assert math.isclose(summary["overall"]["avg_semantics"], 7.0)
    assert math.isclose(summary["overall"]["avg_quality"], 6.5)
    assert math.isclose(summary["overall"]["avg_overall"], expected_overall)
    assert math.isclose(summary["intersection"]["avg_semantics"], 8.0)
