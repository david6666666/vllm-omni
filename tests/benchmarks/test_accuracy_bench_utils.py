import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.accuracy.image_to_image.gedit_bench import (
    GROUPS as GEDIT_GROUPS,
    parse_score_payload,
    summarize_generated_records as summarize_gedit_generated_records,
    summarize_gedit_rows,
)
from benchmarks.accuracy.text_to_image.gbench import (
    summarize_generated_records as summarize_gebench_generated_records,
    summarize_gebench_results,
)


def test_summarize_gebench_generated_records_groups_by_type():
    records = [
        {"data_type": "type1", "sample_name": "english_phone/folder_1", "output_path": "a.png"},
        {"data_type": "type1", "sample_name": "english_phone/folder_2", "output_path": "b.png"},
        {"data_type": "type2", "sample_name": "english_phone/folder_3", "output_path": "c.png"},
    ]

    summary = summarize_gebench_generated_records(records)

    assert summary["count"] == 3
    assert summary["by_type"]["type1"]["count"] == 2
    assert summary["by_type"]["type2"]["count"] == 1
    assert summary["by_type"]["type1"]["samples"] == [
        "english_phone/folder_1",
        "english_phone/folder_2",
    ]


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


def test_summarize_gedit_generated_records_groups_by_task_and_language():
    records = []
    for group in GEDIT_GROUPS[:2]:
        records.append(
            {
                "task_type": group,
                "instruction_language": "en",
                "key": f"{group}_en",
                "output_path": f"{group}_en.png",
            }
        )
        records.append(
            {
                "task_type": group,
                "instruction_language": "cn",
                "key": f"{group}_cn",
                "output_path": f"{group}_cn.png",
            }
        )

    summary = summarize_gedit_generated_records(records)

    assert summary["count"] == 4
    assert summary["by_task"][GEDIT_GROUPS[0]]["count"] == 2
    assert summary["by_language"]["en"]["count"] == 2
    assert summary["by_language"]["cn"]["samples"] == [
        f"{GEDIT_GROUPS[0]}_cn",
        f"{GEDIT_GROUPS[1]}_cn",
    ]


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
