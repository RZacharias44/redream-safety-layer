from collections import Counter

from experiments.evaluation_statistics.chapter_4.section_4_5_ablation_analysis import (
    CONDITION_FILES,
    DEV_DATASET_FILE,
    Metrics,
    MultiMaladaptiveMetrics,
    calculate_all_conditions,
    calculate_all_multi_maladaptive,
    calculate_stratum,
    dataset_index,
    illustrative_case,
    multipart_composition,
    read_csv,
)


def test_section_4_5_main_ablation_table() -> None:
    expected = {
        "Full S1 pipeline": Metrics(480, 420, 60, 240, 180, 411, 179, 328, 8, 1),
        "Stage-merge": Metrics(480, 420, 60, 240, 180, 413, 177, 365, 4, 3),
        "No segmentation": Metrics(480, 420, 60, 240, 180, 413, 173, 352, 0, 7),
        "Meta filtering off": Metrics(480, 420, 60, 240, 180, 401, 179, 318, 18, 1),
        "Naive splitter": Metrics(480, 420, 60, 240, 180, 409, 178, 332, 9, 2),
    }
    assert calculate_all_conditions() == expected


def test_section_4_5_stratified_no_segmentation_table() -> None:
    dev_index = dataset_index(DEV_DATASET_FILE)
    expected = {
        ("Full S1 pipeline", "clean"): Metrics(
            360, 360, 0, 210, 150, 351, 149, 286, 8, 1
        ),
        ("No segmentation", "clean"): Metrics(
            360, 360, 0, 210, 150, 353, 143, 310, 0, 7
        ),
        ("Full S1 pipeline", "multipart"): Metrics(
            60, 60, 0, 30, 30, 60, 30, 42, 0, 0
        ),
        ("No segmentation", "multipart"): Metrics(
            60, 60, 0, 30, 30, 60, 30, 42, 0, 0
        ),
    }
    for key, metrics in expected.items():
        condition, test_type = key
        rows = read_csv(CONDITION_FILES[condition])
        assert calculate_stratum(rows, dev_index, test_type) == metrics


def test_section_4_5_multipart_composition() -> None:
    result = multipart_composition(dataset_index(DEV_DATASET_FILE))
    assert result == {
        "total": 60,
        "safe": 30,
        "unsafe": 30,
        "pattern_counts": Counter(
            {
                "META + ADAPTIVE": 10,
                "FEELING + MALADAPTIVE": 10,
                "ADAPTIVE + MALADAPTIVE": 10,
                "META + MALADAPTIVE": 10,
                "FEELING + ADAPTIVE": 10,
                "ADAPTIVE + ADAPTIVE": 10,
            }
        ),
        "max_maladaptive_patterns": 1,
    }


def test_section_4_5_multi_maladaptive_table() -> None:
    expected = {
        "Full S1 pipeline": MultiMaladaptiveMetrics(35, 35, 34, 35),
        "Stage-merge": MultiMaladaptiveMetrics(35, 35, 35, 35),
        "No segmentation": MultiMaladaptiveMetrics(35, 25, 6, 31),
        "Meta filtering off": MultiMaladaptiveMetrics(35, 35, 34, 35),
        "Naive splitter": MultiMaladaptiveMetrics(35, 25, 6, 31),
    }
    assert calculate_all_multi_maladaptive() == expected


def test_section_4_5_illustrative_case_trace() -> None:
    assert illustrative_case() == {
        "expected_primary": "VIOLENT_REVENGE",
        "expected_secondary": "AVOIDANCE",
        "full_pred_safe": False,
        "full_pred_node": "VIOLENT_REVENGE",
        "full_segment_nodes": {"AVOIDANCE", "VIOLENT_REVENGE"},
        "no_segmentation_pred_safe": True,
        "no_segmentation_pred_node": "BEHAVIORAL_MASTERY",
        "no_segmentation_segment_nodes": {"BEHAVIORAL_MASTERY"},
    }
