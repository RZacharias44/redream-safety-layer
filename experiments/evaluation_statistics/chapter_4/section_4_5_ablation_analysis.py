"""Statistics reported in Chapter 4.5: Ablation and Pipeline Analysis.

Run from the repository root:

    uv run python experiments/evaluation_statistics/chapter_4/section_4_5_ablation_analysis.py

The calculations use the row-level development and multi-maladaptive benchmark
files directly. Development accuracy metrics exclude the 60 unscored boundary
cases. The supplementary 35-case slice is reported separately.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEV_DATASET_FILE = ROOT / "data/benchmarks/dev_dataset.json"
MULTI_MALADAPTIVE_DATASET_FILE = (
    ROOT / "data/benchmarks/multi_maladaptive_dataset.json"
)

CONDITION_FILES = {
    "Full S1 pipeline": ROOT / "data/benchmarks/dev_s1_baseline.csv",
    "Stage-merge": ROOT / "data/benchmarks/dev_s1_ablation_stage_merge.csv",
    "No segmentation": (
        ROOT / "data/benchmarks/dev_s1_ablation_no_segmentation.csv"
    ),
    "Meta filtering off": ROOT / "data/benchmarks/dev_s1_ablation_meta_filter_off.csv",
    "Naive splitter": ROOT / "data/benchmarks/dev_s1_ablation_naive_splitter.csv",
}

MULTI_MALADAPTIVE_FILES = {
    "Full S1 pipeline": (
        ROOT / "data/benchmarks/multi_maladaptive_s1_baseline.csv"
    ),
    "Stage-merge": (
        ROOT / "data/benchmarks/multi_maladaptive_s1_ablation_stage_merge.csv"
    ),
    "No segmentation": (
        ROOT / "data/benchmarks/multi_maladaptive_s1_ablation_no_segmentation.csv"
    ),
    "Meta filtering off": (
        ROOT / "data/benchmarks/multi_maladaptive_s1_ablation_meta_filter_off.csv"
    ),
    "Naive splitter": (
        ROOT / "data/benchmarks/multi_maladaptive_s1_ablation_naive_splitter.csv"
    ),
}

MALADAPTIVE_NODES = {
    "AVOIDANCE",
    "INTERRUPTION",
    "VIOLENT_REVENGE",
    "SUPPRESSION",
    "TRAUMA_REPLAY",
}


@dataclass(frozen=True)
class Metrics:
    """Integer counts underlying one development-benchmark result row."""

    total_rows: int
    scored_rows: int
    unscored_rows: int
    safe_cases: int
    unsafe_cases: int
    safety_correct: int
    unsafe_detected: int
    node_correct: int
    false_alarms: int
    unsafe_misses: int


@dataclass(frozen=True)
class MultiMaladaptiveMetrics:
    """Counts for the supplementary two-maladaptive-pattern slice."""

    total: int
    primary_node_correct: int
    secondary_node_caught: int
    any_maladaptive_caught: int


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a row-level benchmark CSV."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_dataset(path: Path) -> list[dict[str, object]]:
    """Flatten scenario-based benchmark JSON into one row per test case."""
    suites = json.loads(path.read_text(encoding="utf-8"))
    tests: list[dict[str, object]] = []
    for suite in suites:
        for test in suite["tests"]:
            tests.append({**test, "suite": suite["suite"]})
    return tests


def dataset_index(path: Path) -> dict[str, dict[str, object]]:
    """Index dataset metadata by the exact user input used in result CSVs."""
    tests = read_dataset(path)
    index = {str(test["input"]): test for test in tests}
    assert len(index) == len(tests), f"Duplicate test input in {path}"
    return index


def as_bool(value: str) -> bool | None:
    """Parse CSV booleans; blank ground-truth values are unscored."""
    text = value.strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    if text == "":
        return None
    raise ValueError(f"Unexpected boolean value: {value!r}")


def calculate_metrics(rows: list[dict[str, str]]) -> Metrics:
    """Calculate safety and strict primary-node metrics from row-level data."""
    scored = [row for row in rows if as_bool(row["gt_safe"]) is not None]
    safe_cases = sum(as_bool(row["gt_safe"]) is True for row in scored)
    unsafe_cases = sum(as_bool(row["gt_safe"]) is False for row in scored)
    unsafe_detected = sum(
        as_bool(row["gt_safe"]) is False and as_bool(row["pred_safe"]) is False
        for row in scored
    )
    unsafe_misses = sum(
        as_bool(row["gt_safe"]) is False and as_bool(row["pred_safe"]) is True
        for row in scored
    )
    false_alarms = sum(
        as_bool(row["gt_safe"]) is True and as_bool(row["pred_safe"]) is False
        for row in scored
    )
    safe_correct = sum(
        as_bool(row["gt_safe"]) is True and as_bool(row["pred_safe"]) is True
        for row in scored
    )
    node_correct = sum(row["gt_node"] == row["pred_node"] for row in scored)

    assert unsafe_detected + unsafe_misses == unsafe_cases
    assert false_alarms + safe_correct == safe_cases

    return Metrics(
        total_rows=len(rows),
        scored_rows=len(scored),
        unscored_rows=len(rows) - len(scored),
        safe_cases=safe_cases,
        unsafe_cases=unsafe_cases,
        safety_correct=unsafe_detected + safe_correct,
        unsafe_detected=unsafe_detected,
        node_correct=node_correct,
        false_alarms=false_alarms,
        unsafe_misses=unsafe_misses,
    )


def calculate_all_conditions() -> dict[str, Metrics]:
    """Calculate and validate the five main ablation conditions."""
    dev_index = dataset_index(DEV_DATASET_FILE)
    results: dict[str, Metrics] = {}
    for condition, path in CONDITION_FILES.items():
        rows = read_csv(path)
        inputs = [row["user_input"] for row in rows]
        assert len(inputs) == len(set(inputs)), f"Duplicate row in {path}"
        assert set(inputs) == set(dev_index), f"Dataset mismatch in {path}"
        results[condition] = calculate_metrics(rows)
    return results


def calculate_stratum(
    rows: list[dict[str, str]],
    dev_index: dict[str, dict[str, object]],
    test_type: str,
) -> Metrics:
    """Calculate metrics after filtering by the dataset's test_type field."""
    selected = [
        row
        for row in rows
        if dev_index[row["user_input"]].get("test_type") == test_type
    ]
    return calculate_metrics(selected)


def multipart_composition(
    dev_index: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Describe the scored 60-case multipart stratum from dataset metadata."""
    rows = [
        row for row in dev_index.values() if row.get("test_type") == "multipart"
    ]
    pattern_counts = Counter(str(row.get("pattern")) for row in rows)
    max_maladaptive_patterns = max(
        str(row.get("pattern")).split(" + ").count("MALADAPTIVE")
        for row in rows
    )
    return {
        "total": len(rows),
        "safe": sum(row.get("expected_safe") is True for row in rows),
        "unsafe": sum(row.get("expected_safe") is False for row in rows),
        "pattern_counts": pattern_counts,
        "max_maladaptive_patterns": max_maladaptive_patterns,
    }


def segment_nodes(row: dict[str, str]) -> set[str]:
    """Return all node IDs preserved in a row's structured segment output."""
    raw = row.get("segments_json", "")
    if not raw:
        return set()
    segments = json.loads(raw)
    return {
        str(segment.get("node_id"))
        for segment in segments
        if isinstance(segment, dict) and segment.get("node_id")
    }


def calculate_multi_maladaptive_metrics(
    rows: list[dict[str, str]],
    multi_index: dict[str, dict[str, object]],
) -> MultiMaladaptiveMetrics:
    """Calculate primary, secondary, and any-maladaptive detection counts."""
    inputs = [row["user_input"] for row in rows]
    assert len(inputs) == len(set(inputs)), "Duplicate multi-maladaptive row"
    assert set(inputs) == set(multi_index), "Multi-maladaptive dataset mismatch"

    primary_node_correct = 0
    secondary_node_caught = 0
    any_maladaptive_caught = 0
    for row in rows:
        nodes = segment_nodes(row)
        expected_secondary = str(
            multi_index[row["user_input"]]["expected_secondary_node"]
        )
        primary_node_correct += row["gt_node"] == row["pred_node"]
        secondary_node_caught += expected_secondary in nodes
        any_maladaptive_caught += bool(nodes & MALADAPTIVE_NODES)

    return MultiMaladaptiveMetrics(
        total=len(rows),
        primary_node_correct=primary_node_correct,
        secondary_node_caught=secondary_node_caught,
        any_maladaptive_caught=any_maladaptive_caught,
    )


def calculate_all_multi_maladaptive() -> dict[str, MultiMaladaptiveMetrics]:
    """Calculate and validate all five supplementary-slice conditions."""
    multi_index = dataset_index(MULTI_MALADAPTIVE_DATASET_FILE)
    return {
        condition: calculate_multi_maladaptive_metrics(read_csv(path), multi_index)
        for condition, path in MULTI_MALADAPTIVE_FILES.items()
    }


def illustrative_case() -> dict[str, object]:
    """Trace the 'Cower then crush snakes' example across two conditions."""
    multi_index = dataset_index(MULTI_MALADAPTIVE_DATASET_FILE)
    user_input = next(
        input_text
        for input_text, metadata in multi_index.items()
        if metadata["description"] == "Cower then crush snakes"
    )
    full = next(
        row
        for row in read_csv(MULTI_MALADAPTIVE_FILES["Full S1 pipeline"])
        if row["user_input"] == user_input
    )
    no_segmentation = next(
        row
        for row in read_csv(MULTI_MALADAPTIVE_FILES["No segmentation"])
        if row["user_input"] == user_input
    )
    return {
        "expected_primary": str(multi_index[user_input]["expected_node"]),
        "expected_secondary": str(
            multi_index[user_input]["expected_secondary_node"]
        ),
        "full_pred_safe": as_bool(full["pred_safe"]),
        "full_pred_node": full["pred_node"],
        "full_segment_nodes": segment_nodes(full),
        "no_segmentation_pred_safe": as_bool(no_segmentation["pred_safe"]),
        "no_segmentation_pred_node": no_segmentation["pred_node"],
        "no_segmentation_segment_nodes": segment_nodes(no_segmentation),
    }


def percent(count: int, total: int) -> str:
    """Format a table value with numerator and denominator."""
    return f"{100 * count / total:.2f}% ({count}/{total})"


def main() -> None:
    dev_index = dataset_index(DEV_DATASET_FILE)
    type_counts = Counter(
        str(row.get("test_type")) for row in dev_index.values()
    )
    print("Chapter 4.5 -- Ablation and Pipeline Analysis")
    print("\n1. Development benchmark coverage")
    print(f"all cases: {len(dev_index)}")
    for test_type, count in sorted(type_counts.items()):
        print(f"{test_type}: {count}")

    print("\n2. Main ablation results")
    print(
        "condition | safety accuracy | unsafe recall | primary-node accuracy | "
        "false alarms | unsafe misses"
    )
    for condition, metrics in calculate_all_conditions().items():
        print(
            f"{condition} | "
            f"{percent(metrics.safety_correct, metrics.scored_rows)} | "
            f"{percent(metrics.unsafe_detected, metrics.unsafe_cases)} | "
            f"{percent(metrics.node_correct, metrics.scored_rows)} | "
            f"{metrics.false_alarms} | {metrics.unsafe_misses}"
        )

    print("\n3. Clean and multipart strata")
    for condition in ("Full S1 pipeline", "No segmentation"):
        rows = read_csv(CONDITION_FILES[condition])
        for test_type in ("clean", "multipart"):
            metrics = calculate_stratum(rows, dev_index, test_type)
            print(
                f"{condition} / {test_type} | "
                f"{percent(metrics.safety_correct, metrics.scored_rows)} | "
                f"{percent(metrics.unsafe_detected, metrics.unsafe_cases)} | "
                f"{percent(metrics.node_correct, metrics.scored_rows)} | "
                f"{metrics.false_alarms} | {metrics.unsafe_misses}"
            )

    composition = multipart_composition(dev_index)
    print("\n4. Multipart composition")
    print(
        f"total: {composition['total']}; safe: {composition['safe']}; "
        f"unsafe: {composition['unsafe']}; maximum maladaptive patterns per case: "
        f"{composition['max_maladaptive_patterns']}"
    )
    for pattern, count in composition["pattern_counts"].most_common():
        print(f"{pattern}: {count}")

    print("\n5. Supplementary multi-maladaptive slice")
    print("condition | primary node | secondary node | any maladaptive")
    for condition, metrics in calculate_all_multi_maladaptive().items():
        print(
            f"{condition} | "
            f"{percent(metrics.primary_node_correct, metrics.total)} | "
            f"{percent(metrics.secondary_node_caught, metrics.total)} | "
            f"{percent(metrics.any_maladaptive_caught, metrics.total)}"
        )

    example = illustrative_case()
    print("\n6. Cower then crush snakes trace")
    for key, value in example.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
