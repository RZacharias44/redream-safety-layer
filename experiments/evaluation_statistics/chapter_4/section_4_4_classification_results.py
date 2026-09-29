"""Statistics reported in Chapter 4.4: Safety Layer Classification Results.

Run from the repository root:

    uv run python experiments/evaluation_statistics/chapter_4/section_4_4_classification_results.py

The calculations use the row-level benchmark CSV files directly. Rows with a
blank ``gt_safe`` value are unscored boundary cases and are excluded from the
accuracy denominators.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from experiments.evaluation_statistics.consistency_votes import derive_vote_verdicts


ROOT = Path(__file__).resolve().parents[3]

CONDITION_FILES = {
    "S1 baseline": ROOT / "data/benchmarks/dev_s1_baseline.csv",
    "S1 consistency": ROOT / "data/benchmarks/dev_s1_consistency_n5.csv",
    "S2 baseline": ROOT / "data/benchmarks/dev_s2_baseline.csv",
    "S2 consistency": ROOT / "data/benchmarks/dev_s2_consistency_n5.csv",
}

HELDOUT_RESULTS_FILE = ROOT / "data/benchmarks/test_s1_baseline.csv"
HELDOUT_DATASET_FILE = ROOT / "data/benchmarks/test_dataset.json"

MALADAPTIVE_NODES = (
    "AVOIDANCE",
    "INTERRUPTION",
    "VIOLENT_REVENGE",
    "SUPPRESSION",
    "TRAUMA_REPLAY",
)

AFFECT_EMOTIONAL_PAIR = {"AFFECT_EXPRESSION", "EMOTIONAL_MASTERY"}


@dataclass(frozen=True)
class Metrics:
    """Integer counts from which all reported percentages are calculated."""

    total_rows: int
    scored_rows: int
    unscored_rows: int
    safe_cases: int
    unsafe_cases: int
    safety_correct: int
    unsafe_detected: int
    false_alarms: int
    unsafe_misses: int
    node_correct: int
    collapsed_node_correct: int
    meta_only_outputs: int
    per_node: dict[str, tuple[int, int]]
    per_node_safety_correct: dict[str, int]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def as_bool(value: str) -> bool | None:
    """Parse CSV booleans; a blank value represents an unscored case."""
    text = value.strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    if text == "":
        return None
    raise ValueError(f"Unexpected boolean value: {value!r}")


def calculate_metrics(rows: list[dict[str, str]]) -> Metrics:
    scored = [row for row in rows if as_bool(row["gt_safe"]) is not None]

    for row in scored:
        assert as_bool(row["pred_safe"]) is not None
        assert row["gt_node"], "A scored row has no reference node."
        assert row["pred_node"], "A scored row has no predicted node."

    safe_cases = sum(as_bool(row["gt_safe"]) is True for row in scored)
    unsafe_cases = sum(as_bool(row["gt_safe"]) is False for row in scored)

    # Unsafe is the positive class.
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
    collapsed_node_correct = sum(
        row["gt_node"] == row["pred_node"]
        or {row["gt_node"], row["pred_node"]} == AFFECT_EMOTIONAL_PAIR
        for row in scored
    )

    per_node: dict[str, tuple[int, int]] = {}
    per_node_safety_correct: dict[str, int] = {}
    for node in sorted({row["gt_node"] for row in scored}):
        node_rows = [row for row in scored if row["gt_node"] == node]
        correct = sum(row["pred_node"] == node for row in node_rows)
        per_node[node] = (correct, len(node_rows))
        per_node_safety_correct[node] = sum(
            as_bool(row["gt_safe"]) == as_bool(row["pred_safe"])
            for row in node_rows
        )

    safety_correct = unsafe_detected + safe_correct
    assert unsafe_detected + unsafe_misses == unsafe_cases
    assert false_alarms + safe_correct == safe_cases
    assert safety_correct == sum(
        as_bool(row["gt_safe"]) == as_bool(row["pred_safe"]) for row in scored
    )

    return Metrics(
        total_rows=len(rows),
        scored_rows=len(scored),
        unscored_rows=len(rows) - len(scored),
        safe_cases=safe_cases,
        unsafe_cases=unsafe_cases,
        safety_correct=safety_correct,
        unsafe_detected=unsafe_detected,
        false_alarms=false_alarms,
        unsafe_misses=unsafe_misses,
        node_correct=node_correct,
        collapsed_node_correct=collapsed_node_correct,
        meta_only_outputs=sum(row["pred_node"] == "META_ONLY" for row in scored),
        per_node=per_node,
        per_node_safety_correct=per_node_safety_correct,
    )


def load_heldout_scored_rows() -> list[dict[str, str]]:
    """Apply the held-out exclusions described in Chapter 4."""
    suites = json.loads(HELDOUT_DATASET_FILE.read_text(encoding="utf-8"))
    test_type_by_input = {
        test["input"]: test.get("test_type", "clean")
        for suite in suites
        for test in suite["tests"]
    }

    rows = read_csv(HELDOUT_RESULTS_FILE)
    return [
        row
        for row in rows
        if "buried alive" not in row["suite"].lower()
        and test_type_by_input[row["user_input"]] != "boundary"
    ]


def condition_rows(condition: str, path: Path) -> list[dict[str, str]]:
    """Load one condition's rows, scored on the decision rule it describes.

    A consistency run stores two predictions per row: the temperature-0 pipeline
    output in ``pred_safe``/``pred_node``, and the per-segment votes in
    ``vote_distribution``. Only the votes instantiate the majority-vote rule, so
    for those conditions the vote-derived verdict replaces the pipeline columns
    before any metric is calculated.
    """
    rows = read_csv(path)
    if "consistency" not in condition.lower():
        return rows
    voted, _diagnostics = derive_vote_verdicts(rows)
    return [
        {**row, "pred_safe": row["vote_safe"], "pred_node": row["vote_node"]}
        for row in voted
    ]


def calculate_all() -> dict[str, Metrics]:
    results = {
        condition: calculate_metrics(condition_rows(condition, path))
        for condition, path in CONDITION_FILES.items()
    }
    results["Held-out test"] = calculate_metrics(load_heldout_scored_rows())
    return results


def node_confusion_pairs(
    rows: list[dict[str, str]],
) -> Counter[tuple[str, str]]:
    """Count strict primary-node errors by reference/prediction pair."""
    return Counter(
        (row["gt_node"], row["pred_node"])
        for row in rows
        if as_bool(row["gt_safe"]) is not None
        and row["gt_node"] != row["pred_node"]
    )


def safety_error_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return scored rows whose final binary safety verdict is incorrect."""
    return [
        row
        for row in rows
        if as_bool(row["gt_safe"]) is not None
        and as_bool(row["gt_safe"]) != as_bool(row["pred_safe"])
    ]


def percent(numerator: int, denominator: int, digits: int = 2) -> str:
    return f"{100 * numerator / denominator:.{digits}f}% ({numerator}/{denominator})"


def percentage_points(after: int, before: int, denominator: int) -> float:
    return 100 * (after - before) / denominator


def maladaptive_range(metrics: Metrics) -> tuple[float, float]:
    accuracies = []
    for node in MALADAPTIVE_NODES:
        correct, total = metrics.per_node[node]
        accuracies.append(100 * correct / total)
    return min(accuracies), max(accuracies)


def print_results(results: dict[str, Metrics]) -> None:
    print("Chapter 4.4 — Safety Layer Classification Results")

    print("\n1. Development benchmark coverage")
    coverage = results["S1 baseline"]
    print(f"all rows: {coverage.total_rows}")
    print(f"scored rows: {coverage.scored_rows}")
    print(f"unscored boundary rows: {coverage.unscored_rows}")
    print(f"safe scored cases: {coverage.safe_cases}")
    print(f"unsafe scored cases: {coverage.unsafe_cases}")

    for condition in CONDITION_FILES:
        metrics = results[condition]
        assert (
            metrics.total_rows,
            metrics.scored_rows,
            metrics.unscored_rows,
            metrics.safe_cases,
            metrics.unsafe_cases,
        ) == (480, 420, 60, 240, 180)

    print("\n2. Safety classification results")
    print("condition | safety accuracy | unsafe recall | false alarms | unsafe misses")
    for condition in CONDITION_FILES:
        metrics = results[condition]
        print(
            f"{condition} | "
            f"{percent(metrics.safety_correct, metrics.scored_rows)} | "
            f"{percent(metrics.unsafe_detected, metrics.unsafe_cases)} | "
            f"{metrics.false_alarms} | {metrics.unsafe_misses}"
        )

    print("\n3. Strict primary-node results")
    print("condition | node accuracy | maladaptive-node range | META_ONLY outputs")
    for condition in CONDITION_FILES:
        metrics = results[condition]
        low, high = maladaptive_range(metrics)
        print(
            f"{condition} | "
            f"{percent(metrics.node_correct, metrics.scored_rows)} | "
            f"{low:.1f}–{high:.1f}% | {metrics.meta_only_outputs}"
        )

    print("\n4. Affect-expression / emotional-mastery sensitivity check")
    for condition in ("S1 consistency", "S2 consistency", "Held-out test"):
        metrics = results[condition]
        print(
            f"{condition}: "
            f"{percent(metrics.node_correct, metrics.scored_rows, 1)} -> "
            f"{percent(metrics.collapsed_node_correct, metrics.scored_rows, 1)}"
        )

    print("\n5. S2 consistency accuracy by maladaptive node")
    s2_consistency = results["S2 consistency"]
    for node in MALADAPTIVE_NODES:
        correct, total = s2_consistency.per_node[node]
        print(f"{node}: {percent(correct, total, 1)}")

    print("\n6. Direct comparisons reported in the text")
    s1 = results["S1 baseline"]
    s2 = results["S2 baseline"]
    print(
        "S2 baseline vs S1 baseline: "
        f"{s2.safety_correct - s1.safety_correct:+d} correct safety case; "
        f"{s2.node_correct - s1.node_correct:+d} correct primary nodes; "
        f"{s2.false_alarms - s1.false_alarms:+d} false alarm; "
        f"{s2.unsafe_misses - s1.unsafe_misses:+d} unsafe misses."
    )

    s1_consistency = results["S1 consistency"]
    print(
        "S1 consistency vs S1 baseline: "
        f"{s1_consistency.safety_correct - s1.safety_correct:+d} correct safety cases; "
        f"{s1_consistency.node_correct - s1.node_correct:+d} correct primary nodes."
    )

    s2_consistency = results["S2 consistency"]
    print(
        "S2 consistency vs S2 baseline: "
        f"{s2_consistency.safety_correct - s2.safety_correct:+d} correct safety cases "
        f"({percentage_points(s2_consistency.safety_correct, s2.safety_correct, s2.scored_rows):+.2f} pp); "
        f"{s2_consistency.node_correct - s2.node_correct:+d} correct primary nodes "
        f"({percentage_points(s2_consistency.node_correct, s2.node_correct, s2.scored_rows):+.2f} pp); "
        f"{s2_consistency.false_alarms - s2.false_alarms:+d} false alarms; "
        f"{s2_consistency.unsafe_misses - s2.unsafe_misses:+d} unsafe misses."
    )

    heldout_rows = load_heldout_scored_rows()
    heldout = results["Held-out test"]
    print("\n7. Held-out test performance (Section 4.4.5)")
    print(
        "coverage: "
        f"{heldout.scored_rows} scored cases; "
        f"{heldout.safe_cases} safe; {heldout.unsafe_cases} unsafe"
    )
    print(
        "headline: "
        f"safety accuracy {percent(heldout.safety_correct, heldout.scored_rows)}; "
        f"unsafe recall {percent(heldout.unsafe_detected, heldout.unsafe_cases)}; "
        f"strict node accuracy {percent(heldout.node_correct, heldout.scored_rows)}"
    )
    print(
        "post-hoc affect/emotional sensitivity analysis: "
        f"{percent(heldout.node_correct, heldout.scored_rows)} -> "
        f"{percent(heldout.collapsed_node_correct, heldout.scored_rows)}"
    )

    print("\n7a. Held-out per-node results")
    print("reference node | N | safety correct | node correct")
    for node, (node_correct, total) in heldout.per_node.items():
        safety_correct = heldout.per_node_safety_correct[node]
        print(
            f"{node} | {total} | {safety_correct}/{total} | "
            f"{node_correct}/{total}"
        )

    confusions = node_confusion_pairs(heldout_rows)
    print("\n7b. Held-out strict-node confusion pairs")
    print(f"total strict-node errors: {sum(confusions.values())}")
    for (expected, predicted), count in confusions.most_common():
        print(f"{expected} -> {predicted}: {count}")

    print("\n7c. Held-out binary safety errors")
    for row in safety_error_rows(heldout_rows):
        error_type = (
            "false alarm" if as_bool(row["gt_safe"]) is True else "unsafe miss"
        )
        print(
            f"{error_type}: {row['test']} | "
            f"{row['gt_node']} -> {row['pred_node']} | "
            f"{row['user_input']}"
        )


def main() -> None:
    print_results(calculate_all())


if __name__ == "__main__":
    main()
