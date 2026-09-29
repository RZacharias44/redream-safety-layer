"""Reproduce the dataset and evaluation-design counts in Chapter 4.1--4.2.

Run from the repository root:

    uv run python experiments/evaluation_statistics/chapter_4/sections_4_1_2_evaluation_design.py

All counts come from the row-level JSON artifacts used by the evaluation.  The
annotation-log summary is checked against its detailed review arrays so that the
reported review coverage is not accepted as an unexplained prose constant.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEV_FILE = ROOT / "data/benchmarks/dev_dataset.json"
TEST_FILE = ROOT / "data/benchmarks/test_dataset.json"
MULTI_MALADAPTIVE_FILE = ROOT / "data/benchmarks/multi_maladaptive_dataset.json"
ANNOTATION_LOG_FILE = ROOT / "data/benchmarks/test_dataset_annotation_log.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_suites(suites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"suite": suite["suite"], **test}
        for suite in suites
        for test in suite["tests"]
    ]


def dataset_summary(suites: list[dict[str, Any]]) -> dict[str, Any]:
    rows = flatten_suites(suites)
    scored = [row for row in rows if row.get("expected_safe") is not None]
    return {
        "n_scenarios": len(suites),
        "n_cases": len(rows),
        "n_scored": len(scored),
        "n_boundary": sum(row.get("test_type") == "boundary" for row in rows),
        "n_clean": sum(row.get("test_type") == "clean" for row in scored),
        "n_multipart": sum(row.get("test_type") == "multipart" for row in scored),
        "n_safe": sum(row.get("expected_safe") is True for row in scored),
        "n_unsafe": sum(row.get("expected_safe") is False for row in scored),
        "nodes": sorted({row["expected_node"] for row in scored}),
        "per_node": Counter(row["expected_node"] for row in scored),
    }


def retained_test_suites(suites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the chapter's provider-refusal exclusion rule."""
    return [suite for suite in suites if "buried alive" not in suite["suite"].lower()]


def annotation_review_summary(
    log: dict[str, Any], retained_rows: list[dict[str, Any]]
) -> dict[str, int]:
    summary = log["summary"]
    breakdown = summary["breakdown"]
    detailed_boundary = len(log["boundary_cases"])
    detailed_affect = len(log["affect_expression_cases"]["cases"])
    detailed_violent = len(log["violent_revenge_cases"]["cases"])
    detailed_spotcheck = (
        log["spot_check_cases"]["total"] + log["spot_check_cases_round2"]["total"]
    )

    assert breakdown["boundary_cases"]["reviewed"] == detailed_boundary
    assert breakdown["affect_expression"]["reviewed"] == detailed_affect
    assert breakdown["violent_revenge"]["reviewed"] == detailed_violent
    assert breakdown["spot_check"]["reviewed"] == detailed_spotcheck

    def normalized(text: str) -> str:
        return text.replace("’", "'").replace("“", '"').replace("”", '"')

    by_input = {normalized(row["input"]): row for row in retained_rows}
    explicitly_recorded = (
        log["affect_expression_cases"]["cases"]
        + log["violent_revenge_cases"]["cases"]
    )
    explicit_label_changes = sum(
        by_input[normalized(case["input"])]["expected_node"]
        != case["ramon_label"]["node"]
        or by_input[normalized(case["input"])]["expected_safe"]
        != case["ramon_label"]["safe"]
        for case in explicitly_recorded
    )
    assert "60/60 spot-check agreement" in " ".join(summary["key_findings"])

    return {
        "boundary": detailed_boundary,
        "affect_expression": detailed_affect,
        "violent_revenge": detailed_violent,
        "spot_check": detailed_spotcheck,
        "reviewed_scored": detailed_affect + detailed_violent + detailed_spotcheck,
        "label_changes": explicit_label_changes,
    }


def calculate_all() -> dict[str, Any]:
    dev_suites = read_json(DEV_FILE)
    test_suites = read_json(TEST_FILE)
    retained = retained_test_suites(test_suites)
    excluded = [suite for suite in test_suites if suite not in retained]
    multi = dataset_summary(read_json(MULTI_MALADAPTIVE_FILE))
    return {
        "development": dataset_summary(dev_suites),
        "heldout_initial": dataset_summary(test_suites),
        "heldout_retained": dataset_summary(retained),
        "excluded_scenarios": len(excluded),
        "excluded_cases": sum(len(suite["tests"]) for suite in excluded),
        "multi_maladaptive": multi,
        "author_review": annotation_review_summary(
            read_json(ANNOTATION_LOG_FILE), flatten_suites(retained)
        ),
    }


def main() -> None:
    results = calculate_all()
    dev = results["development"]
    test = results["heldout_retained"]
    review = results["author_review"]

    print("Chapter 4.1--4.2 -- Evaluation Design and Dataset Rationale")
    print("\n1. Development benchmark")
    print(
        f"{dev['n_cases']} cases across {dev['n_scenarios']} scenarios; "
        f"{dev['n_scored']} scored ({dev['n_safe']} safe, {dev['n_unsafe']} unsafe); "
        f"{dev['n_boundary']} boundary; {len(dev['nodes'])} ontology nodes"
    )
    print("\n2. Held-out pool and exclusion")
    print(
        f"initial: {results['heldout_initial']['n_cases']} cases across "
        f"{results['heldout_initial']['n_scenarios']} scenarios"
    )
    print(
        f"excluded: {results['excluded_cases']} cases in "
        f"{results['excluded_scenarios']} truncated scenario"
    )
    print(
        f"retained: {test['n_cases']} cases across {test['n_scenarios']} themes; "
        f"{test['n_scored']} scored and {test['n_boundary']} boundary"
    )
    print(
        f"scored composition: {test['n_clean']} clean, {test['n_multipart']} multipart; "
        f"{test['n_safe']} safe, {test['n_unsafe']} unsafe; {len(test['nodes'])} nodes"
    )
    print("\n3. Supplementary multi-maladaptive slice")
    print(f"cases: {results['multi_maladaptive']['n_cases']}")
    print("\n4. Targeted internal author review")
    print(
        f"boundary {review['boundary']}; affect expression {review['affect_expression']}; "
        f"violent revenge {review['violent_revenge']}; stratified remainder {review['spot_check']}"
    )
    print(
        f"reviewed scored cases: {review['reviewed_scored']}; "
        f"label changes retained in dataset: {review['label_changes']}"
    )


if __name__ == "__main__":
    main()
