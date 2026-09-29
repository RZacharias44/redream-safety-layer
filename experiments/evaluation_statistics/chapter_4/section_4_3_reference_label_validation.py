"""Statistics for the held-out reference-label validation in Chapter 4.

Run from the repository root:

    uv run python experiments/evaluation_statistics/chapter_4/section_4_3_reference_label_validation.py

This script reads the two row-level expert CSV files and prints the results in
the order they appear in the revised evaluation chapter. Boundary cases are
kept out of reference-label agreement because they have no fixed gold label.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
AG_FILE = ROOT / "data/expert_analysis/ag_2026_07/spotcheck_merged.csv"
KL_FILE = ROOT / "data/expert_analysis/kl_2026_06/spotcheck_merged.csv"

EXPERT_DATA_NOTICE = (
    "The row-level expert annotation tables under data/expert_analysis/ are not "
    "present: they are personal data of the raters and are distributed only with "
    "their consent. The aggregate results are in "
    "data/evaluation_statistics/chapter4/chapter4_statistics.json ('expert_spotcheck')."
)

MALADAPTIVE_NODES = {
    "AVOIDANCE",
    "INTERRUPTION",
    "VIOLENT_REVENGE",
    "SUPPRESSION",
    "TRAUMA_REPLAY",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read one expert's row-level annotation table."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def as_bool(value: str) -> bool | None:
    """Convert CSV text to True/False; return None for an empty field."""
    value = value.strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def completed(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep rows where the expert supplied both required judgments."""
    return [
        row
        for row in rows
        if as_bool(row["expert_safe"]) is not None and row["expert_node"].strip()
    ]


def select_slice(rows: list[dict[str, str]], slice_name: str) -> list[dict[str, str]]:
    """Select one of the three slices reported in the first results table."""
    rows = completed(rows)
    if slice_name == "all":
        return rows
    if slice_name == "primary":
        return [row for row in rows if row["source_bucket"] != "boundary"]
    if slice_name == "boundary":
        return [row for row in rows if row["source_bucket"] == "boundary"]
    raise ValueError(f"Unknown slice: {slice_name}")


def compute_spot_check_agreement(rows: list[dict[str, str]]) -> dict[str, int]:
    """Compute expert agreement with scored dataset reference labels.

    Raw safety agreement:
        expert_safe == reference_safe

    Node-derived safety agreement:
        expert selected a maladaptive node == reference is unsafe

    Strict node agreement:
        expert_node == reference_node
    """
    rows = completed(rows)
    assert all(row["source_bucket"] != "boundary" for row in rows)

    raw_safety = sum(
        as_bool(row["expert_safe"]) == as_bool(row["reference_safe"])
        for row in rows
    )
    node_derived_safety = sum(
        (row["expert_node"] in MALADAPTIVE_NODES)
        == (as_bool(row["reference_safe"]) is False)
        for row in rows
    )
    strict_node = sum(
        row["expert_node"] == row["reference_node"]
        for row in rows
    )
    adequacy_penalties = sum(
        as_bool(row["reference_safe"]) is True
        and as_bool(row["expert_safe"]) is False
        and row["expert_node"] not in MALADAPTIVE_NODES
        for row in rows
    )

    return {
        "n": len(rows),
        "raw_safety": raw_safety,
        "node_derived_safety": node_derived_safety,
        "strict_node": strict_node,
        "adequacy_penalties": adequacy_penalties,
    }


def compute_classifier_agreement(rows: list[dict[str, str]]) -> dict[str, int]:
    """Compute classifier agreement on scored expert-completed rows."""
    rows = completed(rows)
    assert all(row["source_bucket"] != "boundary" for row in rows)

    classifier_reference_safety = sum(
        as_bool(row["classifier_safe"]) == as_bool(row["reference_safe"])
        for row in rows
    )
    classifier_reference_node = sum(
        row["classifier_node"] == row["reference_node"]
        for row in rows
    )
    classifier_expert_raw_safety = sum(
        as_bool(row["classifier_safe"]) == as_bool(row["expert_safe"])
        for row in rows
    )
    classifier_expert_node_derived_safety = sum(
        (row["classifier_node"] in MALADAPTIVE_NODES)
        == (row["expert_node"] in MALADAPTIVE_NODES)
        for row in rows
    )
    classifier_expert_strict_node = sum(
        row["classifier_node"] == row["expert_node"]
        for row in rows
    )

    return {
        "n": len(rows),
        "classifier_reference_safety": classifier_reference_safety,
        "classifier_reference_node": classifier_reference_node,
        "classifier_expert_raw_safety": classifier_expert_raw_safety,
        "classifier_expert_node_derived_safety": classifier_expert_node_derived_safety,
        "classifier_expert_strict_node": classifier_expert_strict_node,
    }


def cohen_kappa(pairs: list[tuple[str | bool, str | bool]]) -> float:
    """Compute Cohen's kappa: (observed - chance) / (1 - chance)."""
    n = len(pairs)
    categories = {value for pair in pairs for value in pair}

    observed = sum(left == right for left, right in pairs) / n
    chance = sum(
        (sum(left == category for left, _ in pairs) / n)
        * (sum(right == category for _, right in pairs) / n)
        for category in categories
    )
    return (observed - chance) / (1 - chance)


def compute_interrater_agreement(
    ag_rows: list[dict[str, str]],
    kl_rows: list[dict[str, str]],
) -> dict[str, float | int]:
    """Compare AG and KL on cases for which both completed both fields."""
    ag = {row["case_id"]: row for row in completed(ag_rows)}
    kl = {row["case_id"]: row for row in completed(kl_rows)}
    shared_ids = sorted(set(ag) & set(kl))

    raw_safety_pairs = [
        (as_bool(ag[case]["expert_safe"]), as_bool(kl[case]["expert_safe"]))
        for case in shared_ids
    ]
    node_derived_pairs = [
        (
            ag[case]["expert_node"] in MALADAPTIVE_NODES,
            kl[case]["expert_node"] in MALADAPTIVE_NODES,
        )
        for case in shared_ids
    ]
    strict_node_pairs = [
        (ag[case]["expert_node"], kl[case]["expert_node"])
        for case in shared_ids
    ]

    return {
        "n": len(shared_ids),
        "raw_safety": sum(left == right for left, right in raw_safety_pairs),
        "raw_safety_kappa": cohen_kappa(raw_safety_pairs),
        "node_derived_safety": sum(left == right for left, right in node_derived_pairs),
        "node_derived_safety_kappa": cohen_kappa(node_derived_pairs),
        "strict_node": sum(left == right for left, right in strict_node_pairs),
        "strict_node_kappa": cohen_kappa(strict_node_pairs),
    }


def percent(count: int, n: int) -> str:
    """Format one thesis result with its visible numerator and denominator."""
    return f"{100 * count / n:.1f}% ({count}/{n})"


def print_spot_check_table(ag_rows: list[dict[str, str]], kl_rows: list[dict[str, str]]) -> None:
    print("\n1. Expert agreement with reference labels, scored non-boundary rows")
    print("rater | complete | raw safety | node-derived safety | strict node | adequacy penalties")
    for rater, rows in (("KL", kl_rows), ("AG", ag_rows)):
        result = compute_spot_check_agreement(select_slice(rows, "primary"))
        n = result["n"]
        print(
            f"{rater} | {n} | "
            f"{percent(result['raw_safety'], n)} | "
            f"{percent(result['node_derived_safety'], n)} | "
            f"{percent(result['strict_node'], n)} | "
            f"{percent(result['adequacy_penalties'], n)}"
        )


def print_interrater_table(ag_rows: list[dict[str, str]], kl_rows: list[dict[str, str]]) -> None:
    result = compute_interrater_agreement(
        select_slice(ag_rows, "primary"),
        select_slice(kl_rows, "primary"),
    )
    n = int(result["n"])
    print("\n2. Direct AG-KL agreement, shared scored non-boundary rows")
    for label, count_key, kappa_key in (
        ("raw safety", "raw_safety", "raw_safety_kappa"),
        ("node-derived safety", "node_derived_safety", "node_derived_safety_kappa"),
        ("strict primary node", "strict_node", "strict_node_kappa"),
    ):
        print(
            f"{label}: {percent(int(result[count_key]), n)}, "
            f"Cohen's kappa = {result[kappa_key]:.3f}"
        )


def print_classifier_table(ag_rows: list[dict[str, str]], kl_rows: list[dict[str, str]]) -> None:
    print("\n3. Classifier agreement with reference and experts, scored non-boundary rows")
    print("rater | N | classifier-reference safety | classifier-reference node | "
          "classifier-expert raw safety | classifier-expert node-derived safety | "
          "classifier-expert strict node")
    for rater, rows in (("KL", kl_rows), ("AG", ag_rows)):
        result = compute_classifier_agreement(select_slice(rows, "primary"))
        n = result["n"]
        fields = (
            "classifier_reference_safety",
            "classifier_reference_node",
            "classifier_expert_raw_safety",
            "classifier_expert_node_derived_safety",
            "classifier_expert_strict_node",
        )
        values = " | ".join(percent(result[field], n) for field in fields)
        print(f"{rater} | {n} | {values}")


def compute_narrative_setting_result(kl_rows: list[dict[str, str]]) -> dict[str, int]:
    """Compute the NARRATIVE_SETTING result described after the classifier table."""
    rows = [
        row
        for row in completed(kl_rows)
        if row["reference_node"] == "NARRATIVE_SETTING"
    ]
    return {
        "n": len(rows),
        "marked_unsafe": sum(as_bool(row["expert_safe"]) is False for row in rows),
        "node_derived_safety_agreement": sum(
            row["expert_node"] not in MALADAPTIVE_NODES for row in rows
        ),
    }


def print_narrative_setting_result(kl_rows: list[dict[str, str]]) -> None:
    result = compute_narrative_setting_result(kl_rows)
    n = result["n"]
    print("\n4. KL NARRATIVE_SETTING observation")
    print(f"marked unsafe: {percent(result['marked_unsafe'], n)}")
    print(
        "node-derived safety agreement: "
        f"{percent(result['node_derived_safety_agreement'], n)}"
    )


def print_kl_confidence_table(kl_rows: list[dict[str, str]]) -> None:
    print("\n5. KL agreement by confidence, primary non-boundary rows")
    primary = select_slice(kl_rows, "primary")
    for confidence in ("high", "medium", "low"):
        rows = [row for row in primary if row["expert_confidence"] == confidence]
        result = compute_spot_check_agreement(rows)
        n = result["n"]
        print(
            f"{confidence}: N={n}; raw {percent(result['raw_safety'], n)}; "
            f"node-derived {percent(result['node_derived_safety'], n)}; "
            f"strict node {percent(result['strict_node'], n)}; "
            f"adequacy penalties {percent(result['adequacy_penalties'], n)}"
        )


def main() -> None:
    if not (AG_FILE.exists() and KL_FILE.exists()):
        print("Chapter 4.3 -- Held-Out Reference-Label Validation")
        print(EXPERT_DATA_NOTICE)
        return
    ag_rows = read_rows(AG_FILE)
    kl_rows = read_rows(KL_FILE)

    # Simple checks to catch an accidental wrong or duplicated input file.
    assert len(ag_rows) == len(kl_rows) == 100
    assert len({row["case_id"] for row in ag_rows}) == 100
    assert len({row["case_id"] for row in kl_rows}) == 100

    print("Chapter 4 — Held-Out Reference-Label Validation")
    print(f"AG completed: {len(completed(ag_rows))}/100")
    print(f"KL completed: {len(completed(kl_rows))}/100")

    print_spot_check_table(ag_rows, kl_rows)
    print_interrater_table(ag_rows, kl_rows)
    print_classifier_table(ag_rows, kl_rows)
    print_narrative_setting_result(kl_rows)
    print_kl_confidence_table(kl_rows)


if __name__ == "__main__":
    main()
