"""Reproduce the blinded system-level expert results in Chapter 4.8.

Run from the repository root:

    uv run python experiments/evaluation_statistics/chapter_4/section_4_8_system_expert_evaluation.py

The calculations use the merged row-level exports.  Condition summaries,
within-case changes, shared-response agreement, and qualitative-note counts are
all derived here rather than copied from the thesis tables.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
FILES = {
    "KL": ROOT / "data/expert_analysis/kl_2026_06/system_eval_merged.csv",
    "AG": ROOT / "data/expert_analysis/ag_2026_07/system_eval_merged.csv",
}

EXPERT_DATA_NOTICE = (
    "The row-level expert annotation tables under data/expert_analysis/ are not "
    "present: they are personal data of the raters and are distributed only with "
    "their consent. The aggregate results are in "
    "data/evaluation_statistics/chapter4/chapter4_statistics.json ('system_expert_evaluation')."
)

DIMENSIONS = {
    "Clinical acceptability": (
        "clinical_acceptability_score",
        "expert_clinically_acceptable",
        True,
    ),
    "Redirect quality": ("redirect_quality_score", "expert_redirect_quality", True),
    "Distress validation": (
        "validation_score",
        "expert_validates_distress_appropriately",
        True,
    ),
    "Maladaptive endorsement": (
        "maladaptive_endorsement_score",
        "expert_maladaptive_endorsement",
        False,
    ),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def complete(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["expert_clinically_acceptable"].strip()]


def distribution(rows: list[dict[str, str]], field: str) -> Counter[str]:
    return Counter(row[field] or "n/a" for row in rows)


def condition_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in ("control", "treatment"):
        group = [row for row in complete(rows) if row["condition"] == condition]
        clinical_scores = [int(row["clinical_acceptability_score"]) for row in group]
        result[condition] = {
            "n": len(group),
            "clinical_acceptability": distribution(
                group, "expert_clinically_acceptable"
            ),
            "mean_clinical_acceptability": sum(clinical_scores) / len(clinical_scores),
            "redirect_quality": distribution(group, "expert_redirect_quality"),
            "maladaptive_endorsement": distribution(
                group, "expert_maladaptive_endorsement"
            ),
        }
    return result


def paired_rows(rows: list[dict[str, str]]) -> list[tuple[dict[str, str], dict[str, str]]]:
    by_case: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in complete(rows):
        by_case[row["case_id"]][row["condition"]] = row
    return [
        (conditions["control"], conditions["treatment"])
        for conditions in by_case.values()
        if {"control", "treatment"} <= conditions.keys()
    ]


def paired_dimension_metrics(
    pairs: list[tuple[dict[str, str], dict[str, str]]],
    score_field: str,
    higher_is_better: bool,
) -> dict[str, int | float]:
    raw_deltas = [
        int(treatment[score_field]) - int(control[score_field])
        for control, treatment in pairs
        if control[score_field].strip() and treatment[score_field].strip()
    ]
    oriented = [delta if higher_is_better else -delta for delta in raw_deltas]
    return {
        "n": len(oriented),
        "better": sum(delta > 0 for delta in oriented),
        "same": sum(delta == 0 for delta in oriented),
        "worse": sum(delta < 0 for delta in oriented),
        "mean_change": sum(oriented) / len(oriented),
    }


def paired_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    pairs = paired_rows(rows)
    return {
        label: paired_dimension_metrics(pairs, score_field, higher_is_better)
        for label, (score_field, _category_field, higher_is_better) in DIMENSIONS.items()
    }


def shared_response_metrics(
    ag_rows: list[dict[str, str]], kl_rows: list[dict[str, str]]
) -> dict[str, Any]:
    ag = {row["shuffled_row_id"]: row for row in complete(ag_rows)}
    kl = {row["shuffled_row_id"]: row for row in complete(kl_rows)}
    shared_ids = sorted(set(ag) & set(kl))
    result: dict[str, Any] = {"n": len(shared_ids)}
    for label, (_score_field, category_field, _higher) in DIMENSIONS.items():
        result[label] = sum(
            ag[row_id][category_field] == kl[row_id][category_field]
            for row_id in shared_ids
        )
    result["AG strong redirect"] = sum(
        ag[row_id]["expert_redirect_quality"] == "strong" for row_id in shared_ids
    )
    result["KL strong redirect"] = sum(
        kl[row_id]["expert_redirect_quality"] == "strong" for row_id in shared_ids
    )
    result["AG appropriate validation"] = sum(
        ag[row_id]["expert_validates_distress_appropriately"] == "yes"
        for row_id in shared_ids
    )
    result["KL appropriate validation"] = sum(
        kl[row_id]["expert_validates_distress_appropriately"] == "yes"
        for row_id in shared_ids
    )
    return result


def qualitative_note_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    rows = complete(rows)
    return {
        "adaptive_or_partly_adaptive": sum(
            row["adequacy_note_signal"].lower() == "true" for row in rows
        ),
        "false_positive_redirect": sum(
            row["false_positive_note_signal"].lower() == "true" for row in rows
        ),
    }


def calculate_all() -> dict[str, Any]:
    rows = {rater: read_rows(path) for rater, path in FILES.items()}
    return {
        "rows": rows,
        "completion": {
            rater: {
                "queue": len(rater_rows),
                "complete": len(complete(rater_rows)),
                "pairs": len(paired_rows(rater_rows)),
            }
            for rater, rater_rows in rows.items()
        },
        "conditions": {
            rater: condition_summary(rater_rows) for rater, rater_rows in rows.items()
        },
        "paired": {
            rater: paired_summary(rater_rows) for rater, rater_rows in rows.items()
        },
        "shared": shared_response_metrics(rows["AG"], rows["KL"]),
        "kl_notes": qualitative_note_counts(rows["KL"]),
    }


def main() -> None:
    if not all(path.exists() for path in FILES.values()):
        print("Chapter 4.8 -- System-Level and Expert Evaluation")
        print(EXPERT_DATA_NOTICE)
        return
    results = calculate_all()
    print("Chapter 4.8 -- System-Level and Expert Evaluation")
    print("\n1. Completion and matched pairs")
    for rater, values in results["completion"].items():
        print(
            f"{rater}: {values['complete']}/{values['queue']} completed; "
            f"{values['pairs']} complete matched pairs"
        )

    print("\n2. Response quality by condition")
    for rater, conditions in results["conditions"].items():
        for condition, values in conditions.items():
            print(
                f"{rater} {condition}: n={values['n']}; "
                f"clinical={dict(values['clinical_acceptability'])}; "
                f"mean={values['mean_clinical_acceptability']:.2f}; "
                f"redirect={dict(values['redirect_quality'])}; "
                f"endorsement={dict(values['maladaptive_endorsement'])}"
            )

    print("\n3. Within-case safety-augmented versus control")
    for rater, dimensions in results["paired"].items():
        for label, values in dimensions.items():
            print(
                f"{rater} {label}: n={values['n']}; better={values['better']}; "
                f"same={values['same']}; worse={values['worse']}; "
                f"mean change={values['mean_change']:+.2f}"
            )

    shared = results["shared"]
    print("\n4. Inter-expert exact agreement on shared response rows")
    for label in DIMENSIONS:
        print(
            f"{label}: {shared[label]}/{shared['n']} "
            f"({100 * shared[label] / shared['n']:.1f}%)"
        )
    print(
        f"strong redirect: AG {shared['AG strong redirect']}, "
        f"KL {shared['KL strong redirect']}"
    )
    print(
        f"appropriate distress validation: AG {shared['AG appropriate validation']}, "
        f"KL {shared['KL appropriate validation']}"
    )

    notes = results["kl_notes"]
    print("\n5. KL qualitative-note flags")
    print(
        f"adaptive or partly adaptive move: {notes['adaptive_or_partly_adaptive']}; "
        f"explicit false-positive redirect: {notes['false_positive_redirect']}"
    )


if __name__ == "__main__":
    main()
