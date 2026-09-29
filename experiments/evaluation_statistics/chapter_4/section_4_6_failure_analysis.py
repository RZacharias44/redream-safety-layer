"""Statistics reported in Chapter 4.6: Failure Analysis and Edge Cases.

Run from the repository root:

    uv run python experiments/evaluation_statistics/chapter_4/section_4_6_failure_analysis.py

The script derives the reported error counts and boundary-case comparisons from
the row-level development benchmark, ablation, and KL annotation CSV files.
Boundary comparisons are descriptive because those cases have no fixed target.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from experiments.evaluation_statistics.consistency_votes import derive_vote_verdicts


ROOT = Path(__file__).resolve().parents[3]
S1_FILE = ROOT / "data/benchmarks/dev_s1_baseline.csv"
S2_CONSISTENCY_FILE = (
    ROOT / "data/benchmarks/dev_s2_consistency_n5.csv"
)
NO_SEGMENTATION_FILE = (
    ROOT / "data/benchmarks/dev_s1_ablation_no_segmentation.csv"
)
KL_FILE = ROOT / "data/expert_analysis/kl_2026_06/spotcheck_merged.csv"
AG_FILE = ROOT / "data/expert_analysis/ag_2026_07/spotcheck_merged.csv"

EXPERT_DATA_NOTICE = (
    "The row-level expert annotation tables under data/expert_analysis/ are not "
    "present: they are personal data of the raters and are distributed only with "
    "their consent. The aggregate results are in "
    "data/evaluation_statistics/chapter4/chapter4_statistics.json ('expert_spotcheck')."
)
TEST_DATASET_FILE = ROOT / "data/benchmarks/test_dataset.json"

MALADAPTIVE_NODES = {
    "AVOIDANCE",
    "INTERRUPTION",
    "VIOLENT_REVENGE",
    "SUPPRESSION",
    "TRAUMA_REPLAY",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a row-level CSV file."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_consistency_rows(path: Path) -> list[dict[str, str]]:
    """Read a consistency run, scored on its majority-vote verdict.

    A consistency CSV stores the temperature-0 pipeline prediction in
    ``pred_safe``/``pred_node`` and the per-segment votes separately, so the
    error set of the condition Chapter 4.4 reports only appears after the votes
    are triaged. Substituting the vote columns keeps every downstream error
    analysis in this module consistent with the reported condition.
    """
    voted, _diagnostics = derive_vote_verdicts(read_rows(path))
    return [
        {**row, "pred_safe": row["vote_safe"], "pred_node": row["vote_node"]}
        for row in voted
    ]


def as_bool(value: str) -> bool | None:
    """Parse CSV booleans; return None for an empty unscored field."""
    text = value.strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    if text == "":
        return None
    raise ValueError(f"Unexpected boolean value: {value!r}")


def scored_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Exclude unscored boundary rows from benchmark metrics."""
    return [row for row in rows if as_bool(row["gt_safe"]) is not None]


def safety_errors(rows: list[dict[str, str]]) -> dict[str, int]:
    """Count false alarms and missed unsafe cases."""
    rows = scored_rows(rows)
    false_alarms = sum(
        as_bool(row["gt_safe"]) is True and as_bool(row["pred_safe"]) is False
        for row in rows
    )
    unsafe_misses = sum(
        as_bool(row["gt_safe"]) is False and as_bool(row["pred_safe"]) is True
        for row in rows
    )
    return {
        "n": len(rows),
        "total": false_alarms + unsafe_misses,
        "false_alarms": false_alarms,
        "unsafe_misses": unsafe_misses,
    }


def node_error_patterns(rows: list[dict[str, str]]) -> dict[str, object]:
    """Summarize strict primary-node confusion pairs."""
    errors = [
        row
        for row in scored_rows(rows)
        if row["gt_node"] != row["pred_node"]
    ]
    patterns = Counter((row["gt_node"], row["pred_node"]) for row in errors)
    top_three = patterns.most_common(3)
    top_three_count = sum(count for _, count in top_three)
    return {
        "errors": len(errors),
        "patterns": len(patterns),
        "top_three": top_three,
        "top_three_count": top_three_count,
        "top_three_percent": 100 * top_three_count / len(errors),
        "remaining_patterns": len(patterns) - len(top_three),
    }


def false_alarm_mechanisms(rows: list[dict[str, str]]) -> dict[str, object]:
    """Reproduce the two-way grouping of the eight S1 false alarms.

    The four segment-level cases have a TRAUMA_REPLAY primary prediction even
    though their expected node is adaptive. The other four express plausible
    safe/unsafe category-boundary interpretations.
    """
    false_alarms = [
        row
        for row in scored_rows(rows)
        if as_bool(row["gt_safe"]) is True
        and as_bool(row["pred_safe"]) is False
    ]
    segment_level = [
        row for row in false_alarms if row["pred_node"] == "TRAUMA_REPLAY"
    ]
    category_boundary = [
        row for row in false_alarms if row["pred_node"] != "TRAUMA_REPLAY"
    ]
    boundary_pairs = Counter(
        (row["gt_node"], row["pred_node"]) for row in category_boundary
    )
    return {
        "total": len(false_alarms),
        "segment_level": len(segment_level),
        "category_boundary": len(category_boundary),
        "category_boundary_pairs": boundary_pairs,
    }


def no_segmentation_misses(rows: list[dict[str, str]]) -> dict[str, object]:
    """Count unsafe misses and their expected nodes in the no-segmentation run."""
    misses = [
        row
        for row in scored_rows(rows)
        if as_bool(row["gt_safe"]) is False
        and as_bool(row["pred_safe"]) is True
    ]
    return {
        "total": len(misses),
        "expected_nodes": Counter(row["gt_node"] for row in misses),
    }


def completed_kl_boundary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Select KL boundary rows with both expert judgments completed."""
    return [
        row
        for row in rows
        if row["source_bucket"] == "boundary"
        and as_bool(row["expert_safe"]) is not None
        and row["expert_node"].strip()
    ]


def boundary_classifier_expert_agreement(
    rows: list[dict[str, str]],
) -> dict[str, int]:
    """Compute descriptive classifier--KL agreement on boundary rows."""
    rows = completed_kl_boundary_rows(rows)
    return {
        "n": len(rows),
        "raw_safety": sum(
            as_bool(row["classifier_safe"]) == as_bool(row["expert_safe"])
            for row in rows
        ),
        "node_derived_safety": sum(
            (row["classifier_node"] in MALADAPTIVE_NODES)
            == (row["expert_node"] in MALADAPTIVE_NODES)
            for row in rows
        ),
        "strict_node": sum(
            row["classifier_node"] == row["expert_node"] for row in rows
        ),
    }


def unchanged_replay_disagreements(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Find the unchanged-replay pattern described in the boundary analysis.

    KL separately marked these cases unsafe but chose the neutral
    NARRATIVE_SETTING node, while the classifier chose TRAUMA_REPLAY.
    """
    return [
        row
        for row in completed_kl_boundary_rows(rows)
        if as_bool(row["expert_safe"]) is False
        and row["expert_node"] == "NARRATIVE_SETTING"
        and row["classifier_node"] == "TRAUMA_REPLAY"
    ]


def boundary_type_by_input(path: Path = TEST_DATASET_FILE) -> dict[str, str]:
    """Index retained boundary proposals by their generator boundary type."""
    suites = json.loads(path.read_text(encoding="utf-8"))
    return {
        test["input"]: test["boundary"]
        for suite in suites
        if "buried alive" not in suite["suite"].lower()
        for test in suite["tests"]
        if test.get("test_type") == "boundary"
    }


def boundary_theme_summary(
    rows: list[dict[str, str]],
    type_index: dict[str, str],
) -> dict[str, dict[str, object]]:
    """Summarize all five boundary themes reported in Table 4.15."""
    theme_for_type = {
        "description_vs_replay": "unchanged replay versus narrative description",
        "passivity": "freezing and passivity",
        "regulation_vs_denial": "regulation versus suppression",
        "active_vs_passive": "active coping or scene change versus avoidance or interruption",
        "action_vs_escape": "active coping or scene change versus avoidance or interruption",
        "proportionality": "self-protection versus violent revenge",
    }
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["source_bucket"] != "boundary" or not row["expert_node"].strip():
            continue
        boundary_type = type_index[row["case_user_input"]]
        theme = theme_for_type[boundary_type]
        grouped.setdefault(theme, []).append(row)

    return {
        theme: {
            "n": len(group),
            "unsafe": sum(as_bool(row["expert_safe"]) is False for row in group),
            "expert_nodes": Counter(row["expert_node"] for row in group),
            "classifier_nodes": Counter(row["classifier_node"] for row in group),
        }
        for theme, group in grouped.items()
    }


def percent(count: int, total: int) -> str:
    """Format a percentage with visible numerator and denominator."""
    return f"{100 * count / total:.1f}% ({count}/{total})"


def main() -> None:
    s1_rows = read_rows(S1_FILE)
    s2_consistency_rows = read_consistency_rows(S2_CONSISTENCY_FILE)
    no_segmentation_rows = read_rows(NO_SEGMENTATION_FILE)

    print("Chapter 4.6 -- Failure Analysis and Edge Cases")

    print("\n1. Binary safety errors")
    for name, rows in (
        ("Full S1 pipeline", s1_rows),
        ("S2 with classification consistency", s2_consistency_rows),
    ):
        result = safety_errors(rows)
        print(
            f"{name}: {result['total']} total; "
            f"{result['false_alarms']} false alarms; "
            f"{result['unsafe_misses']} unsafe misses"
        )

    node_errors = node_error_patterns(s1_rows)
    print("\n2. Full S1 strict primary-node errors")
    print(
        f"{node_errors['errors']} errors across {node_errors['patterns']} patterns"
    )
    for (expected, predicted), count in node_errors["top_three"]:
        print(f"{expected} -> {predicted}: {count}")
    print(
        f"top three: {node_errors['top_three_count']}/"
        f"{node_errors['errors']} ({node_errors['top_three_percent']:.1f}%); "
        f"remaining patterns: {node_errors['remaining_patterns']}"
    )

    false_alarms = false_alarm_mechanisms(s1_rows)
    print("\n3. Full S1 false-alarm mechanisms")
    print(
        f"total: {false_alarms['total']}; "
        f"category-boundary interpretations: {false_alarms['category_boundary']}; "
        f"segment-level TRAUMA_REPLAY labels: {false_alarms['segment_level']}"
    )
    for pair, count in false_alarms["category_boundary_pairs"].most_common():
        print(f"{pair[0]} -> {pair[1]}: {count}")

    no_segmentation = no_segmentation_misses(no_segmentation_rows)
    print("\n4. No-segmentation unsafe misses")
    print(f"total: {no_segmentation['total']}")
    for node, count in no_segmentation["expected_nodes"].most_common():
        print(f"{node}: {count}")

    if not (KL_FILE.exists() and AG_FILE.exists()):
        print("\n5-7. Expert-annotation sections skipped")
        print(EXPERT_DATA_NOTICE)
        return
    kl_rows = read_rows(KL_FILE)
    ag_rows = read_rows(AG_FILE)

    boundary = boundary_classifier_expert_agreement(kl_rows)
    n = boundary["n"]
    print("\n5. KL boundary rows, descriptive classifier--expert agreement")
    print(f"raw safety: {percent(boundary['raw_safety'], n)}")
    print(f"node-derived safety: {percent(boundary['node_derived_safety'], n)}")
    print(f"strict primary node: {percent(boundary['strict_node'], n)}")

    unchanged = unchanged_replay_disagreements(kl_rows)
    print("\n6. Unchanged-replay node-representation disagreements")
    print(f"total: {len(unchanged)}")
    for row in unchanged:
        print(f"{row['case_id']}: {row['source_test_description']}")

    type_index = boundary_type_by_input()
    print("\n7. Boundary-theme synthesis")
    for rater, rows in (("KL", kl_rows), ("AG", ag_rows)):
        themes = boundary_theme_summary(rows, type_index)
        print(f"{rater}: {sum(value['n'] for value in themes.values())} completed boundary rows")
        for theme, values in themes.items():
            print(
                f"{theme}: n={values['n']}; unsafe={values['unsafe']}; "
                f"expert nodes={dict(values['expert_nodes'])}; "
                f"classifier nodes={dict(values['classifier_nodes'])}"
            )


if __name__ == "__main__":
    main()
