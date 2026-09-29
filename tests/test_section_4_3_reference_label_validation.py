import pytest

from experiments.evaluation_statistics.chapter_4.section_4_3_reference_label_validation import (
    AG_FILE,
    KL_FILE,
    completed,
    compute_classifier_agreement,
    compute_interrater_agreement,
    compute_narrative_setting_result,
    compute_spot_check_agreement,
    read_rows,
    select_slice,
)

pytestmark = pytest.mark.skipif(
    not (AG_FILE.exists() and KL_FILE.exists()), reason="row-level expert annotation tables not present"
)


def test_section_4_3_headline_counts():
    ag_rows = read_rows(AG_FILE)
    kl_rows = read_rows(KL_FILE)

    assert len(completed(ag_rows)) == 37
    assert len(completed(kl_rows)) == 98

    ag = compute_spot_check_agreement(select_slice(ag_rows, "primary"))
    assert ag == {
        "n": 30,
        "raw_safety": 26,
        "node_derived_safety": 26,
        "strict_node": 12,
        "adequacy_penalties": 2,
    }

    pairwise = compute_interrater_agreement(
        select_slice(ag_rows, "primary"),
        select_slice(kl_rows, "primary"),
    )
    assert pairwise["n"] == 28
    assert pairwise["raw_safety"] == 26
    assert round(pairwise["raw_safety_kappa"], 3) == 0.858
    assert pairwise["node_derived_safety"] == 23
    assert round(pairwise["node_derived_safety_kappa"], 3) == 0.602
    assert pairwise["strict_node"] == 11
    assert round(pairwise["strict_node_kappa"], 3) == 0.331


def test_section_4_3_slices_have_visible_denominators():
    ag_rows = read_rows(AG_FILE)
    kl_rows = read_rows(KL_FILE)

    assert len(select_slice(ag_rows, "primary")) == 30
    assert len(select_slice(ag_rows, "boundary")) == 7
    assert len(select_slice(kl_rows, "primary")) == 81
    assert len(select_slice(kl_rows, "boundary")) == 17


def test_every_expert_reference_table_row():
    ag_rows = read_rows(AG_FILE)
    kl_rows = read_rows(KL_FILE)
    expected = {
        "KL": (81, 64, 70, 58, 12),
        "AG": (30, 26, 26, 12, 2),
    }

    for rater, rows in (("KL", kl_rows), ("AG", ag_rows)):
        result = compute_spot_check_agreement(select_slice(rows, "primary"))
        assert (
            result["n"],
            result["raw_safety"],
            result["node_derived_safety"],
            result["strict_node"],
            result["adequacy_penalties"],
        ) == expected[rater]


def test_every_classifier_table_row():
    ag_rows = read_rows(AG_FILE)
    kl_rows = read_rows(KL_FILE)
    expected = {
        "KL": (81, 80, 67, 65, 71, 56),
        "AG": (30, 30, 27, 26, 26, 13),
    }

    for rater, rows in (("KL", kl_rows), ("AG", ag_rows)):
        result = compute_classifier_agreement(select_slice(rows, "primary"))
        assert (
            result["n"],
            result["classifier_reference_safety"],
            result["classifier_reference_node"],
            result["classifier_expert_raw_safety"],
            result["classifier_expert_node_derived_safety"],
            result["classifier_expert_strict_node"],
        ) == expected[rater]


def test_narrative_setting_and_confidence_statistics():
    kl_rows = read_rows(KL_FILE)
    assert compute_narrative_setting_result(kl_rows) == {
        "n": 7,
        "marked_unsafe": 7,
        "node_derived_safety_agreement": 7,
    }

    primary = select_slice(kl_rows, "primary")
    expected = {
        "high": (51, 44, 45, 41, 6),
        "medium": (19, 16, 18, 12, 3),
        "low": (11, 4, 7, 5, 3),
    }
    for confidence, counts in expected.items():
        rows = [row for row in primary if row["expert_confidence"] == confidence]
        result = compute_spot_check_agreement(rows)
        assert (
            result["n"],
            result["raw_safety"],
            result["node_derived_safety"],
            result["strict_node"],
            result["adequacy_penalties"],
        ) == counts
