from collections import Counter

import pytest

from experiments.evaluation_statistics.chapter_4.section_4_8_system_expert_evaluation import (
    FILES,
    calculate_all,
)

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in FILES.values()), reason="row-level expert annotation tables not present"
)


def test_section_4_8_completion_and_condition_tables() -> None:
    results = calculate_all()
    assert results["completion"] == {
        "KL": {"queue": 86, "complete": 85, "pairs": 42},
        "AG": {"queue": 86, "complete": 41, "pairs": 19},
    }

    expected = {
        ("KL", "control"): (43, Counter(yes=29, uncertain=9, no=5), 1.56),
        ("KL", "treatment"): (42, Counter(yes=21, uncertain=12, no=9), 1.29),
        ("AG", "control"): (21, Counter(yes=19, uncertain=1, no=1), 1.86),
        ("AG", "treatment"): (20, Counter(yes=15, uncertain=2, no=3), 1.60),
    }
    for (rater, condition), (n, counts, mean) in expected.items():
        values = results["conditions"][rater][condition]
        assert values["n"] == n
        assert values["clinical_acceptability"] == counts
        assert round(values["mean_clinical_acceptability"], 2) == mean


def test_section_4_8_paired_table_and_shared_agreement() -> None:
    results = calculate_all()
    expected = {
        "KL": {
            "Clinical acceptability": (42, 4, 27, 11, -0.26),
            "Redirect quality": (41, 8, 24, 9, -0.02),
            "Distress validation": (31, 13, 10, 8, 0.19),
            "Maladaptive endorsement": (34, 0, 29, 5, -0.24),
        },
        "AG": {
            "Clinical acceptability": (19, 0, 15, 4, -0.37),
            "Redirect quality": (19, 2, 14, 3, -0.16),
            "Distress validation": (19, 4, 8, 7, -0.16),
            "Maladaptive endorsement": (17, 5, 10, 2, 0.12),
        },
    }
    for rater, dimensions in expected.items():
        for dimension, target in dimensions.items():
            values = results["paired"][rater][dimension]
            actual = (
                values["n"],
                values["better"],
                values["same"],
                values["worse"],
                round(values["mean_change"], 2),
            )
            assert actual == target

    shared = results["shared"]
    assert shared == {
        "n": 41,
        "Clinical acceptability": 33,
        "Redirect quality": 19,
        "Distress validation": 12,
        "Maladaptive endorsement": 28,
        "AG strong redirect": 34,
        "KL strong redirect": 20,
        "AG appropriate validation": 27,
        "KL appropriate validation": 8,
    }
    assert results["kl_notes"] == {
        "adaptive_or_partly_adaptive": 18,
        "false_positive_redirect": 4,
    }
