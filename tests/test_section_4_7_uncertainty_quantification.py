import pytest

from experiments.evaluation_statistics.chapter_4.section_4_7_uncertainty_quantification import (
    KL_FILE,
    calculate_all,
)


def test_section_4_7_auroc_and_thresholds() -> None:
    results = calculate_all()
    # Log-probability confidence is reported for the baselines; agreement for the
    # consistency conditions, scored on the majority-vote verdict they report.
    expected_auc = {
        "S1 baseline log-probability": (416, 4, 0.77, 0.85),
        # Superseded: produced by the defective classification-token extraction.
        "S2 baseline log-probability": (417, 3, 0.62, 0.61),
        "S2 baseline corrected log-probability": (417, 3, 0.59, 0.69),
        "S1 consistency agreement": (417, 3, 0.68, 0.68),
        "S2 consistency agreement": (417, 3, 0.74, 0.71),
    }
    for name, (usable, missing, safety_auc, node_auc) in expected_auc.items():
        values = results["signals"][name]
        assert (values["n_usable"], values["n_missing"]) == (usable, missing)
        assert round(values["safety_auroc"], 2) == safety_auc
        assert round(values["node_auroc"], 2) == node_auc

    assert results["thresholds"] == {
        "S1 0.90": {
            "threshold": 0.90,
            "flagged": 43,
            "safety_errors": 9,
            "caught": 1,
            "correct_flagged": 42,
        },
        "S1 0.95": {
            "threshold": 0.95,
            "flagged": 71,
            "safety_errors": 9,
            "caught": 4,
            "correct_flagged": 67,
        },
        "S2 0.80": {
            "threshold": 0.80,
            "flagged": 160,
            "safety_errors": 8,
            "caught": 5,
            "correct_flagged": 155,
        },
        # After the extraction fix the signal lives in the last decimals: a System 1
        # scale cutoff flags a single case, and a usable operating point sits at
        # four nines, which is not a defensible clinical threshold.
        "S2 corrected 0.90": {
            "threshold": 0.90,
            "flagged": 1,
            "safety_errors": 10,
            "caught": 1,
            "correct_flagged": 0,
        },
        "S2 corrected 0.9999": {
            "threshold": 0.9999,
            "flagged": 150,
            "safety_errors": 10,
            "caught": 5,
            "correct_flagged": 145,
        },
    }


def test_section_4_7_s2_confidence_spread() -> None:
    """The corrected System 2 signal ranks cases but occupies almost no range."""
    results = calculate_all()

    superseded = results["s2_superseded_spread"]
    corrected = results["s2_corrected_spread"]

    # Superseded values are the fluency of a ~150-token rationale, clustered near 0.8.
    assert round(superseded["mean"], 3) == 0.803
    assert round(superseded["node_correct_mean"], 3) == 0.808
    assert round(superseded["node_incorrect_mean"], 3) == 0.783

    # Corrected values are the label tokens themselves, saturated near 1.0.
    assert round(corrected["mean"], 3) == 0.997
    assert round(corrected["node_correct_mean"], 4) == 0.9998
    assert round(corrected["node_incorrect_mean"], 3) == 0.988


@pytest.mark.skipif(not KL_FILE.exists(), reason="row-level expert annotation tables not present")
def test_section_4_7_kl_spotcheck_confidence() -> None:
    # Non-boundary rows only: boundary rows carry the author's internal-review
    # annotation as `reference_node`, not a benchmark reference label (thesis
    # section 4.6.2: boundary cases have no reference node).
    values = calculate_all()["kl_model_confidence"]
    assert values["n"] == 80
    assert values["n_boundary_excluded"] == 17
    assert round(values["classifier_wrong_mean"], 3) == 0.939
    assert round(values["classifier_correct_mean"], 3) == 0.984
    assert round(values["expert_wrong_mean"], 3) == 0.958
    assert round(values["expert_correct_mean"], 3) == 0.985
    assert values["threshold_0.90"] == {
        "flagged": 5,
        "classifier_reference_disagreements": 3,
        "expert_reference_disagreements": 3,
        "both": 3,
    }
    assert values["threshold_0.95"] == {
        "flagged": 12,
        "classifier_reference_disagreements": 5,
        "expert_reference_disagreements": 7,
        "both": 4,
    }


def test_section_4_7_heldout_confidence_by_node_correctness() -> None:
    # Section 4.7.1 held-out replication: scored held-out cases (buried-alive
    # suite and boundary cases excluded), two META_ONLY rows without a
    # confidence value dropped.
    values = calculate_all()["heldout_model_confidence"]
    assert values["n_scored"] == 231
    assert values["n_with_conf"] == 229
    assert values["node_correct_n"] == 186
    assert values["node_wrong_n"] == 43
    assert round(values["node_wrong_mean"], 3) == 0.937
    assert round(values["node_correct_mean"], 3) == 0.982
