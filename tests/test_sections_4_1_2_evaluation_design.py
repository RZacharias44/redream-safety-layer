from experiments.evaluation_statistics.chapter_4.sections_4_1_2_evaluation_design import (
    calculate_all,
)


def test_chapter_4_dataset_and_review_counts() -> None:
    results = calculate_all()

    assert (
        results["development"]["n_scenarios"],
        results["development"]["n_cases"],
        results["development"]["n_scored"],
        results["development"]["n_boundary"],
        results["development"]["n_safe"],
        results["development"]["n_unsafe"],
        len(results["development"]["nodes"]),
    ) == (10, 480, 420, 60, 240, 180, 12)

    assert (
        results["heldout_initial"]["n_scenarios"],
        results["heldout_initial"]["n_cases"],
        results["excluded_scenarios"],
        results["excluded_cases"],
    ) == (7, 315, 1, 48)

    heldout = results["heldout_retained"]
    assert (
        heldout["n_scenarios"],
        heldout["n_cases"],
        heldout["n_scored"],
        heldout["n_boundary"],
        heldout["n_clean"],
        heldout["n_multipart"],
        heldout["n_safe"],
        heldout["n_unsafe"],
        len(heldout["nodes"]),
    ) == (6, 267, 231, 36, 201, 30, 141, 90, 12)

    assert results["multi_maladaptive"]["n_cases"] == 35
    assert results["author_review"] == {
        "boundary": 36,
        "affect_expression": 18,
        "violent_revenge": 16,
        "spot_check": 60,
        "reviewed_scored": 94,
        "label_changes": 0,
    }
