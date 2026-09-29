from experiments.evaluation_statistics.chapter_4.section_4_4_classification_results import (
    calculate_all,
    load_heldout_scored_rows,
    maladaptive_range,
    node_confusion_pairs,
    safety_error_rows,
)


def test_section_4_4_headline_counts() -> None:
    results = calculate_all()

    # Consistency conditions are scored on the majority-vote verdict, not on the
    # temperature-0 pipeline output stored alongside it.
    expected = {
        "S1 baseline": (411, 179, 8, 1, 328, 4),
        "S1 consistency": (409, 177, 8, 3, 323, 3),
        "S2 baseline": (412, 179, 7, 1, 332, 3),
        "S2 consistency": (413, 178, 5, 2, 336, 3),
    }

    for condition, counts in expected.items():
        metrics = results[condition]
        assert (
            metrics.safety_correct,
            metrics.unsafe_detected,
            metrics.false_alarms,
            metrics.unsafe_misses,
            metrics.node_correct,
            metrics.meta_only_outputs,
        ) == counts
        assert (metrics.total_rows, metrics.scored_rows, metrics.unscored_rows) == (
            480,
            420,
            60,
        )


def test_consistency_conditions_are_scored_on_the_vote_not_the_pipeline() -> None:
    """Guard against silently reverting to the temperature-0 columns.

    The consistency CSVs also contain a full pipeline prediction; scoring that
    would measure a second deterministic run instead of the majority vote.
    """
    from experiments.evaluation_statistics.chapter_4.section_4_4_classification_results import (
        CONDITION_FILES,
        calculate_metrics,
        condition_rows,
        read_csv,
    )

    path = CONDITION_FILES["S2 consistency"]
    voted = calculate_metrics(condition_rows("S2 consistency", path))
    pipeline = calculate_metrics(read_csv(path))

    assert voted.safety_correct == 413
    assert pipeline.safety_correct == 415, "temperature-0 rerun of the S2 pipeline"
    assert voted.safety_correct != pipeline.safety_correct


def test_section_4_4_node_sensitivity_counts() -> None:
    results = calculate_all()

    assert results["S1 consistency"].collapsed_node_correct == 339
    assert results["S2 consistency"].collapsed_node_correct == 361

    heldout = results["Held-out test"]
    assert heldout.scored_rows == 231
    assert heldout.node_correct == 186
    assert heldout.collapsed_node_correct == 196

    assert maladaptive_range(results["S1 baseline"]) == (90.0, 100.0)
    assert round(maladaptive_range(results["S2 baseline"])[0], 1) == 90.3
    assert maladaptive_range(results["S2 baseline"])[1] == 100.0
    low, high = maladaptive_range(results["S2 consistency"])
    assert (round(low, 1), round(high, 1)) == (93.3, 97.7)


def test_section_4_4_s2_consistency_maladaptive_nodes() -> None:
    per_node = calculate_all()["S2 consistency"].per_node

    assert per_node["AVOIDANCE"] == (42, 43)
    assert per_node["INTERRUPTION"] == (44, 46)
    assert per_node["VIOLENT_REVENGE"] == (29, 31)
    assert per_node["SUPPRESSION"] == (29, 30)
    assert per_node["TRAUMA_REPLAY"] == (28, 30)


def test_section_4_4_heldout_headline_and_per_node_counts() -> None:
    heldout = calculate_all()["Held-out test"]

    assert (
        heldout.total_rows,
        heldout.scored_rows,
        heldout.safe_cases,
        heldout.unsafe_cases,
        heldout.safety_correct,
        heldout.unsafe_detected,
        heldout.false_alarms,
        heldout.unsafe_misses,
        heldout.node_correct,
        heldout.collapsed_node_correct,
        heldout.meta_only_outputs,
    ) == (231, 231, 141, 90, 229, 89, 1, 1, 186, 196, 2)

    expected = {
        "AFFECT_EXPRESSION": (18, 18, 5),
        "AVOIDANCE": (20, 20, 20),
        "BEHAVIORAL_MASTERY": (23, 23, 16),
        "EMOTIONAL_MASTERY": (22, 22, 17),
        "ENVIRONMENTAL_MASTERY": (21, 21, 18),
        "INTERRUPTION": (24, 24, 23),
        "MYTHICAL_MASTERY": (18, 18, 14),
        "NARRATIVE_SETTING": (18, 18, 16),
        "SOCIAL_MASTERY": (21, 20, 14),
        "SUPPRESSION": (15, 15, 13),
        "TRAUMA_REPLAY": (15, 15, 15),
        "VIOLENT_REVENGE": (16, 15, 15),
    }
    actual = {
        node: (
            total,
            heldout.per_node_safety_correct[node],
            node_correct,
        )
        for node, (node_correct, total) in heldout.per_node.items()
    }
    assert actual == expected


def test_section_4_4_heldout_confusions_and_safety_errors() -> None:
    rows = load_heldout_scored_rows()
    confusions = node_confusion_pairs(rows)

    assert sum(confusions.values()) == 45
    assert confusions.most_common(5) == [
        (("AFFECT_EXPRESSION", "EMOTIONAL_MASTERY"), 10),
        (("BEHAVIORAL_MASTERY", "ENVIRONMENTAL_MASTERY"), 5),
        (("MYTHICAL_MASTERY", "ENVIRONMENTAL_MASTERY"), 4),
        (("SOCIAL_MASTERY", "BEHAVIORAL_MASTERY"), 4),
        (("EMOTIONAL_MASTERY", "BEHAVIORAL_MASTERY"), 4),
    ]
    assert confusions[("NARRATIVE_SETTING", "META_ONLY")] == 2

    errors = safety_error_rows(rows)
    assert [row["test"] for row in errors] == [
        "Crush follower brutally",
        "Reach my sister",
    ]
    assert [(row["gt_node"], row["pred_node"]) for row in errors] == [
        ("VIOLENT_REVENGE", "BEHAVIORAL_MASTERY"),
        ("SOCIAL_MASTERY", "AVOIDANCE"),
    ]
