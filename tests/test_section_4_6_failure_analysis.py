from collections import Counter

import pytest

from experiments.evaluation_statistics.chapter_4.section_4_6_failure_analysis import (
    AG_FILE,
    KL_FILE,
    NO_SEGMENTATION_FILE,
    S1_FILE,
    S2_CONSISTENCY_FILE,
    boundary_classifier_expert_agreement,
    boundary_theme_summary,
    boundary_type_by_input,
    false_alarm_mechanisms,
    no_segmentation_misses,
    node_error_patterns,
    read_consistency_rows,
    read_rows,
    safety_errors,
    unchanged_replay_disagreements,
)


def test_section_4_6_safety_error_counts() -> None:
    assert safety_errors(read_rows(S1_FILE)) == {
        "n": 420,
        "total": 9,
        "false_alarms": 8,
        "unsafe_misses": 1,
    }
    # Scored on the majority vote; the temperature-0 prediction stored in the
    # same rows gives a different, unreported error set (2 false alarms, 3 misses).
    assert safety_errors(read_consistency_rows(S2_CONSISTENCY_FILE)) == {
        "n": 420,
        "total": 7,
        "false_alarms": 5,
        "unsafe_misses": 2,
    }


def test_section_4_6_node_error_patterns() -> None:
    result = node_error_patterns(read_rows(S1_FILE))
    assert result["errors"] == 92
    assert result["patterns"] == 30
    assert result["top_three"] == [
        (("AFFECT_EXPRESSION", "EMOTIONAL_MASTERY"), 17),
        (("MYTHICAL_MASTERY", "ENVIRONMENTAL_MASTERY"), 9),
        (("SOCIAL_MASTERY", "EMOTIONAL_MASTERY"), 7),
    ]
    assert result["top_three_count"] == 33
    assert round(result["top_three_percent"], 1) == 35.9
    assert result["remaining_patterns"] == 27


def test_section_4_6_false_alarm_grouping() -> None:
    result = false_alarm_mechanisms(read_rows(S1_FILE))
    assert result == {
        "total": 8,
        "segment_level": 4,
        "category_boundary": 4,
        "category_boundary_pairs": Counter(
            {
                ("ENVIRONMENTAL_MASTERY", "AVOIDANCE"): 2,
                ("EMOTIONAL_MASTERY", "SUPPRESSION"): 1,
                ("EMOTIONAL_MASTERY", "AVOIDANCE"): 1,
            }
        ),
    }


def test_section_4_6_no_segmentation_misses() -> None:
    assert no_segmentation_misses(read_rows(NO_SEGMENTATION_FILE)) == {
        "total": 7,
        "expected_nodes": Counter({"VIOLENT_REVENGE": 7}),
    }


@pytest.mark.skipif(not (KL_FILE.exists() and AG_FILE.exists()), reason="row-level expert annotation tables not present")
def test_section_4_6_boundary_statistics() -> None:
    kl_rows = read_rows(KL_FILE)
    assert boundary_classifier_expert_agreement(kl_rows) == {
        "n": 17,
        "raw_safety": 11,
        "node_derived_safety": 7,
        "strict_node": 3,
    }

    unchanged = unchanged_replay_disagreements(kl_rows)
    assert [row["case_id"] for row in unchanged] == [
        "ts02_c40",
        "ts05_c40",
        "ts07_c40",
        "ts01_c40",
    ]

    type_index = boundary_type_by_input()
    kl_themes = boundary_theme_summary(kl_rows, type_index)
    ag_themes = boundary_theme_summary(read_rows(AG_FILE), type_index)
    assert {theme: values["n"] for theme, values in kl_themes.items()} == {
        "active coping or scene change versus avoidance or interruption": 6,
        "unchanged replay versus narrative description": 4,
        "freezing and passivity": 4,
        "regulation versus suppression": 1,
        "self-protection versus violent revenge": 2,
    }
    assert sum(values["unsafe"] for values in kl_themes.values()) == 17
    assert {theme: values["n"] for theme, values in ag_themes.items()} == {
        "active coping or scene change versus avoidance or interruption": 2,
        "unchanged replay versus narrative description": 3,
        "freezing and passivity": 1,
        "regulation versus suppression": 1,
    }
    assert sum(values["unsafe"] for values in ag_themes.values()) == 6
    boundary_theme_summary,
    boundary_type_by_input,
