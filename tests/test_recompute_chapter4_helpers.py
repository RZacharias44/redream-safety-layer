"""Unit tests for the pure helpers added to recompute_chapter4.py.

No API access needed; synthetic rows only.
"""

from experiments.evaluation_statistics.recompute_chapter4 import (
    agreement_distribution,
    confusion_families,
    directional_confusions,
    leading_confusion,
    missed_unsafe_overlap,
    node_flow,
    verdict_partition,
    multi_run_safety_stability,
    parse_segments,
    rerun_divergence,
    segment_identity_on_flips,
    tex_pct_two,
    vote_count_stats,
)


def row(user_input, gt_safe, pred_safe, pred_node, agreement="", segments="", gt_node=""):
    return {
        "user_input": user_input,
        "gt_safe": gt_safe,
        "pred_safe": pred_safe,
        "pred_node": pred_node,
        "agreement_ratio": agreement,
        "segments": segments,
        "gt_node": gt_node,
    }


class TestRerunDivergence:
    def test_identical_runs_have_zero_flips(self):
        rows = [row("a", "True", "True", "BEHAVIORAL_MASTERY"),
                row("b", "False", "False", "AVOIDANCE")]
        result = rerun_divergence(rows, [dict(r) for r in rows])
        assert result == {"n_matched": 2, "safety_verdict_flips": 0, "node_flips": 0}

    def test_counts_safety_and_node_flips_independently(self):
        base = [row("a", "True", "True", "BEHAVIORAL_MASTERY"),
                row("b", "False", "False", "AVOIDANCE")]
        rerun = [row("a", "True", "False", "AVOIDANCE"),   # safety + node flip
                 row("b", "False", "False", "SUPPRESSION")]  # node flip only
        result = rerun_divergence(base, rerun)
        assert result["safety_verdict_flips"] == 1
        assert result["node_flips"] == 2

    def test_unscored_rows_are_excluded(self):
        # scored_rows drops rows with blank gt_safe (boundary cases).
        base = [row("a", "", "True", "X")]
        rerun = [row("a", "", "False", "Y")]
        result = rerun_divergence(base, rerun)
        assert result["n_matched"] == 0

    def test_unmatched_rerun_rows_are_skipped(self):
        base = [row("a", "True", "True", "X")]
        rerun = [row("other", "True", "False", "Y")]
        assert rerun_divergence(base, rerun)["n_matched"] == 0


class TestAgreementDistribution:
    def test_counts_unanimous_mean_and_min(self):
        rows = [row("a", "True", "True", "X", "1.0"),
                row("b", "False", "False", "Y", "0.6"),
                row("c", "True", "True", "Z", "0.8")]
        result = agreement_distribution(rows)
        assert result["n"] == 3
        assert result["unanimous"] == 1
        assert result["min"] == 0.6
        assert abs(result["mean"] - 0.8) < 1e-9

    def test_blank_agreement_rows_are_skipped(self):
        rows = [row("a", "True", "True", "X", "1.0"),
                row("b", "False", "False", "Y", "")]  # META-only case, no votes
        assert agreement_distribution(rows)["n"] == 1

    def test_empty_input(self):
        assert agreement_distribution([]) == {"n": 0}


def test_tex_pct_two_formats_two_decimals():
    assert tex_pct_two(0.978571) == "97.86\\%"
    assert tex_pct_two(None) == "n/a"


class TestVoteCountStats:
    def test_counts_segments_and_below_nominal(self):
        rows = [
            row("a", "True", "True", "X") | {"vote_distribution": '{"seg0": {"BM": 5}, "seg1": {"AVO": 2, "BM": 1}}'},
            row("b", "False", "False", "Y") | {"vote_distribution": '{"seg0": {"AVO": 5}}'},
        ]
        result = vote_count_stats(rows)
        assert result["n_segments"] == 3
        assert result["segments_with_five_votes"] == 2
        assert result["segments_below_nominal"] == 1
        assert result["min_votes"] == 3
        assert result["vote_count_distribution"] == {3: 1, 5: 2}

    def test_rows_without_votes_counted_separately(self):
        rows = [
            row("a", "True", "True", "META_ONLY") | {"vote_distribution": "{}"},
            row("b", "False", "False", "Y") | {"vote_distribution": '{"seg0": {"AVO": 5}}'},
        ]
        result = vote_count_stats(rows)
        assert result["rows_without_votes"] == 1
        assert result["n_segments"] == 1

    def test_unscored_rows_are_included(self):
        # Full-run scope: boundary rows (blank gt_safe) count too.
        rows = [row("a", "", "True", "X") | {"vote_distribution": '{"seg0": {"BM": 5}}'}]
        assert vote_count_stats(rows)["n_segments"] == 1


SEG_ENV = '[THOUGHT] ctx:being cornered by the int... "I notice the window over the sink" → ENVIRONMENTAL_MASTERY (82%)'
SEG_INT = '[ACTION] ctx:escaping the intruder "I climb out into the morning light" → INTERRUPTION (73%)'
SEG_META = '[META] "It plays out exactly the way it always d..." → ? (?)'


class TestParseSegments:
    def test_type_and_text_extracted(self):
        assert parse_segments(SEG_ENV) == [
            ("THOUGHT", "I notice the window over the sink")
        ]

    def test_meta_segment_without_context_or_label(self):
        # META segments carry no ctx: prefix and an unclassified "? (?)" tail.
        assert parse_segments(SEG_META) == [
            ("META", "It plays out exactly the way it always d...")
        ]

    def test_multiple_segments_split_on_pipe(self):
        parsed = parse_segments(f"{SEG_ENV} | {SEG_INT}")
        assert [seg_type for seg_type, _ in parsed] == ["THOUGHT", "ACTION"]
        assert parsed[1][1] == "I climb out into the morning light"

    def test_inner_quotes_do_not_break_text_capture(self):
        seg = '[ACTION] ctx:facing them "I say "no" and stand still" → BEHAVIORAL_MASTERY (90%)'
        assert parse_segments(seg) == [("ACTION", 'I say "no" and stand still')]


class TestSegmentIdentityOnFlips:
    def test_only_safety_flips_are_inspected(self):
        base = [row("a", "True", "True", "X", segments=SEG_ENV),
                row("b", "False", "False", "Y", segments=SEG_ENV)]
        rerun = [row("a", "True", "True", "X", segments=SEG_INT),  # no flip
                 row("b", "False", "True", "Y", segments=SEG_ENV)]  # flip
        result = segment_identity_on_flips(base, rerun)
        assert result["safety_flips"] == 1
        assert result["same_segment_boundaries"] == 1
        assert result["same_boundaries_and_types"] == 1

    def test_type_change_keeps_boundary_identity_only(self):
        # THOUGHT -> META retyping on identical text (the §4.4.5 replay case).
        base = [row("a", "False", "False", "X",
                    segments='[THOUGHT] ctx:repeat... "It plays out" → TRAUMA_REPLAY (65%)')]
        rerun = [row("a", "False", "True", "X",
                     segments='[META] "It plays out" → ? (?)')]
        result = segment_identity_on_flips(base, rerun)
        assert result["safety_flips"] == 1
        assert result["same_segment_boundaries"] == 1
        assert result["same_boundaries_and_types"] == 0

    def test_different_boundaries_counted_as_neither(self):
        base = [row("a", "True", "False", "X", segments=f"{SEG_ENV} | {SEG_INT}")]
        rerun = [row("a", "True", "True", "X", segments=SEG_ENV)]
        result = segment_identity_on_flips(base, rerun)
        assert result["safety_flips"] == 1
        assert result["same_segment_boundaries"] == 0
        assert result["same_boundaries_and_types"] == 0


class TestMultiRunSafetyStability:
    def test_identical_runs_are_fully_stable(self):
        runs = [[row("a", "True", "True", "X"), row("b", "False", "False", "Y")]] * 3
        result = multi_run_safety_stability(runs)
        assert result == {"n_matched": 2, "cases_with_safety_flip": 0, "cases_stable": 2}

    def test_flip_in_one_run_counts_once(self):
        base = [row("a", "True", "True", "X"), row("b", "False", "False", "Y")]
        third = [row("a", "True", "False", "X"), row("b", "False", "False", "Y")]
        result = multi_run_safety_stability([base, [dict(r) for r in base], third])
        assert result["cases_with_safety_flip"] == 1
        assert result["cases_stable"] == 1

    def test_only_cases_scored_in_every_run_are_matched(self):
        result = multi_run_safety_stability([
            [row("a", "True", "True", "X"), row("b", "True", "True", "X")],
            [row("a", "True", "True", "X")],
        ])
        assert result["n_matched"] == 1


class TestConfusionFamilies:
    def test_family_split_and_maladaptive_pairs(self):
        rows = [
            # correct row: not an error
            row("a", "True", "True", "BEHAVIORAL_MASTERY", gt_node="BEHAVIORAL_MASTERY"),
            # adaptive -> adaptive
            row("b", "True", "True", "ENVIRONMENTAL_MASTERY", gt_node="BEHAVIORAL_MASTERY"),
            # neutral -> meta-only (upstream filtering)
            row("c", "True", "True", "META_ONLY", gt_node="NARRATIVE_SETTING"),
            # maladaptive -> maladaptive (safety verdict unchanged)
            row("d", "False", "False", "TRAUMA_REPLAY", gt_node="SUPPRESSION"),
            # maladaptive -> adaptive (the unsafe-miss direction)
            row("e", "False", "True", "BEHAVIORAL_MASTERY", gt_node="VIOLENT_REVENGE"),
            # neutral -> maladaptive (cross-family false alarm)
            row("f", "True", "False", "TRAUMA_REPLAY", gt_node="AFFECT_EXPRESSION"),
        ]
        result = confusion_families(rows)
        assert result["node_errors"] == 5
        assert result["non_maladaptive"] == 2
        assert result["within_maladaptive"] == 1
        assert result["cross_family"] == 2
        assert result["pairs_involving_maladaptive"] == {
            "SUPPRESSION -> TRAUMA_REPLAY": 1,
            "VIOLENT_REVENGE -> BEHAVIORAL_MASTERY": 1,
            "AFFECT_EXPRESSION -> TRAUMA_REPLAY": 1,
        }

    def test_unscored_rows_are_excluded(self):
        rows = [row("a", "", "True", "AVOIDANCE", gt_node="BEHAVIORAL_MASTERY")]
        assert confusion_families(rows)["node_errors"] == 0


class TestMissedUnsafeOverlap:
    def test_recovered_shared_and_introduced(self):
        # A misses unsafe cases a1, a2, a3; B misses a3 (shared) and b1 (new).
        def run(missed_inputs):
            rows = []
            for ui in ("a1", "a2", "a3", "b1", "safe1"):
                gt_safe = "True" if ui == "safe1" else "False"
                pred_safe = "True" if ui in missed_inputs else gt_safe
                rows.append(row(ui, gt_safe, pred_safe, "X"))
            return rows

        result = missed_unsafe_overlap(run({"a1", "a2", "a3"}), run({"a3", "b1"}))
        assert result == {
            "missed_a": 3,
            "missed_b": 2,
            "shared": 1,
            "recovered_by_b": 2,
            "introduced_by_b": 1,
        }

    def test_false_alarms_do_not_count_as_misses(self):
        # A safe case flagged unsafe is a false alarm, not a missed unsafe.
        base = [row("a", "True", "False", "X")]
        other = [row("a", "True", "True", "X")]
        result = missed_unsafe_overlap(base, other)
        assert result["missed_a"] == 0
        assert result["missed_b"] == 0

    def test_unscored_rows_are_excluded(self):
        base = [row("a", "", "True", "X")]
        assert missed_unsafe_overlap(base, base)["missed_a"] == 0


def node_row(gt_node, pred_node, gt_safe="True", pred_safe="True", user_input=None):
    return row(
        user_input or f"{gt_node}->{pred_node}",
        gt_safe,
        pred_safe,
        pred_node,
        gt_node=gt_node,
    )


def test_directional_confusions_separates_one_way_from_reciprocal_pairs():
    rows = (
        [node_row("A", "B", user_input=f"ab{i}") for i in range(3)]
        + [node_row("C", "D", user_input=f"cd{i}") for i in range(2)]
        + [node_row("D", "C"), node_row("E", "F"), node_row("D", "E"), node_row("E", "D")]
    )
    result = directional_confusions(rows, min_count=2)

    # Only pairs at or above the threshold are reported.
    assert set(result["pairs"]) == {"A -> B", "C -> D"}
    assert result["n_large"] == 2
    # A -> B never occurs reversed; C -> D does, so it is not one-directional.
    assert set(result["one_directional"]) == {"A -> B"}
    assert result["n_one_directional"] == 1
    assert result["max_reverse"] == 1
    # D sits on both sides of every reciprocal pair; C and E do not.
    assert result["shared_by_all_bidirectional_pairs"] == ["D"]


def test_directional_confusions_reports_no_reciprocal_nodes_when_none_exist():
    rows = [node_row("A", "B", user_input=f"ab{i}") for i in range(2)]
    result = directional_confusions(rows, min_count=2)
    assert result["max_reverse"] == 0
    assert result["shared_by_all_bidirectional_pairs"] == []


def test_node_flow_separates_absorbing_from_leaking_categories():
    rows = [
        node_row("A", "A", user_input="a-ok"),
        node_row("A", "B", user_input="a-to-b"),
        node_row("B", "B", user_input="b-ok"),
        node_row("C", "B", user_input="c-to-b"),
    ]
    flow = node_flow(rows)

    # B absorbs two cases and loses none.
    assert flow["B"]["gained"] == 2
    assert flow["B"]["lost"] == 0
    assert flow["B"]["net"] == 2
    assert flow["B"]["predicted"] == 3
    assert flow["B"]["recall"] == 1.0
    assert flow["B"]["precision"] == round(1 / 3, 6)

    # A leaks one case; it is never wrongly assigned, so precision stays 1.0.
    assert flow["A"]["net"] == -1
    assert flow["A"]["recall"] == 0.5
    assert flow["A"]["precision"] == 1.0

    # C is never predicted at all, so precision is undefined rather than zero.
    assert flow["C"]["predicted"] == 0
    assert flow["C"]["precision"] is None
    assert flow["C"]["recall"] == 0.0


def test_verdict_partition_treats_unclassified_as_the_unsafe_side():
    rows = [
        # Stays on the safe side: adaptive -> neutral. Cannot move the verdict.
        node_row("SOCIAL_MASTERY", "EMOTIONAL_MASTERY", gt_safe="True", pred_safe="True"),
        # Stays on the safe side: META_ONLY is a filtering outcome, still safe.
        node_row("NARRATIVE_SETTING", "META_ONLY", gt_safe="True", pred_safe="True"),
        # Stays on the unsafe side: one maladaptive category for another.
        node_row("SUPPRESSION", "TRAUMA_REPLAY", gt_safe="False", pred_safe="False"),
        # Crosses: UNCLASSIFIED is returned unsafe, so this is a false alarm.
        node_row("AFFECT_EXPRESSION", "UNCLASSIFIED", gt_safe="True", pred_safe="False"),
        # Crosses the other way: a missed unsafe case.
        node_row("VIOLENT_REVENGE", "BEHAVIORAL_MASTERY", gt_safe="False", pred_safe="True"),
    ]
    part = verdict_partition(rows)

    assert part["errors_within_safe_side"] == 2
    assert part["errors_within_unsafe_side"] == 1
    assert part["errors_crossing"] == 2
    assert part["n_safety_errors"] == 2
    assert part["matches_safety_errors"] is True


def test_verdict_partition_flags_a_safety_error_without_a_crossing_node_error():
    rows = [
        node_row("VIOLENT_REVENGE", "BEHAVIORAL_MASTERY", gt_safe="False", pred_safe="True"),
        # Correct node on the safe side but an unsafe verdict: the triage rule
        # this identity depends on would no longer hold.
        node_row("BEHAVIORAL_MASTERY", "BEHAVIORAL_MASTERY", gt_safe="True", pred_safe="False"),
    ]
    assert verdict_partition(rows)["matches_safety_errors"] is False


def test_node_flow_breaks_gains_and_losses_down_by_partner_category():
    rows = [
        node_row("A", "B", user_input="ab0"),
        node_row("A", "B", user_input="ab1"),
        node_row("C", "B", user_input="cb0"),
        node_row("B", "C", user_input="bc0"),
        node_row("B", "B", user_input="ok"),
    ]
    flow = node_flow(rows)

    # Partner breakdowns are sorted by count and sum to gained/lost.
    assert flow["B"]["gained_from"] == {"A": 2, "C": 1}
    assert flow["B"]["lost_to"] == {"C": 1}
    assert flow["A"]["lost_to"] == {"B": 2}
    assert flow["A"]["gained_from"] == {}
    assert sum(flow["B"]["gained_from"].values()) == flow["B"]["gained"]


def test_leading_confusion_reports_the_top_pair_with_its_margin():
    rows = (
        [node_row("A", "B", user_input=f"ab{i}") for i in range(3)]
        + [node_row("C", "D", user_input=f"cd{i}") for i in range(2)]
        + [node_row("E", "E", user_input="ok")]
    )
    lead = leading_confusion(rows)

    assert lead["pair"] == "A -> B"
    assert lead["count"] == 3
    assert lead["runner_up"] == "C -> D"
    assert lead["runner_up_count"] == 2
    assert lead["margin"] == 1


def test_leading_confusion_scores_the_vote_columns_when_asked():
    base = node_row("A", "B")
    base["vote_node"] = "A"  # the vote recovers the reference category
    other = node_row("C", "D")
    other["vote_node"] = "D"
    lead = leading_confusion([base, other], "vote_node")

    assert lead["pair"] == "C -> D"
    assert lead["count"] == 1
    assert lead["runner_up"] is None
    assert lead["margin"] == 1


def test_verdict_partition_keys_the_identity_on_rows_not_user_inputs():
    # Two rows share a user_input. Row one has a crossing node error but a
    # matching verdict; row two has a safety error with a correct node. A
    # user_input-keyed comparison would collapse both onto the same key and
    # report the identity as holding; positionally the sets differ.
    rows = [
        node_row("AFFECT_EXPRESSION", "AVOIDANCE", gt_safe="True", pred_safe="True",
                 user_input="same turn"),
        node_row("BEHAVIORAL_MASTERY", "BEHAVIORAL_MASTERY", gt_safe="True",
                 pred_safe="False", user_input="same turn"),
    ]
    part = verdict_partition(rows)

    assert part["errors_crossing"] == 1
    assert part["n_safety_errors"] == 1
    assert part["matches_safety_errors"] is False


def test_verdict_partition_scores_the_vote_columns_when_asked():
    base = node_row("SUPPRESSION", "TRAUMA_REPLAY", gt_safe="False", pred_safe="False")
    # The pipeline columns stay on the unsafe side; the vote verdict crosses.
    base["vote_safe"] = "True"
    base["vote_node"] = "BEHAVIORAL_MASTERY"
    part = verdict_partition([base], "vote_safe", "vote_node")

    assert part["errors_within_unsafe_side"] == 0
    assert part["errors_crossing"] == 1
    assert part["n_safety_errors"] == 1
    assert part["matches_safety_errors"] is True
