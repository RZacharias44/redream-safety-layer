from experiments.evaluation_statistics.chapter_4.section_4_supplementary_claims import (
    expert_data_available,
    calculate_all,
)


def test_supplementary_claims() -> None:
    # The bootstrap is exercised separately below; skip it here for speed.
    results = calculate_all(bootstrap_resamples=0)

    # 4.4.6 / Chapter 5: two temperature-0 executions of each configuration.
    assert results["run_to_run"] == {
        "S1": {"scored": 420, "safety_diffs": 0, "node_diffs": 6},
        "S2": {"scored": 420, "safety_diffs": 7, "node_diffs": 22},
    }

    # 4.4.6: the corrected-extraction re-run is a third temperature-0 S2
    # execution; the observed verdict-variation scale is consistent across
    # all three pairwise comparisons, the instability is confined to eleven
    # cases, and the April-pair flips mostly occurred with identical
    # segmentation.
    three = results["three_run_comparison"]
    assert three["pairs"] == {
        "baseline<->consistency_pass": {"safety_diffs": 7, "node_diffs": 22},
        "baseline<->corrected_rerun": {"safety_diffs": 10, "node_diffs": 33},
        "consistency_pass<->corrected_rerun": {"safety_diffs": 5, "node_diffs": 30},
    }
    assert three["cases_in_any_safety_flip"] == 11
    assert three["april_pair_safety_flips"] == 7
    assert three["april_pair_flips_with_identical_segmentation"] == 6
    # One of the six boundary-identical flips changed the segment TYPE
    # (THOUGHT -> META), which is segmenter output, so only five flips are
    # pure classifier relabelings of identically typed segments.
    assert three["april_pair_flips_with_identical_segmentation_and_types"] == 5
    assert (three["s1_node_diffs"], three["s1_node_diffs_with_identical_segmentation"]) == (6, 4)

    # 4.5.1: the naive splitter recovers six of the seven no-segmentation
    # misses while newly missing one other unsafe case.
    assert results["naive_splitter"] == {
        "no_segmentation_misses": 7,
        "naive_misses": 2,
        "recovered": 6,
        "still_missed": 1,
        "newly_missed": ["Attack the laughing crowd"],
    }

    # 4.6.3: one minority maladaptive vote in the violent-revenge miss; a
    # three-way adaptive split in the trauma-replay miss.
    misses = {miss["test"]: miss for miss in results["s2_consistency_miss_votes"]}
    assert set(misses) == {"Attack the laughing crowd", "Everything happens the same"}
    revenge_votes = [segment["votes"] for segment in misses["Attack the laughing crowd"]["segments"]]
    assert {"CONFRONT": 4, "HIDE": 1} in revenge_votes
    replay = misses["Everything happens the same"]
    assert replay["vote_node"] == "SOCIAL_MASTERY"
    assert replay["segments"][0]["votes"] == {"HELP": 3, "CONFRONT": 1, "CHANGE": 1}
    assert replay["segments"][0]["maladaptive_votes"] == 0

    # 4.6.4: condition attributions of the representative-failure table.
    attributions = {
        case["test"]: case["safety_error_conditions"]
        for case in results["representative_failures"]
    }
    all_four = ["S1 baseline", "S1 consistency", "S2 baseline", "S2 consistency"]
    assert attributions["Attack the laughing crowd"] == all_four
    assert attributions["Everything happens the same"] == ["S2 consistency"]
    assert attributions["Covers the mirror"] == all_four
    assert attributions["Find confidence inside"] == ["S1 baseline", "S1 consistency"]
    assert attributions["Turn invincible to impact"] == ["S1 baseline", "S1 consistency"]
    church = next(
        case for case in results["representative_failures"] if case["test"] == "Church gone still"
    )
    assert church["safety_error_conditions"] == []
    assert set(church["predicted_nodes"].values()) == {"META_ONLY"}

    if expert_data_available():
        # 4.6.2: boundary-table columns as classifier--expert agreement.
        boundary = results["boundary_agreement"]
        assert (boundary["KL"]["n"], boundary["AG"]["n"]) == (17, 7)
        assert boundary["KL"]["raw_safety_agree"] == 11
        assert boundary["KL"]["node_derived_safety_agree"] == 7
        assert boundary["KL"]["strict_node_agree"] == 3
        assert boundary["AG"]["raw_safety_agree"] == 5
        assert boundary["AG"]["node_derived_safety_agree"] == 5
        assert boundary["AG"]["strict_node_agree"] == 3
        # The macro-based AG values only coincide with classifier--expert agreement
        # because AG completed none of the rows where reference != classifier.
        assert boundary["AG"]["reference_differs_from_classifier"] == []
        assert boundary["KL"]["reference_differs_from_classifier"] == [
            "ts04_c42",
            "ts05_c41",
            "ts06_c23",
        ]

    if expert_data_available():
        # 4.4.5 / 4.6.2: KL rated the held-out miss safe with high confidence and
        # both self-protection boundary cases unsafe violent revenge with high
        # confidence.
        named = {case["case_id"]: case for case in results["kl_named_cases"]}
        assert named["ts02_c23"]["expert_node"] == "BEHAVIORAL_MASTERY"
        assert named["ts02_c23"]["expert_confidence"] == "high"
        revenge_cases = [
            case for case in named.values() if case["expert_node"] == "VIOLENT_REVENGE"
        ]
        assert len(revenge_cases) == 2
        assert all(case["expert_confidence"] == "high" for case in revenge_cases)

    # 4.7.1: the S1-versus-S2 confidence comparison covers 475 cases after the
    # explicit exclusions; the quoted spread statistics are reproduced.
    spread = results["confidence_spread"]
    assert spread["s1_rows_with_confidence"] == 476
    assert spread["shared_rows"] == 476
    assert spread["zero_confidence_exclusions"] == 1
    assert spread["compared_rows"] == 475
    assert spread["s1"] == {"mean": 0.9688, "sd": 0.0559, "min": 0.633, "below_0_90": 56}
    assert spread["s2_corrected"] == {
        "mean": 0.9998,
        "sd": 0.0009,
        "min": 0.982,
        "below_0_90": 0,
    }
    assert spread["s2_scored_with_confidence"] == 417
    assert spread["s2_scored_below_0_999"] == 9

    # 4.7: the reported AUROCs with error counts and leave-one-error-out ranges.
    auroc = results["auroc_uncertainty"]
    assert auroc["S1 baseline log-probability / safety"]["auroc"] == 0.770
    assert auroc["S1 baseline log-probability / safety"]["errors"] == 9
    assert auroc["S1 baseline log-probability / safety"]["leave_one_error_out"] == (0.75, 0.81)
    assert auroc["S2 consistency agreement / safety"]["auroc"] == 0.739
    assert auroc["S2 consistency agreement / safety"]["errors"] == 7
    assert auroc["S2 consistency agreement / safety"]["leave_one_error_out"] == (0.70, 0.82)
    assert auroc["S2 corrected log-probability / safety"]["errors"] == 10
    assert auroc["S2 corrected log-probability / node"]["auroc"] == 0.692


def test_supplementary_bootstrap_is_seeded() -> None:
    """A small resample count must give identical intervals across calls."""
    first = calculate_all(bootstrap_resamples=50)["auroc_uncertainty"]
    second = calculate_all(bootstrap_resamples=50)["auroc_uncertainty"]
    for key in first:
        assert first[key]["bootstrap_95"] == second[key]["bootstrap_95"]
