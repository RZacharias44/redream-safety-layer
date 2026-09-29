"""Supplementary Chapter 4 claims not covered by the per-section scripts.

Run from the repository root:

    uv run python experiments/evaluation_statistics/chapter_4/section_4_supplementary_claims.py

Each numbered block names the chapter passage it verifies. These claims were
identified during the August 2026 number audit as appearing in the chapter
text without a computing script; every value is derived here from the same
row-level CSV files the other section scripts read. Block 8 additionally
quantifies the sampling uncertainty of the reported AUROC values
(leave-one-error-out ranges and bootstrap confidence intervals), supporting
the small-sample caveat in Section 4.7.
"""

from __future__ import annotations

import csv
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any

from experiments.evaluation_statistics.consistency_votes import derive_vote_verdicts


ROOT = Path(__file__).resolve().parents[3]

S1_BASELINE = ROOT / "data/benchmarks/dev_s1_baseline.csv"
S1_CONSISTENCY = ROOT / "data/benchmarks/dev_s1_consistency_n5.csv"
S2_BASELINE = ROOT / "data/benchmarks/dev_s2_baseline.csv"
S2_CONSISTENCY = ROOT / "data/benchmarks/dev_s2_consistency_n5.csv"
S2_CORRECTED = ROOT / "data/benchmarks/dev_s2_baseline_rerun_corrected_logprobs.csv"
NO_SEGMENTATION = ROOT / "data/benchmarks/dev_s1_ablation_no_segmentation.csv"
NAIVE_SPLITTER = ROOT / "data/benchmarks/dev_s1_ablation_naive_splitter.csv"
AG_SPOTCHECK = ROOT / "data/expert_analysis/ag_2026_07/spotcheck_merged.csv"
KL_SPOTCHECK = ROOT / "data/expert_analysis/kl_2026_06/spotcheck_merged.csv"

EXPERT_DATA_NOTICE = (
    "The row-level expert annotation tables under data/expert_analysis/ are not "
    "present: they are personal data of the raters and are distributed only with "
    "their consent. The aggregate results are in "
    "data/evaluation_statistics/chapter4/chapter4_statistics.json ('expert_spotcheck')."
)


def expert_data_available() -> bool:
    return AG_SPOTCHECK.exists() and KL_SPOTCHECK.exists()

MALADAPTIVE_NODES = {
    "AVOIDANCE",
    "INTERRUPTION",
    "VIOLENT_REVENGE",
    "SUPPRESSION",
    "TRAUMA_REPLAY",
}
MALADAPTIVE_CODES = ("HIDE", "ESCAPE", "DESTROY", "DENY", "REPLAY")

# Test names repeat across nightmare suites, so representative cases are keyed
# by (suite_id, test); suite 7 holds the S2-consistency trauma-replay miss.
REPRESENTATIVE_CASES = (
    ("6", "Attack the laughing crowd"),
    ("7", "Everything happens the same"),
    ("8", "Covers the mirror"),
    ("2", "Find confidence inside"),
    ("3", "Turn invincible to impact"),
    ("2", "Church gone still"),
)

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260818


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def scored(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if str(row.get("gt_safe", "")).strip()]


def by_input(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["user_input"]: row for row in rows}


def percent(numerator: int, denominator: int) -> str:
    return f"{100.0 * numerator / denominator:.2f}% ({numerator}/{denominator})"


def run_to_run_variation() -> dict[str, dict[str, int]]:
    """Section 4.4.6 and Chapter 5: two temperature-0 executions compared.

    Each consistency run also wrote a full temperature-0 pipeline prediction
    to ``pred_safe``/``pred_node``, so those columns are a second independent
    execution of the corresponding baseline configuration.
    """
    pairs = (
        ("S1", S1_BASELINE, S1_CONSISTENCY),
        ("S2", S2_BASELINE, S2_CONSISTENCY),
    )
    out: dict[str, dict[str, int]] = {}
    for name, first_path, second_path in pairs:
        first = by_input(read_csv(first_path))
        second = by_input(read_csv(second_path))
        keys = [key for key, row in first.items() if str(row.get("gt_safe", "")).strip()]
        out[name] = {
            "scored": len(keys),
            "safety_diffs": sum(
                first[key]["pred_safe"] != second[key]["pred_safe"] for key in keys
            ),
            "node_diffs": sum(
                first[key]["pred_node"] != second[key]["pred_node"] for key in keys
            ),
        }
    return out


def s2_three_run_comparison() -> dict[str, Any]:
    """Section 4.4.6: three temperature-0 System 2 executions compared pairwise.

    The corrected-extraction re-run (Section 4.7) is a third deterministic
    execution of the System 2 configuration, about three months after the
    April baseline and consistency runs; only its confidence logging differs.
    Segment boundaries are compared on the logged (truncated) segment spans,
    ignoring labels: segments partition the turn in order, so equal counts
    with identical span starts pin the same boundaries. Because the segment
    type (for example THOUGHT versus META) is also segmenter output and META
    segments are filtered before classification, flips are additionally
    decomposed by whether the types matched. The S1 pair is decomposed the
    same way for its six node differences.
    """

    def span_texts(row: dict[str, str]) -> tuple[str, ...]:
        return tuple(re.findall(r'"([^"]*)"', row.get("segments", "") or ""))

    def span_types(row: dict[str, str]) -> tuple[str, ...]:
        return tuple(re.findall(r"\[([A-Z]+)\]", row.get("segments", "") or ""))

    runs = {
        "baseline": by_input(read_csv(S2_BASELINE)),
        "consistency_pass": by_input(read_csv(S2_CONSISTENCY)),
        "corrected_rerun": by_input(read_csv(S2_CORRECTED)),
    }
    keys = [
        key for key, row in runs["baseline"].items() if str(row.get("gt_safe", "")).strip()
    ]
    pair_names = (
        ("baseline", "consistency_pass"),
        ("baseline", "corrected_rerun"),
        ("consistency_pass", "corrected_rerun"),
    )
    pairs: dict[str, dict[str, int]] = {}
    flipped_cases: set[str] = set()
    for left, right in pair_names:
        safety = {key for key in keys if runs[left][key]["pred_safe"] != runs[right][key]["pred_safe"]}
        node = {key for key in keys if runs[left][key]["pred_node"] != runs[right][key]["pred_node"]}
        flipped_cases |= safety
        pairs[f"{left}<->{right}"] = {"safety_diffs": len(safety), "node_diffs": len(node)}

    april_flips = [
        key for key in keys
        if runs["baseline"][key]["pred_safe"] != runs["consistency_pass"][key]["pred_safe"]
    ]
    april_same_segments = sum(
        span_texts(runs["baseline"][key]) == span_texts(runs["consistency_pass"][key])
        for key in april_flips
    )
    april_same_segments_and_types = sum(
        span_texts(runs["baseline"][key]) == span_texts(runs["consistency_pass"][key])
        and span_types(runs["baseline"][key]) == span_types(runs["consistency_pass"][key])
        for key in april_flips
    )

    s1_base = by_input(read_csv(S1_BASELINE))
    s1_pass = by_input(read_csv(S1_CONSISTENCY))
    s1_node_diffs = [
        key for key in keys if s1_base[key]["pred_node"] != s1_pass[key]["pred_node"]
    ]
    s1_same_segments = sum(
        span_texts(s1_base[key]) == span_texts(s1_pass[key]) for key in s1_node_diffs
    )

    return {
        "scored": len(keys),
        "pairs": pairs,
        "cases_in_any_safety_flip": len(flipped_cases),
        "flipped_case_tests": sorted(runs["baseline"][key]["test"] for key in flipped_cases),
        "april_pair_safety_flips": len(april_flips),
        "april_pair_flips_with_identical_segmentation": april_same_segments,
        "april_pair_flips_with_identical_segmentation_and_types": april_same_segments_and_types,
        "s1_node_diffs": len(s1_node_diffs),
        "s1_node_diffs_with_identical_segmentation": s1_same_segments,
    }


def naive_splitter_recovery() -> dict[str, Any]:
    """Section 4.5.1: what the naive splitter changes about no-segmentation misses."""

    def misses(path: Path) -> set[str]:
        return {
            row["user_input"]
            for row in scored(read_csv(path))
            if str(row["gt_safe"]).strip() == "False" and str(row["pred_safe"]).strip() == "True"
        }

    no_seg = misses(NO_SEGMENTATION)
    naive = misses(NAIVE_SPLITTER)
    tests = {row["user_input"]: row["test"] for row in read_csv(NO_SEGMENTATION)}
    return {
        "no_segmentation_misses": len(no_seg),
        "naive_misses": len(naive),
        "recovered": len(no_seg - naive),
        "still_missed": len(no_seg & naive),
        "newly_missed": sorted(tests[user_input] for user_input in naive - no_seg),
    }


def s2_consistency_miss_votes() -> list[dict[str, Any]]:
    """Section 4.6.3: vote composition behind the two S2-consistency unsafe misses."""
    annotated, _ = derive_vote_verdicts(read_csv(S2_CONSISTENCY))
    out = []
    for row in annotated:
        if str(row.get("gt_safe", "")).strip() != "False" or row["vote_safe"] != "True":
            continue
        segments = []
        for segment_key, votes in json.loads(row["vote_distribution"]).items():
            segments.append({
                "segment": segment_key,
                "votes": dict(votes),
                "maladaptive_votes": sum(
                    count for code, count in votes.items() if code in MALADAPTIVE_CODES
                ),
                "total_votes": sum(votes.values()),
            })
        out.append({
            "test": row["test"],
            "expected_node": row["gt_node"],
            "vote_node": row["vote_node"],
            "segments": segments,
        })
    return out


def representative_failure_conditions() -> list[dict[str, Any]]:
    """Section 4.6.4, representative-failure table: per-condition attribution."""
    baseline_s1 = read_csv(S1_BASELINE)
    baseline_s2 = by_input(read_csv(S2_BASELINE))
    consistency_s1, _ = derive_vote_verdicts(read_csv(S1_CONSISTENCY))
    consistency_s2, _ = derive_vote_verdicts(read_csv(S2_CONSISTENCY))
    vote_s1 = by_input(consistency_s1)
    vote_s2 = by_input(consistency_s2)

    out = []
    for suite_id, test_name in REPRESENTATIVE_CASES:
        row = next(
            row
            for row in baseline_s1
            if row["test"] == test_name and row["suite_id"] == suite_id
        )
        user_input = row["user_input"]
        conditions = {
            "S1 baseline": (row["pred_safe"], row["pred_node"]),
            "S1 consistency": (vote_s1[user_input]["vote_safe"], vote_s1[user_input]["vote_node"]),
            "S2 baseline": (
                baseline_s2[user_input]["pred_safe"],
                baseline_s2[user_input]["pred_node"],
            ),
            "S2 consistency": (vote_s2[user_input]["vote_safe"], vote_s2[user_input]["vote_node"]),
        }
        out.append({
            "test": test_name,
            "expected_node": row["gt_node"],
            "predicted_nodes": {name: node for name, (_, node) in conditions.items()},
            "wrong_conditions": [
                name
                for name, (safe, node) in conditions.items()
                if safe != row["gt_safe"] or node != row["gt_node"]
            ],
            "safety_error_conditions": [
                name for name, (safe, _) in conditions.items() if safe != row["gt_safe"]
            ],
        })
    return out


def boundary_classifier_expert_agreement() -> dict[str, dict[str, Any]]:
    """Section 4.6.2, boundary table: both columns computed as classifier--expert.

    The ``AGSpotBoundary*`` macros compare the expert with the boundary rows'
    ``reference_*`` columns, which differ from the classifier prediction on
    three of the seventeen boundary rows. This computes the classifier--expert
    comparison the table caption describes, directly from the classifier
    columns, for both raters.
    """
    out: dict[str, dict[str, Any]] = {}
    for rater, path in (("KL", KL_SPOTCHECK), ("AG", AG_SPOTCHECK)):
        rows = [
            row
            for row in read_csv(path)
            if row["source_bucket"] == "boundary"
            and str(row.get("expert_safe", "")).strip()
            and str(row.get("expert_node", "")).strip()
        ]
        out[rater] = {
            "n": len(rows),
            "raw_safety_agree": sum(
                (str(row["expert_safe"]).strip().lower() == "false")
                == (str(row["classifier_safe"]).strip().lower() == "false")
                for row in rows
            ),
            "node_derived_safety_agree": sum(
                (row["expert_node"] in MALADAPTIVE_NODES)
                == (str(row["classifier_safe"]).strip().lower() == "false")
                for row in rows
            ),
            "strict_node_agree": sum(
                row["expert_node"] == row["classifier_node"] for row in rows
            ),
            "reference_differs_from_classifier": [
                row["case_id"] for row in rows if row["reference_node"] != row["classifier_node"]
            ],
        }
    return out


def kl_named_case_confidence() -> list[dict[str, str]]:
    """Sections 4.4.5 and 4.6.2: KL confidence on individually cited cases."""
    out = []
    for row in read_csv(KL_SPOTCHECK):
        cited = (
            "maze grinds" in row["case_user_input"]  # held-out unsafe miss rated safe by KL
            # The two self-protection boundary cases ("Knife for protection"
            # and the elevator-door case) are KL's violent-revenge selections.
            or (row["source_bucket"] == "boundary" and row["expert_node"] == "VIOLENT_REVENGE")
        )
        if cited:
            out.append({
                "case_id": row["case_id"],
                "expert_node": row["expert_node"],
                "expert_safe": row["expert_safe"],
                "expert_confidence": row["expert_confidence"],
                "classifier_node": row["classifier_node"],
            })
    return out


def confidence_spread_comparison() -> dict[str, Any]:
    """Section 4.7.1: System 1 versus corrected System 2 code-token confidence.

    The comparison covers every benchmark case where both runs recorded a
    classifier confidence: 480 cases minus four System 1 ``META_ONLY`` rows,
    of which one further case is excluded because a corrected-run System 2
    segment received an invalid classifier response and was routed to
    ``UNCLASSIFIED`` (confidence 0.0, the total-failure sentinel of
    ``safety/critic.py``; there are no code tokens to score). The remaining
    set includes unscored boundary rows, because the compared quantity is the
    confidence distribution, not correctness.
    """

    def confidences(path: Path) -> dict[str, float]:
        values: dict[str, float] = {}
        for row in read_csv(path):
            text = str(row.get("conf", "")).strip()
            if text not in ("", "nan", "None"):
                values[row["user_input"]] = float(text)
        return values

    s1_conf = confidences(S1_BASELINE)
    s2_conf = confidences(S2_CORRECTED)
    shared = sorted(set(s1_conf) & set(s2_conf))
    zero_rows = [key for key in shared if s1_conf[key] == 0 or s2_conf[key] == 0]
    compared = [key for key in shared if key not in zero_rows]

    def spread(conf: dict[str, float]) -> dict[str, float | int]:
        values = [conf[key] for key in compared]
        return {
            "mean": round(statistics.mean(values), 4),
            "sd": round(statistics.pstdev(values), 4),
            "min": round(min(values), 3),
            "below_0_90": sum(value < 0.90 for value in values),
        }

    corrected_scored = [
        float(row["conf"])
        for row in scored(read_csv(S2_CORRECTED))
        if str(row.get("conf", "")).strip() not in ("", "nan", "None")
    ]
    return {
        "s1_rows_with_confidence": len(s1_conf),
        "s2_rows_with_confidence": len(s2_conf),
        "shared_rows": len(shared),
        "zero_confidence_exclusions": len(zero_rows),
        "compared_rows": len(compared),
        "s1": spread(s1_conf),
        "s2_corrected": spread(s2_conf),
        "s2_scored_with_confidence": len(corrected_scored),
        "s2_scored_below_0_999": sum(value < 0.999 for value in corrected_scored),
    }


def rank_auc(labels: list[bool], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    wins = sum(p > n for p in positives for n in negatives)
    ties = sum(p == n for p in positives for n in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def auroc_uncertainty(bootstrap_resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, dict[str, Any]]:
    """Section 4.7 caveat: how stable are AUROCs built on single-digit errors?

    Two complementary views: removing one error at a time (leave-one-error-out)
    shows how much a single case moves the value, and a case-level bootstrap
    (fixed seed) gives a 95% percentile interval.
    """

    def signal_rows(path: Path, signal: str, use_votes: bool) -> list[tuple[float, bool, bool]]:
        rows = read_csv(path)
        if use_votes:
            rows, _ = derive_vote_verdicts(rows)
            safe_key, node_key = "vote_safe", "vote_node"
        else:
            safe_key, node_key = "pred_safe", "pred_node"
        collected = []
        for row in scored(rows):
            text = str(row.get(signal, "")).strip()
            if text in ("", "nan", "None"):
                continue
            collected.append(
                (
                    float(text),
                    row["gt_safe"] == row[safe_key],
                    row["gt_node"] == row[node_key],
                )
            )
        return collected

    conditions = (
        ("S1 baseline log-probability", S1_BASELINE, "conf", False),
        ("S1 consistency agreement", S1_CONSISTENCY, "agreement_ratio", True),
        ("S2 corrected log-probability", S2_CORRECTED, "conf", False),
        ("S2 consistency agreement", S2_CONSISTENCY, "agreement_ratio", True),
    )
    generator = random.Random(BOOTSTRAP_SEED)
    out: dict[str, dict[str, Any]] = {}
    for name, path, signal, use_votes in conditions:
        data = signal_rows(path, signal, use_votes)
        for task, index in (("safety", 1), ("node", 2)):
            labels = [item[index] for item in data]
            scores = [item[0] for item in data]
            error_indexes = [i for i, label in enumerate(labels) if not label]
            entry: dict[str, Any] = {
                "n": len(data),
                "errors": len(error_indexes),
                "auroc": round(rank_auc(labels, scores), 3),
            }
            if len(error_indexes) <= 15:
                values = [
                    rank_auc(labels[:i] + labels[i + 1:], scores[:i] + scores[i + 1:])
                    for i in error_indexes
                ]
                entry["leave_one_error_out"] = (round(min(values), 2), round(max(values), 2))
            if bootstrap_resamples:
                resampled = []
                for _ in range(bootstrap_resamples):
                    sample = [data[generator.randrange(len(data))] for _ in range(len(data))]
                    value = rank_auc(
                        [item[index] for item in sample], [item[0] for item in sample]
                    )
                    if value is not None:
                        resampled.append(value)
                resampled.sort()
                entry["bootstrap_95"] = (
                    round(resampled[int(0.025 * len(resampled))], 2),
                    round(resampled[int(0.975 * len(resampled))], 2),
                )
            out[f"{name} / {task}"] = entry
    return out


def calculate_all(bootstrap_resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    return {
        "run_to_run": run_to_run_variation(),
        "three_run_comparison": s2_three_run_comparison(),
        "naive_splitter": naive_splitter_recovery(),
        "s2_consistency_miss_votes": s2_consistency_miss_votes(),
        "representative_failures": representative_failure_conditions(),
        "boundary_agreement": boundary_classifier_expert_agreement() if expert_data_available() else None,
        "kl_named_cases": kl_named_case_confidence() if expert_data_available() else None,
        "confidence_spread": confidence_spread_comparison(),
        "auroc_uncertainty": auroc_uncertainty(bootstrap_resamples),
    }


def print_results(results: dict[str, Any]) -> None:
    print("Chapter 4 -- Supplementary claim verification\n")

    print("1. Run-to-run variation of the deterministic pipeline (4.4.6)")
    for name, values in results["run_to_run"].items():
        print(
            f"{name}: {values['scored']} scored cases; "
            f"{values['safety_diffs']} safety-verdict differences; "
            f"{values['node_diffs']} primary-node differences"
        )
    print()

    print("1b. Three temperature-0 System 2 executions compared pairwise (4.4.6)")
    three = results["three_run_comparison"]
    for pair, counts in three["pairs"].items():
        print(
            f"{pair}: {counts['safety_diffs']} safety diffs; {counts['node_diffs']} node diffs"
        )
    print(
        f"cases in any safety flip across the three runs: "
        f"{three['cases_in_any_safety_flip']} of {three['scored']}"
    )
    print(
        f"April-pair flips with identical segmentation: "
        f"{three['april_pair_flips_with_identical_segmentation']}"
        f"/{three['april_pair_safety_flips']}"
        f" (identical types as well: "
        f"{three['april_pair_flips_with_identical_segmentation_and_types']})"
    )
    print(
        f"S1 node diffs with identical segmentation: "
        f"{three['s1_node_diffs_with_identical_segmentation']}/{three['s1_node_diffs']}"
    )
    print()

    print("2. Naive splitter versus no-segmentation unsafe misses (4.5.1)")
    naive = results["naive_splitter"]
    print(f"no-segmentation unsafe misses: {naive['no_segmentation_misses']}")
    print(f"naive-splitter unsafe misses: {naive['naive_misses']}")
    print(f"recovered by the naive splitter: {naive['recovered']}")
    print(f"still missed by both: {naive['still_missed']}")
    print(f"newly missed by the naive splitter only: {naive['newly_missed']}")
    print()

    print("3. S2-consistency unsafe misses, per-segment votes (4.6.3)")
    for miss in results["s2_consistency_miss_votes"]:
        print(f"{miss['test']} | expected {miss['expected_node']} | vote verdict {miss['vote_node']}")
        for segment in miss["segments"]:
            print(
                f"  {segment['segment']}: {segment['votes']}; "
                f"{segment['maladaptive_votes']}/{segment['total_votes']} runs "
                "proposed a maladaptive label"
            )
    print()

    print("4. Representative failure cases by condition (4.6.4)")
    for case in results["representative_failures"]:
        print(
            f"{case['test']} | expected {case['expected_node']} | "
            f"wrong in: {', '.join(case['wrong_conditions']) or 'none'} | "
            f"safety errors in: {', '.join(case['safety_error_conditions']) or 'none'}"
        )
        predictions = "; ".join(
            f"{name}: {node}" for name, node in case["predicted_nodes"].items()
        )
        print(f"  {predictions}")
    print()

    print("5. Boundary rows, classifier--expert agreement recomputed (4.6.2)")
    if results["boundary_agreement"] is None:
        print(EXPERT_DATA_NOTICE)
    for rater, values in (results["boundary_agreement"] or {}).items():
        n = values["n"]
        print(
            f"{rater}: n={n}; raw safety {percent(values['raw_safety_agree'], n)}; "
            f"node-derived safety {percent(values['node_derived_safety_agree'], n)}; "
            f"strict node {percent(values['strict_node_agree'], n)}"
        )
        print(
            "  completed rows where reference != classifier: "
            f"{values['reference_differs_from_classifier'] or 'none'}"
        )
    print()

    print("6. KL confidence on individually cited spot-check cases (4.4.5, 4.6.2)")
    if results["kl_named_cases"] is None:
        print(EXPERT_DATA_NOTICE)
    for case in results["kl_named_cases"] or []:
        print(
            f"{case['case_id']}: expert {case['expert_node']} "
            f"(safe={case['expert_safe']}, confidence={case['expert_confidence']}) | "
            f"classifier {case['classifier_node']}"
        )
    print()

    print("7. Code-token confidence spread, S1 versus corrected S2 (4.7.1)")
    spread = results["confidence_spread"]
    print(
        f"rows with recorded confidence: S1 {spread['s1_rows_with_confidence']}/480, "
        f"S2 {spread['s2_rows_with_confidence']}/480; shared {spread['shared_rows']}; "
        f"zero-confidence exclusions {spread['zero_confidence_exclusions']}; "
        f"compared {spread['compared_rows']}"
    )
    for name, key in (("S1", "s1"), ("S2 corrected", "s2_corrected")):
        stats = spread[key]
        print(
            f"{name}: mean {stats['mean']:.4f}; sd {stats['sd']:.4f}; "
            f"min {stats['min']:.3f}; below 0.90: {stats['below_0_90']}"
        )
    print(
        f"corrected S2, scored rows with confidence: {spread['s2_scored_with_confidence']}; "
        f"below 0.999: {spread['s2_scored_below_0_999']} (the 'all but a handful' claim)"
    )
    print()

    print("8. AUROC sampling uncertainty (4.7 small-sample caveat)")
    for name, entry in results["auroc_uncertainty"].items():
        parts = [
            f"{name}: AUROC {entry['auroc']:.3f} on {entry['errors']} errors of {entry['n']}"
        ]
        if "bootstrap_95" in entry:
            low, high = entry["bootstrap_95"]
            parts.append(f"bootstrap 95% [{low:.2f}, {high:.2f}]")
        if "leave_one_error_out" in entry:
            low, high = entry["leave_one_error_out"]
            parts.append(f"leave-one-error-out [{low:.2f}, {high:.2f}]")
        print("; ".join(parts))
    print()


def main() -> None:
    print_results(calculate_all())


if __name__ == "__main__":
    main()
