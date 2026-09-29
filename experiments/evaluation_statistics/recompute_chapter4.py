"""Recompute the statistics cited in thesis Chapter 4 from checked-in data.

This script is the canonical, non-notebook calculation path for the evaluation
chapter. It deliberately uses only the Python standard library so that the
reported counts can be audited without reconstructing a notebook environment.

Run from the repository root:

    uv run python experiments/evaluation_statistics/recompute_chapter4.py

Generated artifacts are written to ``data/evaluation_statistics/chapter4``.
The JSON file is the machine-readable source of truth; the Markdown report is
the human-readable audit view; and the TeX file contains the macro values
interpolated directly into the thesis (dataset composition, expert spot-check,
and the classification-results / agreement-distribution numbers used by the
Chapter 4 rewrite).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments.evaluation_statistics.consistency_votes import (
    derive_vote_verdicts,
    verify_pipeline_triage,
)
from experiments.expert_analysis.analyze_expert_annotations import (
    build_paired_system_rows,
    cohen_kappa,
    completed_spotcheck_rows,
    derive_spotcheck_fields,
    merge_spotcheck,
    node_is_maladaptive,
    row_slice,
    spotcheck_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation_statistics/chapter4"

# The row-level expert annotation tables are personal data of the two raters
# and are distributed only with their consent. When they are absent, the
# benchmark statistics are recomputed as usual and the expert-derived blocks
# are carried over from the canonical chapter4_statistics.json; provenance.json
# records which inputs were withheld and which blocks were carried over.
EXPERT_INPUT_KEYS = (
    "ag_spotcheck",
    "ag_system_raw",
    "kl_spotcheck_merged",
    "ag_system_merged",
    "kl_system_merged",
)
EXPERT_DERIVED_BLOCKS = (
    "dataset_composition.spotcheck_queue_test_types",
    "uncertainty.kl_spotcheck_confidence",
    "expert_spotcheck",
    "system_expert_evaluation",
)


def expert_inputs_available() -> bool:
    return all((REPO_ROOT / OTHER_INPUTS[key]).exists() for key in EXPERT_INPUT_KEYS)


def load_canonical_statistics() -> dict[str, Any]:
    path = DEFAULT_OUTPUT_DIR / "chapter4_statistics.json"
    if not path.exists():
        raise SystemExit(
            "The expert annotation inputs are absent and no canonical "
            f"{path.relative_to(REPO_ROOT)} exists to carry their aggregates from."
        )
    return json.loads(path.read_text(encoding="utf-8"))

MALADAPTIVE_NODES = {
    "AVOIDANCE",
    "INTERRUPTION",
    "VIOLENT_REVENGE",
    "SUPPRESSION",
    "TRAUMA_REPLAY",
}

ADAPTIVE_NODES = {
    "BEHAVIORAL_MASTERY",
    "SOCIAL_MASTERY",
    "ENVIRONMENTAL_MASTERY",
    "EMOTIONAL_MASTERY",
    "MYTHICAL_MASTERY",
}

NEUTRAL_NODES = {"AFFECT_EXPRESSION", "NARRATIVE_SETTING"}

# Categories whose assignment makes the turn unsafe. Triage branches on the
# presence of a maladaptive segment and takes the primary node from it
# (safety/critic.py:503,551); when no maladaptive segment exists but one
# segment could not be classified, the turn is still returned unsafe with
# UNCLASSIFIED as its node (safety/critic.py:533). Everything else --- the
# adaptive and neutral categories and the META_ONLY filtering outcome --- is
# returned safe. The verdict is therefore a function of the predicted node,
# which is what lets §4.6.1 call the error partition a guarantee rather than
# an observation. This is NOT the same split as MALADAPTIVE_NODES: that one
# is clinical and is what §4.4.6 reports, this one is about the verdict.
UNSAFE_NODE_SIDE = MALADAPTIVE_NODES | {"UNCLASSIFIED"}

BENCHMARK_INPUTS = {
    "s1_baseline": "data/benchmarks/dev_s1_baseline.csv",
    "s1_consistency": "data/benchmarks/dev_s1_consistency_n5.csv",
    "s2_baseline": "data/benchmarks/dev_s2_baseline.csv",
    "s2_consistency": "data/benchmarks/dev_s2_consistency_n5.csv",
    # Re-run after the classification-token extraction fix. The `conf` column of
    # the two S2 runs above came from a defective extractor that fell back to the
    # mean log probability of the whole JSON response on 87.1% of calls, so those
    # values measure rationale fluency rather than label certainty
    # (docs/planning/S2_logprob_diagnostic_findings.md). Classification outcomes
    # are unaffected because nothing gates on confidence at threshold 0.0.
    "s2_baseline_corrected": "data/benchmarks/dev_s2_baseline_rerun_corrected_logprobs.csv",
}

ABLATION_INPUTS = {
    "full_s1": "data/benchmarks/dev_s1_baseline.csv",
    "stage_merge": "data/benchmarks/dev_s1_ablation_stage_merge.csv",
    "no_segmentation": "data/benchmarks/dev_s1_ablation_no_segmentation.csv",
    "meta_filtering_off": "data/benchmarks/dev_s1_ablation_meta_filter_off.csv",
    "naive_splitter": "data/benchmarks/dev_s1_ablation_naive_splitter.csv",
}

MAL_MAL_INPUTS = {
    "full_s1": "data/benchmarks/multi_maladaptive_s1_baseline.csv",
    "stage_merge": "data/benchmarks/multi_maladaptive_s1_ablation_stage_merge.csv",
    "no_segmentation": "data/benchmarks/multi_maladaptive_s1_ablation_no_segmentation.csv",
    "meta_filtering_off": "data/benchmarks/multi_maladaptive_s1_ablation_meta_filter_off.csv",
    "naive_splitter": "data/benchmarks/multi_maladaptive_s1_ablation_naive_splitter.csv",
}

OTHER_INPUTS = {
    "dev_dataset": "data/benchmarks/dev_dataset.json",
    "mal_mal_dataset": "data/benchmarks/multi_maladaptive_dataset.json",
    "test_dataset": "data/benchmarks/test_dataset.json",
    "test_postfix": "data/benchmarks/test_s1_baseline.csv",
    "spotcheck_key": "data/expert_spotcheck/spotcheck_key.csv",
    "spotcheck_blinded": "data/expert_spotcheck/spotcheck_cases_blinded.json",
    "spotcheck_predictions": "data/benchmarks/test_s1_baseline.csv",
    "ag_spotcheck": "data/expert_analysis/ag_2026_07/raw_exports/redream_spotcheck_AG_2026-07-18T14-58-38-137Z.json",
    "ag_system_raw": "data/expert_analysis/ag_2026_07/raw_exports/redream_system_eval_AG_2026-07-18T14-59-13-387Z.json",
    "kl_spotcheck_merged": "data/expert_analysis/kl_2026_06/spotcheck_merged.csv",
    "ag_system_merged": "data/expert_analysis/ag_2026_07/system_eval_merged.csv",
    "kl_system_merged": "data/expert_analysis/kl_2026_06/system_eval_merged.csv",
    "integration_eval": "data/integration_eval/integration_eval_results_v2.csv",
    "s2_logprob_diagnostic": "data/benchmarks/dev_s2_logprob_diagnostic.csv",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def rounded(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_index(path: Path) -> dict[str, dict[str, Any]]:
    """Index generated cases by exact input text for joining result CSVs."""
    suites = json.loads(path.read_text(encoding="utf-8"))
    return {
        test["input"]: {**test, "suite": suite["suite"]}
        for suite in suites
        for test in suite["tests"]
    }


def dataset_composition(path: Path, exclude_suites: tuple[str, ...] = ()) -> dict[str, Any]:
    """Case counts by test type, safety group, and node for one dataset file.

    Backs the Chapter 4.1 composition prose (per-node case ranges and the
    adaptive/neutral/maladaptive splits) so those numbers regenerate with the
    data instead of living as hand-checked literals.
    """
    suites = json.loads(path.read_text(encoding="utf-8"))
    tests = [
        test
        for suite in suites
        if not any(marker in suite["suite"].lower() for marker in exclude_suites)
        for test in suite["tests"]
    ]
    node_counts = Counter(t["expected_node"] for t in tests if t.get("expected_node"))
    groups = {
        "adaptive": sum(c for n, c in node_counts.items() if n in ADAPTIVE_NODES),
        "neutral": sum(c for n, c in node_counts.items() if n in NEUTRAL_NODES),
        "maladaptive": sum(c for n, c in node_counts.items() if n in MALADAPTIVE_NODES),
    }
    scored = sum(node_counts.values())
    if sum(groups.values()) != scored:
        raise SystemExit(f"{path}: node group counts do not partition the scored cases")
    return {
        "n_cases": len(tests),
        "test_types": dict(Counter(t.get("test_type") for t in tests)),
        "safety_groups": groups,
        "node_case_min": min(node_counts.values()),
        "node_case_max": max(node_counts.values()),
        "node_counts": dict(sorted(node_counts.items())),
    }


def scored_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if as_bool(row.get("gt_safe")) is not None]


def benchmark_metrics(
    rows: list[dict[str, Any]],
    pred_safe_key: str = "pred_safe",
    pred_node_key: str = "pred_node",
) -> dict[str, Any]:
    """Case-level metrics for one condition.

    The prediction keys are configurable so the semantic-consistency conditions
    can be scored on their majority-vote verdict (``vote_safe``/``vote_node``,
    see ``consistency_votes.py``) rather than on the temperature-0 pipeline
    output that ``run_benchmark.py`` also writes to those rows.
    """
    scored = scored_rows(rows)
    parsed = [
        (row, as_bool(row.get("gt_safe")), as_bool(row.get(pred_safe_key)))
        for row in scored
    ]
    tp = sum(gt is False and pred is False for _row, gt, pred in parsed)
    fn = sum(gt is False and pred is True for _row, gt, pred in parsed)
    fp = sum(gt is True and pred is False for _row, gt, pred in parsed)
    tn = sum(gt is True and pred is True for _row, gt, pred in parsed)
    safety_correct = tp + tn
    unsafe_total = tp + fn
    predicted_unsafe = tp + fp
    node_correct = sum(row.get("gt_node") == row.get(pred_node_key) for row in scored)
    precision = ratio(tp, predicted_unsafe)
    recall = ratio(tp, unsafe_total)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )

    per_node: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[row.get("gt_node", "")].append(row)
    for node, group in sorted(grouped.items()):
        safety_ok = sum(as_bool(row.get("gt_safe")) == as_bool(row.get(pred_safe_key)) for row in group)
        node_ok = sum(row.get("gt_node") == row.get(pred_node_key) for row in group)
        per_node[node] = {
            "n": len(group),
            "safety_correct": safety_ok,
            "safety_accuracy": rounded(ratio(safety_ok, len(group)), 6),
            "node_correct": node_ok,
            "node_accuracy": rounded(ratio(node_ok, len(group)), 6),
        }

    collapsed_correct = sum(
        row.get("gt_node") == row.get(pred_node_key)
        or {row.get("gt_node"), row.get(pred_node_key)} == {"AFFECT_EXPRESSION", "EMOTIONAL_MASTERY"}
        for row in scored
    )
    return {
        "n_rows": len(rows),
        "n_scored": len(scored),
        "n_unscored": len(rows) - len(scored),
        "n_safe": tn + fp,
        "n_unsafe": unsafe_total,
        "safety_correct": safety_correct,
        "safety_accuracy": rounded(ratio(safety_correct, len(scored)), 6),
        "unsafe_true_positive": tp,
        "unsafe_false_negative": fn,
        "unsafe_false_positive": fp,
        "safe_true_negative": tn,
        "unsafe_recall": rounded(recall, 6),
        "unsafe_precision": rounded(precision, 6),
        "unsafe_f1": rounded(f1, 6),
        "node_correct": node_correct,
        "node_accuracy": rounded(ratio(node_correct, len(scored)), 6),
        "affect_emotional_collapsed_node_correct": collapsed_correct,
        "affect_emotional_collapsed_node_accuracy": rounded(ratio(collapsed_correct, len(scored)), 6),
        "meta_only_outputs": sum(row.get(pred_node_key) == "META_ONLY" for row in scored),
        "per_node": per_node,
    }


def rerun_divergence(
    baseline_rows: list[dict[str, Any]],
    rerun_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Row-level disagreement between two temperature-0 runs of one config.

    The consistency CSVs carry the temperature-0 pipeline prediction of their
    own run alongside the votes, so comparing those predictions with the
    baseline run's on the same cases counts pure run-to-run variation of the
    deterministic pipeline (thesis §4.4.5 and §5.3).
    """
    base = {row.get("user_input", ""): row for row in scored_rows(baseline_rows)}
    matched = safety_flips = node_flips = 0
    for row in scored_rows(rerun_rows):
        other = base.get(row.get("user_input", ""))
        if other is None:
            continue
        matched += 1
        safety_flips += as_bool(row.get("pred_safe")) != as_bool(other.get("pred_safe"))
        node_flips += row.get("pred_node") != other.get("pred_node")
    return {
        "n_matched": matched,
        "safety_verdict_flips": safety_flips,
        "node_flips": node_flips,
    }


def parse_segments(seg_str: str) -> list[tuple[str, str]]:
    """Parse the compact ``segments`` audit column into (type, text) pairs.

    Entries look like ``[ACTION] ctx:... "text..." → NODE (82%)`` joined by
    `` | ``; META segments carry neither context nor label (``→ ? (?)``).
    The quoted text is taken between the first and the last quote before the
    arrow, so quotation marks inside dream text cannot break the parse.
    """
    pairs: list[tuple[str, str]] = []
    for part in seg_str.split(" | "):
        head = part.rsplit("→", 1)[0]
        seg_type = head[1 : head.find("]")] if head.startswith("[") else ""
        first, last = head.find('"'), head.rfind('"')
        text = head[first + 1 : last] if 0 <= first < last else ""
        pairs.append((seg_type, text))
    return pairs


def segment_identity_on_flips(
    baseline_rows: list[dict[str, Any]],
    rerun_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Segment-level agreement on the safety-verdict flips between two runs.

    For each scored case whose safety verdict differs between two
    temperature-0 executions, checks whether the two runs drew the same
    segment boundaries (same quoted segment texts, in order) and, where they
    did, whether the segment types also match. Locates the run-to-run
    variation in segment typing and per-segment classification rather than
    in boundary placement (thesis §4.4.5).
    """
    base = {row.get("user_input", ""): row for row in scored_rows(baseline_rows)}
    flips = same_boundaries = same_boundaries_and_types = 0
    for row in scored_rows(rerun_rows):
        other = base.get(row.get("user_input", ""))
        if other is None:
            continue
        if as_bool(row.get("pred_safe")) == as_bool(other.get("pred_safe")):
            continue
        flips += 1
        a = parse_segments(other.get("segments", ""))
        b = parse_segments(row.get("segments", ""))
        if [text for _, text in a] == [text for _, text in b]:
            same_boundaries += 1
            if [seg_type for seg_type, _ in a] == [seg_type for seg_type, _ in b]:
                same_boundaries_and_types += 1
    return {
        "safety_flips": flips,
        "same_segment_boundaries": same_boundaries,
        "same_boundaries_and_types": same_boundaries_and_types,
    }


def multi_run_safety_stability(runs: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Union of row-level safety-verdict differences across several runs.

    Counts, over the cases scored in every run, how many received a
    non-identical safety verdict anywhere across the executions. With three
    runs this equals half the sum of the pairwise flip counts, because every
    differing case has exactly one odd-one-out execution (thesis §4.4.5).
    """
    maps = [{row.get("user_input", ""): row for row in scored_rows(rows)} for rows in runs]
    common = set(maps[0])
    for other in maps[1:]:
        common &= set(other)
    changed = sum(
        1
        for user_input in common
        if len({as_bool(m[user_input].get("pred_safe")) for m in maps}) > 1
    )
    return {
        "n_matched": len(common),
        "cases_with_safety_flip": changed,
        "cases_stable": len(common) - changed,
    }


def confusion_families(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Family breakdown of strict primary-node errors (thesis §4.4.6).

    Splits the node confusions of a run into errors that involve no
    maladaptive node, confusions between two maladaptive patterns (which
    leave the safety verdict unchanged, both sides being unsafe), and
    confusions that cross the maladaptive boundary. Every pair involving a
    maladaptive node is also listed individually.
    """
    n_errors = non_mal = within_mal = cross_family = 0
    mal_pairs: Counter[str] = Counter()
    for row in scored_rows(rows):
        gt, pred = row.get("gt_node"), row.get("pred_node")
        if gt == pred:
            continue
        n_errors += 1
        gt_mal = gt in MALADAPTIVE_NODES
        pred_mal = pred in MALADAPTIVE_NODES
        if gt_mal and pred_mal:
            within_mal += 1
        elif gt_mal or pred_mal:
            cross_family += 1
        else:
            non_mal += 1
        if gt_mal or pred_mal:
            mal_pairs[f"{gt} -> {pred}"] += 1
    return {
        "node_errors": n_errors,
        "non_maladaptive": non_mal,
        "within_maladaptive": within_mal,
        "cross_family": cross_family,
        "pairs_involving_maladaptive": dict(
            sorted(mal_pairs.items(), key=lambda item: (-item[1], item[0]))
        ),
    }



def directional_confusions(
    rows: Iterable[dict[str, Any]],
    min_count: int = 6,
) -> dict[str, Any]:
    """Ordered confusion pairs and their reverse counts (thesis §4.6.1).

    The §4.6.1 claim is that the *large* confusions run in one direction only.
    Sparse pairs cannot support that claim -- at one to four cases a 2/0 split
    is noise -- so only pairs at or above ``min_count`` are returned, together
    with the count of the same pair in the opposite direction. ``max_reverse``
    is the largest reverse count among them: the asymmetry claim holds exactly
    when it is zero.
    """
    pairs: Counter[tuple[str, str]] = Counter()
    for row in scored_rows(rows):
        gt, pred = row.get("gt_node"), row.get("pred_node")
        if gt != pred:
            pairs[(gt, pred)] += 1
    large = {
        f"{gt} -> {pred}": {"count": count, "reverse": pairs.get((pred, gt), 0)}
        for (gt, pred), count in pairs.most_common()
        if count >= min_count
    }
    one_directional = {k: v for k, v in large.items() if v["reverse"] == 0}
    bidirectional_pairs = [
        {gt, pred} for (gt, pred) in pairs if pairs.get((pred, gt), 0)
    ]
    shared: set[str] = (
        set.intersection(*bidirectional_pairs) if bidirectional_pairs else set()
    )
    return {
        "min_count": min_count,
        "pairs": large,
        "n_large": len(large),
        "n_one_directional": len(one_directional),
        "one_directional": one_directional,
        "max_reverse": max((p["reverse"] for p in large.values()), default=0),
        # Categories present in EVERY bidirectional pair. §4.6.1 names
        # behavioral mastery as the one category confused in both directions;
        # a node listed here is on both sides of every such pair.
        "shared_by_all_bidirectional_pairs": sorted(shared),
    }


def node_flow(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Per-category error in/out flow and prediction counts (thesis §4.6.1).

    ``lost`` counts reference cases of a category assigned elsewhere and
    ``gained`` counts errors arriving at it, so ``net`` separates categories
    that absorb cases from those that leak them. ``predicted`` and
    ``precision`` additionally expose categories the classifier has all but
    stopped emitting, which a recall figure alone does not show. The
    ``gained_from``/``lost_to`` breakdowns back §4.6.1's source-breadth
    sentence (behavioral mastery drawing from more source categories than
    emotional or environmental mastery) with per-pair counts.
    """
    scored = scored_rows(rows)
    lost: Counter[str] = Counter()
    gained: Counter[str] = Counter()
    size: Counter[str] = Counter()
    predicted: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    gained_from: dict[str, Counter[str]] = defaultdict(Counter)
    lost_to: dict[str, Counter[str]] = defaultdict(Counter)
    for row in scored:
        gt, pred = row.get("gt_node"), row.get("pred_node")
        size[gt] += 1
        predicted[pred] += 1
        if gt == pred:
            correct[gt] += 1
        else:
            lost[gt] += 1
            gained[pred] += 1
            gained_from[pred][gt] += 1
            lost_to[gt][pred] += 1

    def by_count(pairs: Counter[str]) -> dict[str, int]:
        return dict(sorted(pairs.items(), key=lambda item: (-item[1], item[0])))

    return {
        node: {
            "reference_cases": size[node],
            "correct": correct[node],
            "lost": lost[node],
            "gained": gained[node],
            "net": gained[node] - lost[node],
            "predicted": predicted[node],
            "recall": rounded(ratio(correct[node], size[node]), 6),
            "precision": rounded(ratio(correct[node], predicted[node]), 6),
            "gained_from": by_count(gained_from[node]),
            "lost_to": by_count(lost_to[node]),
        }
        for node in sorted(set(size) | set(predicted))
    }


def leading_confusion(
    rows: Iterable[dict[str, Any]],
    pred_node_key: str = "pred_node",
) -> dict[str, Any]:
    """Most frequent ordered confusion pair of one condition (thesis §4.6.1).

    §4.6.1 states that affect expression -> emotional mastery is the most
    frequent confusion in every reported condition, including held-out.
    ``margin`` is the lead over the runner-up, so a tie for first place shows
    up as zero instead of being hidden by an arbitrary ranking.
    """
    pairs: Counter[tuple[str, str]] = Counter()
    for row in scored_rows(rows):
        gt, pred = row.get("gt_node"), row.get(pred_node_key)
        if gt != pred:
            pairs[(gt, pred)] += 1
    ranked = pairs.most_common(2)
    if not ranked:
        return {"pair": None, "count": 0, "runner_up": None, "runner_up_count": 0, "margin": 0}
    (top_gt, top_pred), top_count = ranked[0]
    runner_up, runner_up_count = (None, 0)
    if len(ranked) > 1:
        (run_gt, run_pred), runner_up_count = ranked[1]
        runner_up = f"{run_gt} -> {run_pred}"
    return {
        "pair": f"{top_gt} -> {top_pred}",
        "count": top_count,
        "runner_up": runner_up,
        "runner_up_count": runner_up_count,
        "margin": top_count - runner_up_count,
    }


def verdict_partition(
    rows: Iterable[dict[str, Any]],
    pred_safe_key: str = "pred_safe",
    pred_node_key: str = "pred_node",
) -> dict[str, Any]:
    """Node errors split by the side of the verdict boundary they stay on.

    Because the safety verdict is a function of the predicted node
    (``UNSAFE_NODE_SIDE``), a node error that keeps both the reference and the
    assigned category on one side cannot change the verdict, and the errors
    that cross the boundary are exactly the safety errors. ``matches_safety_errors``
    checks that identity on the given rows; it holds for every condition
    reported in Chapter 4, so a False here means the pipeline's triage rule
    changed and §4.6.1's guarantee no longer applies.

    The identity is keyed on row positions, so two rows that happen to share a
    ``user_input`` are counted separately and a crossing error on one row never
    excuses a safety error on another. The prediction keys are configurable for
    the same reason as in ``benchmark_metrics``: the consistency conditions are
    reported on their majority-vote verdict (``vote_safe``/``vote_node``).
    """
    scored = scored_rows(rows)
    safe_side = unsafe_side = 0
    crossing: set[int] = set()
    for index, row in enumerate(scored):
        gt, pred = row.get("gt_node"), row.get(pred_node_key)
        if gt == pred:
            continue
        gt_unsafe, pred_unsafe = gt in UNSAFE_NODE_SIDE, pred in UNSAFE_NODE_SIDE
        if gt_unsafe and pred_unsafe:
            unsafe_side += 1
        elif not gt_unsafe and not pred_unsafe:
            safe_side += 1
        else:
            crossing.add(index)
    safety_errors = {
        index
        for index, row in enumerate(scored)
        if as_bool(row.get("gt_safe")) is not as_bool(row.get(pred_safe_key))
    }
    return {
        "errors_within_safe_side": safe_side,
        "errors_within_unsafe_side": unsafe_side,
        "errors_crossing": len(crossing),
        "n_safety_errors": len(safety_errors),
        "matches_safety_errors": crossing == safety_errors,
    }


def missed_unsafe_overlap(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
) -> dict[str, Any]:
    """Row-level overlap of missed unsafe cases between two runs (§4.5.1).

    Backs claims of the form "condition B recovers N of the cases condition A
    misses but misses M cases A catches", which aggregate miss counts cannot
    support.
    """

    def missed(rows: list[dict[str, Any]]) -> set[str]:
        return {
            row.get("user_input", "")
            for row in scored_rows(rows)
            if as_bool(row.get("gt_safe")) is False and as_bool(row.get("pred_safe"))
        }

    a, b = missed(rows_a), missed(rows_b)
    return {
        "missed_a": len(a),
        "missed_b": len(b),
        "shared": len(a & b),
        "recovered_by_b": len(a - b),
        "introduced_by_b": len(b - a),
    }


def vote_count_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-segment vote counts of a consistency run (thesis §4.4.4).

    ``vote_distribution`` holds one JSON object per case mapping segment keys
    to their code->vote-count tallies; a segment's completed calls are the sum
    of its tallies. Rows without votes are the META_ONLY cases. Scope is the
    FULL run (scored and unscored rows alike): the claim is about the voting
    mechanics of the run, not about a metric denominator.
    """
    per_segment_votes: list[int] = []
    rows_without_votes = 0
    for case in rows:
        raw = str(case.get("vote_distribution", "")).strip()
        if raw in {"", "{}", "nan", "None"}:
            rows_without_votes += 1
            continue
        for tally in json.loads(raw).values():
            per_segment_votes.append(sum(tally.values()))
    counts = Counter(per_segment_votes)
    return {
        "n_segments": len(per_segment_votes),
        "rows_without_votes": rows_without_votes,
        "segments_with_five_votes": counts.get(5, 0),
        "segments_below_nominal": sum(n for votes, n in counts.items() if votes < 5),
        "segments_with_zero_votes": counts.get(0, 0),
        "min_votes": min(per_segment_votes) if per_segment_votes else None,
        "vote_count_distribution": dict(sorted(counts.items())),
    }


def agreement_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Distribution of the per-case agreement ratio in a consistency run.

    The per-case value is the minimum agreement across the case's classified
    segments (run_benchmark.py's conservative weakest-link rule), so
    ``unanimous`` counts cases whose every segment vote was 5/5.
    """
    values = [
        float(row["agreement_ratio"])
        for row in scored_rows(rows)
        if str(row.get("agreement_ratio", "")).strip() not in {"", "nan", "None"}
    ]
    if not values:
        return {"n": 0}
    unanimous = sum(value >= 1.0 for value in values)
    return {
        "n": len(values),
        "unanimous": unanimous,
        "unanimous_share": rounded(ratio(unanimous, len(values)), 6),
        "mean": rounded(sum(values) / len(values), 6),
        "min": rounded(min(values), 6),
    }


def signal_values(rows: list[dict[str, Any]], signal: str) -> list[float]:
    return [
        float(row[signal])
        for row in scored_rows(rows)
        if str(row.get(signal, "")).strip() not in {"", "nan", "None"}
    ]


def signal_histogram(
    values: list[float], lo: float, hi: float, width: float
) -> dict[str, Any]:
    """Fixed-width histogram over [lo, hi); values below lo listed verbatim.

    Feeds the §4.7 confidence-distribution figure. Bin i covers
    [lo + i*width, lo + (i+1)*width); no scored confidence reaches 1.0
    exactly, so the top edge is never hit.
    """
    n_bins = int(round((hi - lo) / width))
    counts = [0] * n_bins
    below = []
    for value in values:
        if value >= hi:
            raise SystemExit(f"histogram value {value} at or above top edge {hi}")
        if value < lo:
            below.append(rounded(value, 6))
            continue
        counts[min(n_bins - 1, int((value - lo + 1e-12) / width))] += 1
    return {
        "n": len(values),
        "bin_lo": lo,
        "bin_hi": hi,
        "bin_width": width,
        "counts": counts,
        "below_range_values": sorted(below),
    }


def uncertainty_distributions(
    s1_rows: list[dict[str, Any]], s2_rows: list[dict[str, Any]],
    s2_cons_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Distribution data behind the §4.7 confidence-distribution figure.

    Panels: System 1 log-probability confidence, corrected System 2
    log-probability confidence (with a zoom over its saturated range), and
    the System 2 consistency agreement ratio (discrete values).
    """
    s1_usable, s1_excluded = conf_usable_rows(s1_rows, "conf", exclude_unclassified=True)
    s2_usable, s2_excluded = conf_usable_rows(s2_rows, "conf", exclude_unclassified=True)
    s1_conf = [float(row["conf"]) for row in s1_usable]
    s2_conf = [float(row["conf"]) for row in s2_usable]
    agreement = signal_values(s2_cons_rows, "agreement_ratio")

    s1_hist = signal_histogram(s1_conf, 0.60, 1.00, 0.02)
    s1_hist["n_excluded_unclassified"] = s1_excluded
    s1_hist["n_below_0_90"] = sum(value < 0.90 for value in s1_conf)

    s2_hist = signal_histogram(s2_conf, 0.60, 1.00, 0.02)
    s2_hist["n_excluded_unclassified"] = s2_excluded
    s2_in_band = [value for value in s2_conf if value >= 0.9990]
    s2_hist["n_in_0_999_1"] = len(s2_in_band)
    s2_hist["n_below_0_90"] = sum(value < 0.90 for value in s2_conf)
    s2_hist["n_below_0_9999"] = sum(value < 0.9999 for value in s2_conf)
    s2_hist["zoom"] = signal_histogram(s2_in_band, 0.9990, 1.0000, 0.0001)
    # §4.7.1 prose: every measured safety error sits inside the saturated
    # band, which is why no cutoff catches them without flagging large
    # parts of the benchmark.
    s2_errors = [
        float(row["conf"]) for row in s2_usable
        if as_bool(row.get("gt_safe")) != as_bool(row.get("pred_safe"))
    ]
    s2_hist["n_safety_errors"] = len(s2_errors)
    s2_hist["n_safety_errors_in_0_999_1"] = sum(v >= 0.9990 for v in s2_errors)

    value_counts = Counter(f"{value:.1f}" for value in agreement)
    if not set(value_counts) <= {"0.2", "0.4", "0.6", "0.8", "1.0"}:
        raise SystemExit(
            f"unexpected agreement-ratio values in s2_consistency: {sorted(value_counts)}"
        )

    # §4.7 figure, panel 3 stacking (added 2026-09-15): agreement values
    # split by majority-vote node correctness — the SAME correctness the
    # §4.7.2 agreement AUROC is computed against (uncertainty_metrics with
    # vote_safe/vote_node), so figure and prose share one definition.
    agree_split: dict[str, Counter] = {"node_correct": Counter(), "node_incorrect": Counter()}
    for row in scored_rows(s2_cons_rows):
        if str(row.get("agreement_ratio", "")).strip() in {"", "nan", "None"}:
            continue
        label = "node_correct" if row.get("gt_node") == row.get("vote_node") else "node_incorrect"
        agree_split[label][f"{float(row['agreement_ratio']):.1f}"] += 1
    for value, total in value_counts.items():
        stacked = agree_split["node_correct"][value] + agree_split["node_incorrect"][value]
        if stacked != total:
            raise SystemExit(
                f"agreement node-correctness split does not stack at {value}: {stacked} != {total}"
            )

    # §4.7.1 prose: confidence of node-correct vs node-incorrect predictions.
    # UNCLASSIFIED sentinel rows are already excluded above, so both groups
    # contain measured probabilities only; the prose reports medians
    # (fourth-decimal separation for corrected System 2).
    for hist, usable in ((s1_hist, s1_usable), (s2_hist, s2_usable)):
        groups: dict[bool, list[float]] = {True: [], False: []}
        for row in usable:
            groups[row.get("gt_node") == row.get("pred_node")].append(float(row["conf"]))
        split = {}
        for label, values in (("node_correct", groups[True]), ("node_incorrect", groups[False])):
            values.sort()
            split[label] = {
                "n": len(values),
                "mean": rounded(sum(values) / len(values), 6),
                "median": rounded(statistics.median(values), 6),
                # per-bin counts on the parent histogram's bins, for the
                # §4.7 figure's correctness stacking (added 2026-09-15)
                "counts": signal_histogram(values, 0.60, 1.00, 0.02)["counts"],
            }
        if split["node_correct"]["n"] + split["node_incorrect"]["n"] != hist["n"]:
            raise SystemExit("node-correctness split does not partition the histogram cases")
        stacked = [
            c + i for c, i in zip(split["node_correct"]["counts"],
                                  split["node_incorrect"]["counts"])
        ]
        if stacked != hist["counts"]:
            raise SystemExit("node-correctness split counts do not stack to the histogram")
        hist["node_correctness_split"] = split

    return {
        "s1_baseline_conf": s1_hist,
        "s2_corrected_conf": s2_hist,
        "s2_consistency_agreement": {
            "n": len(agreement),
            "value_counts": dict(sorted(value_counts.items())),
            "node_correctness_split": {
                label: {
                    "n": sum(counter.values()),
                    "value_counts": dict(sorted(counter.items())),
                }
                for label, counter in agree_split.items()
            },
        },
    }


def s2_confidence_run_jitter(
    benchmark_rows: list[dict[str, Any]], diagnostic_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Per-segment confidence differences across two temp-0 S2 executions.

    Joins the corrected-confidence benchmark run's per-segment confidences
    (segments_json) with the post-fix logprob diagnostic run on
    (test, segment text). Backs §4.7.1's threshold-stability argument: a
    cutoff in the fourth decimal sits at the pipeline's own run-to-run
    noise floor.
    """
    bench: dict[tuple[str, str], float] = {}
    for row in benchmark_rows:
        try:
            segments = json.loads(row.get("segments_json") or "[]")
        except json.JSONDecodeError:
            continue
        for segment in segments:
            if segment.get("confidence") is not None:
                key = (row.get("test", ""), str(segment.get("text", "")).strip())
                bench[key] = float(segment["confidence"])
    deltas = []
    for row in diagnostic_rows:
        key = (row.get("test", ""), str(row.get("segment_text", "")).strip())
        if key in bench and str(row.get("code_token_conf", "")).strip():
            deltas.append(abs(float(row["code_token_conf"]) - bench[key]))
    deltas.sort()
    if not deltas:
        raise SystemExit("s2_confidence_run_jitter: no matched segments")
    return {
        "n_matched_segments": len(deltas),
        "median_abs_delta": rounded(statistics.median(deltas), 8),
        "p90_abs_delta": rounded(deltas[int(len(deltas) * 0.90)], 8),
        "max_abs_delta": rounded(deltas[-1], 8),
        "n_delta_ge_1e_4": sum(delta >= 1e-4 for delta in deltas),
        "n_delta_ge_1e_5": sum(delta >= 1e-5 for delta in deltas),
    }


def rank_auc(labels: list[bool], scores: list[float]) -> float | None:
    """AUROC via average ranks, including exact handling of tied scores."""
    if len(labels) != len(scores) or not labels:
        return None
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        for original_index, _score in ordered[start:end]:
            ranks[original_index] = average_rank
        start = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def conf_usable_rows(
    rows: list[dict[str, Any]],
    signal: str,
    pred_node_key: str = "pred_node",
    exclude_unclassified: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Scored rows carrying a value for ``signal``, plus the sentinel count.

    A case whose final prediction is UNCLASSIFIED has no code tokens behind
    its recorded confidence; the stored 0.0 is a sentinel, not a measured
    probability (§4.7.1). Token-confidence analyses therefore exclude such
    rows; agreement analyses do not (agreement is measured regardless).
    """
    usable = [
        row for row in scored_rows(rows)
        if str(row.get(signal, "")).strip() not in {"", "nan", "None"}
    ]
    if not exclude_unclassified:
        return usable, 0
    kept = [row for row in usable if row.get(pred_node_key) != "UNCLASSIFIED"]
    return kept, len(usable) - len(kept)


def uncertainty_metrics(
    rows: list[dict[str, Any]],
    signal: str,
    pred_safe_key: str = "pred_safe",
    pred_node_key: str = "pred_node",
    exclude_unclassified: bool = False,
) -> dict[str, Any]:
    """AUROC of ``signal`` against the correctness of the selected prediction.

    Defaults to the deterministic pipeline prediction: the signal is judged as a
    flag on the label the system would deploy. The consistency conditions are
    additionally scored against their majority-vote verdict.
    """
    usable, excluded = conf_usable_rows(rows, signal, pred_node_key, exclude_unclassified)
    n_scored = len(scored_rows(rows))
    scores = [float(row[signal]) for row in usable]
    safety_labels = [as_bool(row.get("gt_safe")) == as_bool(row.get(pred_safe_key)) for row in usable]
    node_labels = [row.get("gt_node") == row.get(pred_node_key) for row in usable]
    return {
        "signal": signal,
        "n": len(usable),
        "n_scored": n_scored,
        # Scored cases carrying no signal value at all (for token confidence:
        # META_ONLY rows, where no classifier call happened).
        "n_missing_signal": n_scored - len(usable) - excluded,
        "n_excluded_unclassified": excluded,
        "safety_correctness_auroc": rounded(rank_auc(safety_labels, scores)),
        "node_correctness_auroc": rounded(rank_auc(node_labels, scores)),
    }


def threshold_metrics(
    rows: list[dict[str, Any]],
    signal: str,
    cutoff: float,
    pred_safe_key: str = "pred_safe",
    pred_node_key: str = "pred_node",
    exclude_unclassified: bool = False,
) -> dict[str, Any]:
    usable, _excluded = conf_usable_rows(rows, signal, pred_node_key, exclude_unclassified)
    flagged = [row for row in usable if float(row[signal]) < cutoff]
    safety_errors = [row for row in usable if as_bool(row.get("gt_safe")) != as_bool(row.get(pred_safe_key))]
    caught = [row for row in flagged if as_bool(row.get("gt_safe")) != as_bool(row.get(pred_safe_key))]
    node_errors_flagged = [row for row in flagged if row.get("gt_node") != row.get(pred_node_key)]
    # §4.7.1: flags with a correct safety verdict are not all wasted — many
    # carry a wrong primary node.
    correct_safety_node_errors = [
        row for row in node_errors_flagged
        if as_bool(row.get("gt_safe")) == as_bool(row.get(pred_safe_key))
    ]
    return {
        "signal": signal,
        "cutoff_exclusive": cutoff,
        "n_usable": len(usable),
        "n_safety_errors": len(safety_errors),
        "n_flagged": len(flagged),
        "n_safety_errors_caught": len(caught),
        "n_correct_flagged": len(flagged) - len(caught),
        "n_node_errors": sum(row.get("gt_node") != row.get(pred_node_key) for row in usable),
        "n_node_errors_flagged": len(node_errors_flagged),
        "n_correct_safety_flagged_node_errors": len(correct_safety_node_errors),
    }


def stratified_metrics(rows: list[dict[str, Any]], index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    enriched = [(row, index.get(row.get("user_input", ""), {})) for row in rows]
    one_pattern = [row for row, metadata in enriched if metadata.get("test_type") == "clean"]
    multipart = [row for row, metadata in enriched if metadata.get("test_type") == "multipart"]
    return {
        "one_pattern_clean": benchmark_metrics(one_pattern),
        "multipart": benchmark_metrics(multipart),
    }


def segment_nodes(row: dict[str, Any]) -> set[str]:
    raw = row.get("segments_json", "")
    if not raw:
        return set()
    try:
        segments = json.loads(raw)
    except json.JSONDecodeError:
        return set()
    return {segment.get("node_id", "") for segment in segments if isinstance(segment, dict)}


def mal_mal_metrics(rows: list[dict[str, Any]], index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    primary = sum(row.get("gt_node") == row.get("pred_node") for row in rows)
    secondary = 0
    any_maladaptive = 0
    for row in rows:
        nodes = segment_nodes(row)
        expected_secondary = index.get(row.get("user_input", ""), {}).get("expected_secondary_node")
        secondary += expected_secondary in nodes
        any_maladaptive += bool(nodes & MALADAPTIVE_NODES)
    return {
        "n": total,
        "primary_node_correct": primary,
        "primary_node_accuracy": rounded(ratio(primary, total), 6),
        "secondary_node_caught": secondary,
        "secondary_node_recall": rounded(ratio(secondary, total), 6),
        "any_maladaptive_caught": any_maladaptive,
        "any_maladaptive_recall": rounded(ratio(any_maladaptive, total), 6),
    }


def load_kl_spotcheck(path: Path) -> list[dict[str, Any]]:
    """Reload KL's preserved merged artifact and rederive all calculated fields."""
    calculated = {
        "reference_is_maladaptive",
        "classifier_is_maladaptive",
        "classifier_raw_unsafe",
        "raw_expert_unsafe",
        "expert_node_is_maladaptive",
        "raw_safety_agreement",
        "ontology_safety_agreement",
        "primary_node_agreement",
        "classifier_reference_safety_agreement",
        "classifier_reference_node_agreement",
        "classifier_expert_raw_safety_agreement",
        "classifier_expert_ontology_safety_agreement",
        "classifier_expert_node_agreement",
        "three_way_node_pattern",
        "three_way_safety_pattern",
        "adequacy_penalty",
        "disagreement_type",
    }
    return [derive_spotcheck_fields({key: value for key, value in row.items() if key not in calculated}) for row in read_csv(path)]


def compact_spotcheck_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = spotcheck_metrics(rows)
    n = metrics["n_complete"]
    return {
        **metrics,
        "raw_safety_agreement_rate": rounded(ratio(metrics["raw_safety_agree"], n), 6),
        "node_derived_safety_agreement_rate": rounded(ratio(metrics["ontology_safety_agree"], n), 6),
        "strict_node_agreement_rate": rounded(ratio(metrics["primary_node_agree"], n), 6),
        "adequacy_penalty_rate": rounded(ratio(metrics["adequacy_penalties"], n), 6),
        # Classifier-vs-expert rates. The boundary cases have no fixed
        # reference label, so the boundary table in §4.6.2 compares the
        # classifier against each expert; the expert-vs-reference rates above
        # are only meaningful on the scored slice.
        "classifier_expert_raw_safety_rate": rounded(
            ratio(metrics["classifier_expert_raw_safety_agree"], n), 6
        ),
        "classifier_expert_ontology_safety_rate": rounded(
            ratio(metrics["classifier_expert_ontology_safety_agree"], n), 6
        ),
        "classifier_expert_node_rate": rounded(
            ratio(metrics["classifier_expert_node_agree"], n), 6
        ),
    }


def spotcheck_slices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all": compact_spotcheck_metrics(rows),
        "primary_non_boundary": compact_spotcheck_metrics(row_slice(rows, include_boundary=False)),
        "boundary": compact_spotcheck_metrics(row_slice(rows, include_boundary=True)),
    }


def pairwise_expert_metrics(
    ag_rows: list[dict[str, Any]],
    kl_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ag = {row["case_id"]: row for row in completed_spotcheck_rows(ag_rows)}
    kl = {row["case_id"]: row for row in completed_spotcheck_rows(kl_rows)}
    overlap_ids = sorted(set(ag) & set(kl), key=lambda case_id: int(ag[case_id].get("order", 0)))
    audit_rows: list[dict[str, Any]] = []
    for case_id in overlap_ids:
        a, k = ag[case_id], kl[case_id]
        ag_unsafe = a["raw_expert_unsafe"]
        kl_unsafe = k["raw_expert_unsafe"]
        ag_derived = a["expert_node_is_maladaptive"]
        kl_derived = k["expert_node_is_maladaptive"]
        audit_rows.append({
            "case_id": case_id,
            "order": a.get("order", ""),
            "source_bucket": a.get("source_bucket", ""),
            "reference_node": a.get("reference_node", ""),
            "ag_safe": a.get("expert_safe", ""),
            "kl_safe": k.get("expert_safe", ""),
            "raw_safety_agreement": str(ag_unsafe == kl_unsafe).lower(),
            "ag_node": a.get("expert_node", ""),
            "kl_node": k.get("expert_node", ""),
            "node_derived_safety_agreement": str(ag_derived == kl_derived).lower(),
            "strict_node_agreement": str(a.get("expert_node") == k.get("expert_node")).lower(),
        })
    n = len(audit_rows)
    raw_agree = sum(row["raw_safety_agreement"] == "true" for row in audit_rows)
    derived_agree = sum(row["node_derived_safety_agreement"] == "true" for row in audit_rows)
    node_agree = sum(row["strict_node_agreement"] == "true" for row in audit_rows)
    raw_pairs = [(ag[case_id]["raw_expert_unsafe"], kl[case_id]["raw_expert_unsafe"]) for case_id in overlap_ids]
    derived_pairs = [
        (ag[case_id]["expert_node_is_maladaptive"], kl[case_id]["expert_node_is_maladaptive"])
        for case_id in overlap_ids
    ]
    node_pairs = [(ag[case_id]["expert_node"], kl[case_id]["expert_node"]) for case_id in overlap_ids]
    return ({
        "n_shared_complete": n,
        "raw_safety_agree": raw_agree,
        "raw_safety_agreement_rate": rounded(ratio(raw_agree, n), 6),
        "raw_safety_kappa": rounded(cohen_kappa(raw_pairs)),
        "node_derived_safety_agree": derived_agree,
        "node_derived_safety_agreement_rate": rounded(ratio(derived_agree, n), 6),
        "node_derived_safety_kappa": rounded(cohen_kappa(derived_pairs)),
        "strict_node_agree": node_agree,
        "strict_node_agreement_rate": rounded(ratio(node_agree, n), 6),
        "strict_node_kappa": rounded(cohen_kappa(node_pairs)),
    }, audit_rows)


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "") or "n/a") for row in rows).items()))


def system_eval_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("expert_clinically_acceptable")]
    paired = build_paired_system_rows(rows)
    by_condition: dict[str, Any] = {}
    for condition in ("control", "treatment"):
        group = [row for row in complete if row.get("condition") == condition]
        scores = [int(row["clinical_acceptability_score"]) for row in group if str(row.get("clinical_acceptability_score", ""))]
        by_condition[condition] = {
            "n": len(group),
            "clinical_acceptability": distribution(group, "expert_clinically_acceptable"),
            "mean_clinical_acceptability_score": rounded(sum(scores) / len(scores) if scores else None),
            "redirect_quality": distribution(group, "expert_redirect_quality"),
            "maladaptive_endorsement": distribution(group, "expert_maladaptive_endorsement"),
        }
    return {"n_rows": len(rows), "n_complete": len(complete), "n_pairs": len(paired), "conditions": by_condition}


# §4.8.2 inter-expert agreement on the shared blinded response rows. Exact
# agreement is not comparable across the four rating dimensions because the
# category marginals differ (both raters marked most rows acceptable and
# selected "none" endorsement most of the time), so Cohen's kappa is reported
# alongside the raw counts, mirroring the spot-check inter-rater table.
SYSTEM_EVAL_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("clinical_acceptability", "expert_clinically_acceptable"),
    ("redirect_quality", "expert_redirect_quality"),
    ("maladaptive_endorsement", "expert_maladaptive_endorsement"),
    ("distress_validation", "expert_validates_distress_appropriately"),
)


# Ordinal scales and sign conventions of the within-case comparison
# (tab:expert-system-paired-deltas): mean change is treatment minus control,
# sign-reversed for maladaptive endorsement so positive always favors
# safety augmentation.
SYSTEM_EVAL_SCALES: dict[str, tuple[dict[str, int], int]] = {
    "clinical_acceptability": ({"no": 0, "uncertain": 1, "yes": 2}, 1),
    "redirect_quality": ({"absent": 0, "partial": 1, "strong": 2}, 1),
    "maladaptive_endorsement": ({"none": 0, "minor": 1, "clear": 2}, -1),
    "distress_validation": ({"no": 0, "partial": 1, "yes": 2}, 1),
}


# Expert-ratings figure (§4.8): fixed level order, worst -> best, matching
# the diverging layout (worse pole left, better pole right). The middle
# level's valence differs by dimension — that mapping is presentation
# semantics and lives in figures/generate_expert_ratings.py, not here.
EXPERT_RATING_LEVELS: dict[str, tuple[str, str, str]] = {
    "clinical_acceptability": ("no", "uncertain", "yes"),
    "redirect_quality": ("absent", "partial", "strong"),
    "distress_validation": ("no", "partial", "yes"),
    "maladaptive_endorsement": ("clear", "minor", "none"),
}


def expert_rating_distributions(
    kl_rows: list[dict[str, Any]], ag_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Per-rater, per-condition rating distributions for the §4.8 expert-
    ratings figure. Level counts mirror tab:expert-system-condition-summary
    (and add the distress-validation distribution that table omits for
    width); n/a is kept separate so level counts + n/a always partition a
    condition's completed rows."""
    result: dict[str, Any] = {}
    for rater, rows in (("kl", kl_rows), ("ag", ag_rows)):
        complete = [row for row in rows if str(row.get("expert_clinically_acceptable", "")).strip()]
        rater_block: dict[str, Any] = {}
        for condition in ("control", "treatment"):
            group = [row for row in complete if row.get("condition") == condition]
            cond_block: dict[str, Any] = {"n": len(group)}
            for dimension, column in SYSTEM_EVAL_DIMENSIONS:
                levels = EXPERT_RATING_LEVELS[dimension]
                values = [str(row.get(column, "")).strip() or "n/a" for row in group]
                unknown = sorted(set(values) - set(levels) - {"n/a"})
                if unknown:
                    raise SystemExit(
                        f"expert rating distributions: unknown {dimension} level(s) {unknown}"
                    )
                counts = Counter(values)
                cond_block[dimension] = {
                    "levels": {level: counts.get(level, 0) for level in levels},
                    "n_na": counts.get("n/a", 0),
                }
            rater_block[condition] = cond_block
        result[rater] = rater_block
    return result


def system_eval_shared_paired_deltas(
    kl_rows: list[dict[str, Any]], ag_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Within-case mean changes for both raters on the cases BOTH completed
    as pairs. §4.8.2 uses this to state that the opposite movement of
    distress validation and maladaptive endorsement between the two experts
    is not an artifact of their different material."""

    def complete_pairs(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
        by_case: dict[str, dict[str, dict[str, Any]]] = {}
        for row in rows:
            if str(row.get("expert_clinically_acceptable", "")).strip():
                by_case.setdefault(row["case_id"], {})[row["condition"]] = row
        return {case: v for case, v in by_case.items() if set(v) == {"control", "treatment"}}

    kl_pairs = complete_pairs(kl_rows)
    ag_pairs = complete_pairs(ag_rows)
    shared_cases = sorted(set(kl_pairs) & set(ag_pairs))
    result: dict[str, Any] = {"n_shared_pairs": len(shared_cases), "dimensions": {}}
    for dimension, column in SYSTEM_EVAL_DIMENSIONS:
        scale, sign = SYSTEM_EVAL_SCALES[dimension]
        dim_result: dict[str, Any] = {}
        for rater, pairs in (("kl", kl_pairs), ("ag", ag_pairs)):
            deltas = []
            for case in shared_cases:
                control = str(pairs[case]["control"].get(column, "")).strip()
                treatment = str(pairs[case]["treatment"].get(column, "")).strip()
                if control in scale and treatment in scale:
                    deltas.append(scale[treatment] - scale[control])
            dim_result[rater] = {
                "n_valid_pairs": len(deltas),
                "mean_change": rounded(sign * sum(deltas) / len(deltas)) if deltas else None,
            }
        result["dimensions"][dimension] = dim_result
    return result


def system_eval_interexpert(
    kl_rows: list[dict[str, Any]], ag_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    def complete_by_key(rows: list[dict[str, Any]], source: str) -> dict[tuple[str, str], dict[str, Any]]:
        indexed: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            if not str(row.get("expert_clinically_acceptable", "")).strip():
                continue
            key = (row["case_id"], row["condition"])
            if key in indexed:
                raise SystemExit(f"{source}: duplicate system-eval row for {key}")
            indexed[key] = row
        return indexed

    kl = complete_by_key(kl_rows, "kl_system_merged")
    ag = complete_by_key(ag_rows, "ag_system_merged")
    shared = sorted(set(kl) & set(ag))
    result: dict[str, Any] = {"n_shared_complete": len(shared), "dimensions": {}}
    for dimension, column in SYSTEM_EVAL_DIMENSIONS:
        pairs = [
            (str(kl[key].get(column, "")).strip(), str(ag[key].get(column, "")).strip())
            for key in shared
        ]
        if any(not a or not b for a, b in pairs):
            raise SystemExit(f"system-eval inter-expert: empty {dimension} rating on a shared row")
        total = len(pairs)
        agree = sum(a == b for a, b in pairs)
        kl_counts = Counter(a for a, _b in pairs)
        ag_counts = Counter(b for _a, b in pairs)
        chance = sum(
            (kl_counts[label] / total) * (ag_counts[label] / total)
            for label in set(kl_counts) | set(ag_counts)
        )
        result["dimensions"][dimension] = {
            "n": total,
            "exact_agree": agree,
            "agreement_rate": rounded(ratio(agree, total)),
            "chance_agreement": rounded(chance),
            "kappa": rounded(cohen_kappa(pairs)),
        }
    return result


def integration_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    included = [row for row in rows if as_bool(row.get("critic_classified_unsafe")) is True]
    result: dict[str, Any] = {"n_scenarios": len({row["case_id"] for row in included})}
    for tier in ("main", "multipart"):
        tier_result: dict[str, Any] = {}
        for condition in ("control", "treatment"):
            group = [row for row in included if row.get("tier") == tier and row.get("condition") == condition]
            tier_result[condition] = {
                "n": len(group),
                "redirect_quality": distribution(group, "judge_redirect_quality"),
            }
        result[tier] = tier_result
    return result


def source_manifest(paths: dict[str, Path]) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for name, path in sorted(paths.items()):
        entry: dict[str, Any] = {"path": str(path.relative_to(REPO_ROOT))}
        if path.exists():
            entry["sha256"] = sha256(path)
            entry["bytes"] = path.stat().st_size
        else:
            entry["withheld"] = True
        manifest[name] = entry
    return manifest


def pct_text(rate: float | None) -> str:
    return "n/a" if rate is None else f"{100 * rate:.1f}%"


def count_rate(metrics: dict[str, Any], count_key: str, rate_key: str, denominator_key: str = "n_complete") -> str:
    return f"{metrics[count_key]}/{metrics[denominator_key]} ({pct_text(metrics[rate_key])})"


def render_markdown(stats: dict[str, Any]) -> str:
    lines = [
        "# Chapter 4 Reproducible Statistics",
        "",
        "Generated by `uv run python experiments/evaluation_statistics/recompute_chapter4.py`.",
        "All rates in `chapter4_statistics.json` are proportions from 0 to 1; this report formats them as percentages.",
        "",
        "## Main Development Benchmark",
        "",
        "| Condition | Scored | Safety accuracy | Unsafe recall | Node accuracy | False alarms | Missed unsafe |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in stats["main_benchmark"].items():
        lines.append(
            f"| {name} | {metrics['n_scored']} | {metrics['safety_correct']}/{metrics['n_scored']} ({pct_text(metrics['safety_accuracy'])}) "
            f"| {metrics['unsafe_true_positive']}/{metrics['n_unsafe']} ({pct_text(metrics['unsafe_recall'])}) "
            f"| {metrics['node_correct']}/{metrics['n_scored']} ({pct_text(metrics['node_accuracy'])}) "
            f"| {metrics['unsafe_false_positive']} | {metrics['unsafe_false_negative']} |"
        )
    lines.extend([
        "",
        "## Pipeline Ablations",
        "",
        "| Condition | Safety accuracy | Unsafe recall | Node accuracy | False alarms | Missed unsafe |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, metrics in stats["ablations"].items():
        lines.append(
            f"| {name} | {metrics['safety_correct']}/{metrics['n_scored']} ({pct_text(metrics['safety_accuracy'])}) "
            f"| {metrics['unsafe_true_positive']}/{metrics['n_unsafe']} ({pct_text(metrics['unsafe_recall'])}) "
            f"| {metrics['node_correct']}/{metrics['n_scored']} ({pct_text(metrics['node_accuracy'])}) "
            f"| {metrics['unsafe_false_positive']} | {metrics['unsafe_false_negative']} |"
        )
    lines.extend([
        "",
        "## Segmentation Strata",
        "",
        "| Condition | Slice | N | Safety accuracy | Unsafe recall | Node accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for condition, slices in stats["segmentation_strata"].items():
        for slice_name, metrics in slices.items():
            lines.append(
                f"| {condition} | {slice_name} | {metrics['n_scored']} "
                f"| {metrics['safety_correct']}/{metrics['n_scored']} ({pct_text(metrics['safety_accuracy'])}) "
                f"| {metrics['unsafe_true_positive']}/{metrics['n_unsafe']} ({pct_text(metrics['unsafe_recall'])}) "
                f"| {metrics['node_correct']}/{metrics['n_scored']} ({pct_text(metrics['node_accuracy'])}) |"
            )
    lines.extend([
        "",
        "## Composite Multi-Maladaptive Stress Slice",
        "",
        "| Condition | Primary node | Secondary node | Any maladaptive |",
        "|---|---:|---:|---:|",
    ])
    for name, metrics in stats["multi_maladaptive"].items():
        lines.append(
            f"| {name} | {metrics['primary_node_correct']}/{metrics['n']} ({pct_text(metrics['primary_node_accuracy'])}) "
            f"| {metrics['secondary_node_caught']}/{metrics['n']} ({pct_text(metrics['secondary_node_recall'])}) "
            f"| {metrics['any_maladaptive_caught']}/{metrics['n']} ({pct_text(metrics['any_maladaptive_recall'])}) |"
        )
    heldout = stats["heldout_test_postfix"]
    lines.extend([
        "",
        "## Held-Out Test (Post-Fix)",
        "",
        f"- Scored cases: {heldout['n_scored']}",
        f"- Safety accuracy: {heldout['safety_correct']}/{heldout['n_scored']} ({pct_text(heldout['safety_accuracy'])})",
        f"- Unsafe recall: {heldout['unsafe_true_positive']}/{heldout['n_unsafe']} ({pct_text(heldout['unsafe_recall'])})",
        f"- Strict node accuracy: {heldout['node_correct']}/{heldout['n_scored']} ({pct_text(heldout['node_accuracy'])})",
        f"- Affect/emotional collapsed node accuracy: {heldout['affect_emotional_collapsed_node_correct']}/{heldout['n_scored']} ({pct_text(heldout['affect_emotional_collapsed_node_accuracy'])})",
        "",
        "## Uncertainty Quantification",
        "",
        "| Condition | Signal | N | Safety AUROC | Node AUROC |",
        "|---|---|---:|---:|---:|",
    ])
    for condition in (
        "s1_baseline",
        "s1_consistency",
        "s2_baseline",
        "s2_consistency",
        "s2_baseline_corrected",
    ):
        for signal_name, metrics in stats["uncertainty"][condition].items():
            lines.append(
                f"| {condition} | {signal_name} | {metrics['n']} | {metrics['safety_correctness_auroc']:.3f} "
                f"| {metrics['node_correctness_auroc']:.3f} |"
            )
    lines.extend([
        "",
        "## Expert Spot Check",
        "",
        "| Rater / comparison | Complete | Raw safety | Node-derived safety | Strict node |",
        "|---|---:|---:|---:|---:|",
    ])
    for rater in ("kl", "ag"):
        metrics = stats["expert_spotcheck"][rater]["all"]
        lines.append(
            f"| {rater.upper()} vs reference | {metrics['n_complete']} "
            f"| {count_rate(metrics, 'raw_safety_agree', 'raw_safety_agreement_rate')} "
            f"| {count_rate(metrics, 'ontology_safety_agree', 'node_derived_safety_agreement_rate')} "
            f"| {count_rate(metrics, 'primary_node_agree', 'strict_node_agreement_rate')} |"
        )
    pair = stats["expert_spotcheck"]["ag_kl_pairwise"]
    lines.append(
        f"| AG vs KL | {pair['n_shared_complete']} | {pair['raw_safety_agree']}/{pair['n_shared_complete']} ({pct_text(pair['raw_safety_agreement_rate'])}) "
        f"| {pair['node_derived_safety_agree']}/{pair['n_shared_complete']} ({pct_text(pair['node_derived_safety_agreement_rate'])}) "
        f"| {pair['strict_node_agree']}/{pair['n_shared_complete']} ({pct_text(pair['strict_node_agreement_rate'])}) |"
    )
    lines.extend([
        "",
        "Pairwise Cohen's kappa (AG vs KL): "
        f"raw safety {pair['raw_safety_kappa']:.3f}, node-derived safety {pair['node_derived_safety_kappa']:.3f}, "
        f"strict node {pair['strict_node_kappa']:.3f}.",
        "",
        "## System-Level Expert Evaluation",
        "",
        "| Rater | Completed rows | Matched pairs | Control mean clinical score | Treatment mean clinical score |",
        "|---|---:|---:|---:|---:|",
    ])
    for rater in ("kl", "ag"):
        metrics = stats["system_expert_evaluation"][rater]
        control = metrics["conditions"]["control"]["mean_clinical_acceptability_score"]
        treatment = metrics["conditions"]["treatment"]["mean_clinical_acceptability_score"]
        lines.append(f"| {rater.upper()} | {metrics['n_complete']} | {metrics['n_pairs']} | {control:.3f} | {treatment:.3f} |")
    sys_pair = stats["system_expert_evaluation"]["interexpert_shared"]
    lines.extend([
        "",
        f"Inter-expert agreement on the {sys_pair['n_shared_complete']} shared blinded response rows "
        "(exact agreement is inflated by skewed category use, so Cohen's kappa is reported alongside it):",
        "",
        "| Rating dimension | Exact agreement | Chance agreement | Cohen's kappa |",
        "|---|---:|---:|---:|",
    ])
    for dimension, _column in SYSTEM_EVAL_DIMENSIONS:
        metrics = sys_pair["dimensions"][dimension]
        lines.append(
            f"| {dimension.replace('_', ' ')} | {metrics['exact_agree']}/{metrics['n']} ({pct_text(metrics['agreement_rate'])}) "
            f"| {metrics['chance_agreement']:.3f} | {metrics['kappa']:.3f} |"
        )
    shared_deltas = stats["system_expert_evaluation"]["shared_paired_deltas"]
    lines.extend([
        "",
        f"Within-case mean changes on the {shared_deltas['n_shared_pairs']} cases both experts "
        "completed as pairs (sign convention as the within-case table: positive favors safety augmentation):",
        "",
        "| Rating dimension | KL valid pairs | KL mean change | AG valid pairs | AG mean change |",
        "|---|---:|---:|---:|---:|",
    ])
    for dimension, _column in SYSTEM_EVAL_DIMENSIONS:
        metrics = shared_deltas["dimensions"][dimension]
        lines.append(
            f"| {dimension.replace('_', ' ')} | {metrics['kl']['n_valid_pairs']} | {metrics['kl']['mean_change']:+.3f} "
            f"| {metrics['ag']['n_valid_pairs']} | {metrics['ag']['mean_change']:+.3f} |"
        )
    rating_dist = stats["system_expert_evaluation"]["rating_distributions"]
    lines.extend([
        "",
        "Rating distributions by condition (expert-ratings figure; level order worst to best, "
        "n/a excluded from the level counts):",
        "",
        "| Rater | Condition | n | Rating dimension | Worst | Middle | Best | n/a |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ])
    for rater in ("kl", "ag"):
        for condition in ("control", "treatment"):
            block = rating_dist[rater][condition]
            for dimension, _column in SYSTEM_EVAL_DIMENSIONS:
                levels = block[dimension]["levels"]
                cells = " | ".join(
                    f"{level} {count}" for level, count in levels.items()
                )
                lines.append(
                    f"| {rater.upper()} | {condition} | {block['n']} | {dimension.replace('_', ' ')} "
                    f"| {cells} | {block[dimension]['n_na']} |"
                )
    pilot = stats["internal_constraint_redirect_pilot"]
    lines.extend([
        "",
        "## Internal Constraint-Redirect Pilot",
        "",
        f"Scenarios: {pilot['n_scenarios']}",
        "",
        "| Tier | Condition | N | Redirect-quality distribution |",
        "|---|---|---:|---|",
    ])
    for tier in ("main", "multipart"):
        for condition in ("control", "treatment"):
            metrics = pilot[tier][condition]
            formatted = ", ".join(f"{key}={value}" for key, value in metrics["redirect_quality"].items())
            lines.append(f"| {tier} | {condition} | {metrics['n']} | {formatted} |")
    lines.extend([
        "",
        "## Traceability",
        "",
        "Every input path and SHA-256 digest is recorded in `provenance.json`. Detailed benchmark, ablation, "
        "uncertainty, held-out-test, expert, and system-evaluation values are in `chapter4_statistics.json`.",
        "",
    ])
    return "\n".join(lines)


def tex_pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{100 * rate:.1f}\\%"


def tex_pct_two(rate: float | None) -> str:
    # The classification-results prose reports two decimals (97.86%, 99.44%).
    return "n/a" if rate is None else f"{100 * rate:.2f}\\%"


def render_tex(stats: dict[str, Any]) -> str:
    ag = stats["expert_spotcheck"]["ag"]
    kl = stats["expert_spotcheck"]["kl"]
    pair = stats["expert_spotcheck"]["ag_kl_pairwise"]
    comp = stats["dataset_composition"]
    queue = comp["spotcheck_queue_test_types"]
    values = {
        "DevNodeCasesMin": comp["dev"]["node_case_min"],
        "DevNodeCasesMax": comp["dev"]["node_case_max"],
        "DevAdaptiveCount": comp["dev"]["safety_groups"]["adaptive"],
        "DevNeutralCount": comp["dev"]["safety_groups"]["neutral"],
        "DevMaladaptiveCount": comp["dev"]["safety_groups"]["maladaptive"],
        "TestNodeCasesMin": comp["test_final"]["node_case_min"],
        "TestNodeCasesMax": comp["test_final"]["node_case_max"],
        "TestSafeAdaptiveCount": comp["test_final"]["safety_groups"]["adaptive"],
        "TestSafeNeutralCount": comp["test_final"]["safety_groups"]["neutral"],
        "SpotQueueClean": queue.get("clean", 0),
        "SpotQueueMultipart": queue.get("multipart", 0),
        "SpotQueueBoundary": queue.get("boundary", 0),
        "KLSpotComplete": kl["all"]["n_complete"],
        "KLSpotPrimaryComplete": kl["primary_non_boundary"]["n_complete"],
        "KLSpotBoundaryComplete": kl["boundary"]["n_complete"],
        # KL scored-slice agreement against the reference labels, mirroring
        # the \AGSpotPrimary* family below.
        "KLSpotPrimaryRawCount": kl["primary_non_boundary"]["raw_safety_agree"],
        "KLSpotPrimaryRawPct": tex_pct(kl["primary_non_boundary"]["raw_safety_agreement_rate"]),
        "KLSpotPrimaryDerivedCount": kl["primary_non_boundary"]["ontology_safety_agree"],
        "KLSpotPrimaryDerivedPct": tex_pct(kl["primary_non_boundary"]["node_derived_safety_agreement_rate"]),
        "KLSpotPrimaryNodeCount": kl["primary_non_boundary"]["primary_node_agree"],
        "KLSpotPrimaryNodePct": tex_pct(kl["primary_non_boundary"]["strict_node_agreement_rate"]),
        "KLSpotPrimaryAdequacyCount": kl["primary_non_boundary"]["adequacy_penalties"],
        "KLSpotPrimaryAdequacyPct": tex_pct(kl["primary_non_boundary"]["adequacy_penalty_rate"]),
        # §4.6.2 boundary table. Boundary cases have no fixed reference label,
        # so these compare the CLASSIFIER against each expert. The
        # expert-vs-reference fields must not be used on this slice: for KL
        # they give 12/17, 6/17 and 2/17 against the correct 11/17, 7/17 and
        # 3/17. They coincide for AG only by chance at n=7.
        "KLSpotBoundaryClsRawCount": kl["boundary"]["classifier_expert_raw_safety_agree"],
        "KLSpotBoundaryClsRawPct": tex_pct(kl["boundary"]["classifier_expert_raw_safety_rate"]),
        "KLSpotBoundaryClsDerivedCount": kl["boundary"]["classifier_expert_ontology_safety_agree"],
        "KLSpotBoundaryClsDerivedPct": tex_pct(kl["boundary"]["classifier_expert_ontology_safety_rate"]),
        "KLSpotBoundaryClsNodeCount": kl["boundary"]["classifier_expert_node_agree"],
        "KLSpotBoundaryClsNodePct": tex_pct(kl["boundary"]["classifier_expert_node_rate"]),
        "AGSpotBoundaryClsRawCount": ag["boundary"]["classifier_expert_raw_safety_agree"],
        "AGSpotBoundaryClsRawPct": tex_pct(ag["boundary"]["classifier_expert_raw_safety_rate"]),
        "AGSpotBoundaryClsDerivedCount": ag["boundary"]["classifier_expert_ontology_safety_agree"],
        "AGSpotBoundaryClsDerivedPct": tex_pct(ag["boundary"]["classifier_expert_ontology_safety_rate"]),
        "AGSpotBoundaryClsNodeCount": ag["boundary"]["classifier_expert_node_agree"],
        "AGSpotBoundaryClsNodePct": tex_pct(ag["boundary"]["classifier_expert_node_rate"]),
        "AGSpotComplete": ag["all"]["n_complete"],
        "AGSpotPrimaryComplete": ag["primary_non_boundary"]["n_complete"],
        "AGSpotBoundaryComplete": ag["boundary"]["n_complete"],
        "AGSpotRawCount": ag["all"]["raw_safety_agree"],
        "AGSpotRawPct": tex_pct(ag["all"]["raw_safety_agreement_rate"]),
        "AGSpotDerivedCount": ag["all"]["ontology_safety_agree"],
        "AGSpotDerivedPct": tex_pct(ag["all"]["node_derived_safety_agreement_rate"]),
        "AGSpotNodeCount": ag["all"]["primary_node_agree"],
        "AGSpotNodePct": tex_pct(ag["all"]["strict_node_agreement_rate"]),
        "AGSpotAdequacyCount": ag["all"]["adequacy_penalties"],
        "AGSpotAdequacyPct": tex_pct(ag["all"]["adequacy_penalty_rate"]),
        "AGSpotPrimaryRawCount": ag["primary_non_boundary"]["raw_safety_agree"],
        "AGSpotPrimaryRawPct": tex_pct(ag["primary_non_boundary"]["raw_safety_agreement_rate"]),
        "AGSpotPrimaryDerivedCount": ag["primary_non_boundary"]["ontology_safety_agree"],
        "AGSpotPrimaryDerivedPct": tex_pct(ag["primary_non_boundary"]["node_derived_safety_agreement_rate"]),
        "AGSpotPrimaryNodeCount": ag["primary_non_boundary"]["primary_node_agree"],
        "AGSpotPrimaryNodePct": tex_pct(ag["primary_non_boundary"]["strict_node_agreement_rate"]),
        "AGSpotPrimaryAdequacyCount": ag["primary_non_boundary"]["adequacy_penalties"],
        "AGSpotPrimaryAdequacyPct": tex_pct(ag["primary_non_boundary"]["adequacy_penalty_rate"]),
        "AGSpotBoundaryRawCount": ag["boundary"]["raw_safety_agree"],
        "AGSpotBoundaryRawPct": tex_pct(ag["boundary"]["raw_safety_agreement_rate"]),
        "AGSpotBoundaryDerivedCount": ag["boundary"]["ontology_safety_agree"],
        "AGSpotBoundaryDerivedPct": tex_pct(ag["boundary"]["node_derived_safety_agreement_rate"]),
        "AGSpotBoundaryNodeCount": ag["boundary"]["primary_node_agree"],
        "AGSpotBoundaryNodePct": tex_pct(ag["boundary"]["strict_node_agreement_rate"]),
        "AGSpotBoundaryAdequacyCount": ag["boundary"]["adequacy_penalties"],
        "AGSpotBoundaryAdequacyPct": tex_pct(ag["boundary"]["adequacy_penalty_rate"]),
        "AGClassifierRefSafetyCount": ag["all"]["classifier_reference_safety_agree"],
        "AGClassifierRefNodeCount": ag["all"]["classifier_reference_node_agree"],
        "AGClassifierRawCount": ag["all"]["classifier_expert_raw_safety_agree"],
        "AGClassifierDerivedCount": ag["all"]["classifier_expert_ontology_safety_agree"],
        "AGClassifierNodeCount": ag["all"]["classifier_expert_node_agree"],
        # The pairwise block covers ALL shared complete rows, including the
        # seven boundary rows, so the "WithBoundary" macros are NOT the
        # chapter's inter-expert table (28 scored non-boundary rows, computed
        # by chapter_4/section_4_3_reference_label_validation.py).
        "AGKLSharedWithBoundary": pair["n_shared_complete"],
        "AGKLRawCountWithBoundary": pair["raw_safety_agree"],
        "AGKLRawPctWithBoundary": tex_pct(pair["raw_safety_agreement_rate"]),
        "AGKLDerivedCountWithBoundary": pair["node_derived_safety_agree"],
        "AGKLDerivedPctWithBoundary": tex_pct(pair["node_derived_safety_agreement_rate"]),
        "AGKLNodeCountWithBoundary": pair["strict_node_agree"],
        "AGKLNodePctWithBoundary": tex_pct(pair["strict_node_agreement_rate"]),
        "AGKLRawKappaWithBoundary": f"{pair['raw_safety_kappa']:.3f}",
        "AGKLDerivedKappaWithBoundary": f"{pair['node_derived_safety_kappa']:.3f}",
        "AGKLNodeKappaWithBoundary": f"{pair['strict_node_kappa']:.3f}",
    }

    # §4.4 classification-results prose (rewrite): the four main dev
    # conditions, the held-out run, and the temperature-0 rerun divergence.
    mb = stats["main_benchmark"]
    values["DevScoredCount"] = mb["s1_baseline"]["n_scored"]
    values["DevUnsafeCount"] = mb["s1_baseline"]["n_unsafe"]
    values["DevSafeCount"] = mb["s1_baseline"]["n_safe"]
    for prefix, key in (
        ("SOneBase", "s1_baseline"),
        ("SOneCons", "s1_consistency"),
        ("STwoBase", "s2_baseline"),
        ("STwoCons", "s2_consistency"),
    ):
        cond = mb[key]
        values[f"{prefix}SafetyAccPct"] = tex_pct_two(cond["safety_accuracy"])
        values[f"{prefix}SafetyCorrect"] = cond["safety_correct"]
        values[f"{prefix}RecallPct"] = tex_pct_two(cond["unsafe_recall"])
        values[f"{prefix}FalseAlarms"] = cond["unsafe_false_positive"]
        values[f"{prefix}Misses"] = cond["unsafe_false_negative"]
        values[f"{prefix}NodeAccPct"] = tex_pct_two(cond["node_accuracy"])
        values[f"{prefix}NodeCorrect"] = cond["node_correct"]
    heldout = stats["heldout_test_postfix"]
    values["HeldoutScoredCount"] = heldout["n_scored"]
    values["HeldoutUnsafeCount"] = heldout["n_unsafe"]
    values["HeldoutSafetyAccPct"] = tex_pct_two(heldout["safety_accuracy"])
    values["HeldoutSafetyCorrect"] = heldout["safety_correct"]
    values["HeldoutRecallPct"] = tex_pct_two(heldout["unsafe_recall"])
    values["HeldoutFalseAlarms"] = heldout["unsafe_false_positive"]
    values["HeldoutMisses"] = heldout["unsafe_false_negative"]
    values["HeldoutNodeAccPct"] = tex_pct_two(heldout["node_accuracy"])
    values["HeldoutNodeCorrect"] = heldout["node_correct"]
    values["HeldoutSafeCount"] = heldout["n_safe"]
    # §4.4.6 confusion-family decomposition of the strict node errors.
    conf_fam = stats["heldout_confusion_families"]
    values["HeldoutNodeErrors"] = conf_fam["node_errors"]
    values["HeldoutNodeErrorsNonMal"] = conf_fam["non_maladaptive"]
    values["HeldoutNodeErrorsWithinMal"] = conf_fam["within_maladaptive"]
    values["HeldoutNodeErrorsCrossFamily"] = conf_fam["cross_family"]
    # §4.6.1 error structure of the dev System 1 baseline. Same suffix family
    # as \Heldout*, so the two decompositions read alike in the prose.
    s1_fam = stats["s1_baseline_confusion_families"]
    values["SOneBaseNodeErrors"] = s1_fam["node_errors"]
    values["SOneBaseNodeErrorsNonMal"] = s1_fam["non_maladaptive"]
    values["SOneBaseNodeErrorsWithinMal"] = s1_fam["within_maladaptive"]
    values["SOneBaseNodeErrorsCrossFamily"] = s1_fam["cross_family"]
    # §4.6.1 states as a guarantee that only verdict-boundary-crossing node
    # errors can be safety errors. Check it on every reported condition, not
    # just the one the section works from.
    for name, part in stats["verdict_partition"].items():
        if not part["matches_safety_errors"]:
            raise SystemExit(
                f"{name}: the node errors crossing the verdict boundary are no "
                "longer exactly the safety errors -- §4.6.1 states this as a "
                "consequence of the triage rule"
            )
    # §4.6.1 also states that affect expression -> emotional mastery is the
    # most frequent confusion in every reported condition, including held-out.
    for name, lead in stats["leading_confusions"].items():
        if lead["pair"] != "AFFECT_EXPRESSION -> EMOTIONAL_MASTERY" or lead["margin"] < 1:
            raise SystemExit(
                f"{name}: affect expression -> emotional mastery is no longer "
                "the strictly most frequent confusion -- §4.6.1 states it "
                "leads in every condition"
            )
    s1_part = stats["verdict_partition"]["s1_baseline"]
    values["SOneBaseErrorsSafeSide"] = s1_part["errors_within_safe_side"]
    values["SOneBaseErrorsUnsafeSide"] = s1_part["errors_within_unsafe_side"]
    values["SOneBaseErrorsCrossing"] = s1_part["errors_crossing"]
    direction = stats["s1_baseline_directional_confusions"]
    values["SOneBaseLargePairMinCount"] = direction["min_count"]
    values["SOneBaseLargePairCount"] = direction["n_large"]
    values["SOneBaseLargePairOneDir"] = direction["n_one_directional"]
    # Diagnostic: the single large pair that is NOT one-directional is
    # behavioral -> environmental mastery, with one reverse instance.
    values["SOneBaseLargePairReverseMax"] = direction["max_reverse"]
    flow = stats["s1_baseline_node_flow"]
    affect = flow["AFFECT_EXPRESSION"]
    values["SOneBaseAffectExprCases"] = affect["reference_cases"]
    values["SOneBaseAffectExprCorrect"] = affect["correct"]
    values["SOneBaseAffectExprPredicted"] = affect["predicted"]
    values["SOneBaseAffectExprRecallPct"] = tex_pct(affect["recall"])
    values["SOneBaseAffectExprPrecisionPct"] = tex_pct(affect["precision"])
    for suffix, node in (
        ("Emo", "EMOTIONAL_MASTERY"),
        ("Env", "ENVIRONMENTAL_MASTERY"),
        ("Tr", "TRAUMA_REPLAY"),
        ("Beh", "BEHAVIORAL_MASTERY"),
    ):
        values[f"SOneBase{suffix}NetGain"] = flow[node]["net"]
    values["SOneBaseTrLost"] = flow["TRAUMA_REPLAY"]["lost"]
    # §4.5.1 ablation conditions (same suffix family as the dev conditions).
    # The full-pipeline reference shares its CSV with s1_baseline, so its
    # macros are the \SOneBase* family — no Abl alias is emitted for it.
    abl = stats["ablations"]
    for prefix, key in (
        ("AblStageMerge", "stage_merge"),
        ("AblNoSeg", "no_segmentation"),
        ("AblMetaOff", "meta_filtering_off"),
        ("AblNaive", "naive_splitter"),
    ):
        cond = abl[key]
        values[f"{prefix}SafetyAccPct"] = tex_pct_two(cond["safety_accuracy"])
        values[f"{prefix}SafetyCorrect"] = cond["safety_correct"]
        values[f"{prefix}RecallPct"] = tex_pct_two(cond["unsafe_recall"])
        values[f"{prefix}FalseAlarms"] = cond["unsafe_false_positive"]
        values[f"{prefix}Misses"] = cond["unsafe_false_negative"]
        values[f"{prefix}NodeAccPct"] = tex_pct_two(cond["node_accuracy"])
        values[f"{prefix}NodeCorrect"] = cond["node_correct"]

    rerun = stats["temp0_rerun_divergence"]
    values["SOneRerunSafetyFlips"] = rerun["s1"]["safety_verdict_flips"]
    values["STwoRerunSafetyFlips"] = rerun["s2"]["safety_verdict_flips"]
    # §4.4.5 run-to-run variation: node flips, the two long-gap pairs against
    # the confidence-fix re-run, the three-execution union, and segment
    # identity on the short-gap safety flips.
    values["SOneRerunNodeFlips"] = rerun["s1"]["node_flips"]
    values["STwoRerunNodeFlips"] = rerun["s2"]["node_flips"]
    values["STwoCorrectedVsBaseSafetyFlips"] = rerun["s2_baseline_vs_corrected"]["safety_verdict_flips"]
    values["STwoCorrectedVsBaseNodeFlips"] = rerun["s2_baseline_vs_corrected"]["node_flips"]
    values["STwoCorrectedVsConsSafetyFlips"] = rerun["s2_consistency_vs_corrected"]["safety_verdict_flips"]
    values["STwoCorrectedVsConsNodeFlips"] = rerun["s2_consistency_vs_corrected"]["node_flips"]
    values["STwoThreeRunFlipCases"] = rerun["s2_three_executions"]["cases_with_safety_flip"]
    values["STwoThreeRunStableCases"] = rerun["s2_three_executions"]["cases_stable"]
    seg_identity = rerun["s2_segment_identity_on_flips"]
    values["STwoRerunSameSegBoundaries"] = seg_identity["same_segment_boundaries"]
    values["STwoRerunSameSegBoundariesTypes"] = seg_identity["same_boundaries_and_types"]

    # §4.4.2: affect/emotional-mastery collapse (post-hoc sensitivity check)
    # and the per-maladaptive-node values for the voting-spread discussion.
    dev_conditions = (
        ("SOneBase", "s1_baseline"),
        ("SOneCons", "s1_consistency"),
        ("STwoBase", "s2_baseline"),
        ("STwoCons", "s2_consistency"),
    )
    for prefix, key in dev_conditions:
        cond = mb[key]
        values[f"{prefix}CollapsedNodeAccPct"] = tex_pct_two(cond["affect_emotional_collapsed_node_accuracy"])
        values[f"{prefix}CollapsedNodeCorrect"] = cond["affect_emotional_collapsed_node_correct"]
    values["HeldoutCollapsedNodeAccPct"] = tex_pct_two(heldout["affect_emotional_collapsed_node_accuracy"])
    values["HeldoutCollapsedNodeCorrect"] = heldout["affect_emotional_collapsed_node_correct"]
    gains = [
        100 * (mb[key]["affect_emotional_collapsed_node_accuracy"] - mb[key]["node_accuracy"])
        for _prefix, key in dev_conditions
    ]
    values["CollapsedGainMinPts"] = f"{min(gains):.1f}"
    values["CollapsedGainMaxPts"] = f"{max(gains):.1f}"
    node_short = {
        "AVOIDANCE": "Avo",
        "INTERRUPTION": "Int",
        "SUPPRESSION": "Supp",
        "VIOLENT_REVENGE": "Vr",
        "TRAUMA_REPLAY": "Tr",
    }
    for prefix, key in (("STwoBase", "s2_baseline"), ("STwoCons", "s2_consistency")):
        for node, short in node_short.items():
            values[f"{prefix}{short}NodeAccPct"] = tex_pct(mb[key]["per_node"][node]["node_accuracy"])
    # S1 counterpoint: voting lowered the weakest maladaptive category.
    for prefix, key in (("SOneBase", "s1_baseline"), ("SOneCons", "s1_consistency")):
        values[f"{prefix}MalNodeAccMinPct"] = tex_pct(
            min(mb[key]["per_node"][node]["node_accuracy"] for node in node_short)
        )
    # §4.4.4 run-incidence facts: per-segment completed votes.
    votes = stats["consistency_vote_counts"]
    for prefix, key in (("SOneCons", "s1_consistency"), ("STwoCons", "s2_consistency")):
        values[f"{prefix}SegmentCount"] = votes[key]["n_segments"]
        values[f"{prefix}SegmentsBelowNominal"] = votes[key]["segments_below_nominal"]
        values[f"{prefix}MinSegmentVotes"] = votes[key]["min_votes"]

    # §4.7 uncertainty prose (rewrite): agreement-ratio distribution of the
    # consistency runs (moved out of the old reproducibility subsection).
    agree = stats["consistency_agreement_distribution"]
    for prefix, key in (("SOneCons", "s1_consistency"), ("STwoCons", "s2_consistency")):
        dist = agree[key]
        values[f"{prefix}AgreementN"] = dist["n"]
        values[f"{prefix}UnanimousCount"] = dist["unanimous"]
        values[f"{prefix}UnanimousPct"] = tex_pct(dist["unanimous_share"])
        values[f"{prefix}AgreementMean"] = f"{dist['mean']:.2f}"
        values[f"{prefix}AgreementMin"] = f"{dist['min']:.1f}"
    # §4.8.2 inter-expert agreement on the shared blinded response ratings
    # (system-level rating study, all four dimensions on the 41 shared rows).
    sys_pair = stats["system_expert_evaluation"]["interexpert_shared"]
    values["SysAGKLShared"] = sys_pair["n_shared_complete"]
    for macro, dimension in (
        ("SysAGKLAcc", "clinical_acceptability"),
        ("SysAGKLRedirect", "redirect_quality"),
        ("SysAGKLEndorse", "maladaptive_endorsement"),
        ("SysAGKLValidate", "distress_validation"),
    ):
        dim = sys_pair["dimensions"][dimension]
        values[f"{macro}Count"] = dim["exact_agree"]
        values[f"{macro}Pct"] = tex_pct(dim["agreement_rate"])
        values[f"{macro}Kappa"] = f"{dim['kappa']:.3f}"

    lines = [
        "% Generated by recompute_chapter4.py; do not edit by hand.",
        "% AGKL*WithBoundary macros cover all 35 shared complete rows including",
        "% 7 boundary rows; the chapter's inter-expert table instead uses the 28",
        "% scored non-boundary rows (section_4_3_reference_label_validation.py).",
    ]
    lines.extend(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items())
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def recompute(output_dir: Path) -> dict[str, Any]:
    benchmark_paths = {name: REPO_ROOT / path for name, path in BENCHMARK_INPUTS.items()}
    ablation_paths = {name: REPO_ROOT / path for name, path in ABLATION_INPUTS.items()}
    mal_mal_paths = {name: REPO_ROOT / path for name, path in MAL_MAL_INPUTS.items()}
    other_paths = {name: REPO_ROOT / path for name, path in OTHER_INPUTS.items()}

    dev_index = dataset_index(other_paths["dev_dataset"])
    mal_index = dataset_index(other_paths["mal_mal_dataset"])
    test_index = dataset_index(other_paths["test_dataset"])
    benchmark_rows = {name: read_csv(path) for name, path in benchmark_paths.items()}
    ablation_rows = {name: read_csv(path) for name, path in ablation_paths.items()}

    # The consistency runs store the majority vote separately from the
    # temperature-0 pipeline output. Derive the vote-based verdict so those
    # conditions are scored on the decision rule they describe.
    consistency_derivation: dict[str, Any] = {}
    for name in ("s1_consistency", "s2_consistency"):
        rows, diagnostics = derive_vote_verdicts(benchmark_rows[name])
        reproduced, checked, skipped = verify_pipeline_triage(benchmark_rows[name])
        benchmark_rows[name] = rows
        consistency_derivation[name] = {
            **diagnostics,
            "triage_reproduces_pipeline": reproduced,
            "triage_rows_checked": checked,
            "triage_rows_unparsed": skipped,
        }

    expert_available = expert_inputs_available()
    canonical = None if expert_available else load_canonical_statistics()
    if expert_available:
        ag_rows = merge_spotcheck(
            other_paths["ag_spotcheck"],
            other_paths["spotcheck_key"],
            other_paths["spotcheck_blinded"],
            other_paths["spotcheck_predictions"],
        )
        kl_rows = load_kl_spotcheck(other_paths["kl_spotcheck_merged"])
        pairwise, pairwise_rows = pairwise_expert_metrics(ag_rows, kl_rows)
    else:
        ag_rows, kl_rows, pairwise_rows = [], [], []
        pairwise = canonical["expert_spotcheck"]["ag_kl_pairwise"]

    test_rows = read_csv(other_paths["test_postfix"])
    heldout = [
        row for row in test_rows
        if "buried alive" not in row.get("suite", "").lower()
        and test_index.get(row.get("user_input", ""), {}).get("test_type") != "boundary"
    ]

    # Log-probability confidence is reported for the baseline conditions, where
    # it accompanies the label the classifier emits. The consistency conditions
    # report agreement scored on their majority-vote verdict.
    uq: dict[str, Any] = {}
    for name, rows in benchmark_rows.items():
        uq[name] = {}
        if "consistency" not in name:
            # UNCLASSIFIED sentinel rows carry a 0.0 confidence without code
            # tokens behind it and are excluded from token-confidence
            # analyses (decided 2026-09-04; §4.7.1 states the exclusion).
            uq[name]["logprob"] = uncertainty_metrics(
                rows, "conf", exclude_unclassified=True
            )
        if "consistency" in name:
            # Agreement is scored on the majority-vote verdict, the decision these
            # conditions report. The temperature-0 call inside a consistency run
            # exists only to record log probabilities and takes no part in the
            # voting rule, so its prediction is not used as a scoring target.
            uq[name]["agreement_ratio"] = uncertainty_metrics(
                rows, "agreement_ratio", "vote_safe", "vote_node"
            )
    uq["thresholds"] = {
        "s1_conf_0_90": threshold_metrics(
            benchmark_rows["s1_baseline"], "conf", 0.90, exclude_unclassified=True
        ),
        "s1_conf_0_95": threshold_metrics(
            benchmark_rows["s1_baseline"], "conf", 0.95, exclude_unclassified=True
        ),
        "s2_conf_0_80": threshold_metrics(
            benchmark_rows["s2_baseline"], "conf", 0.80, exclude_unclassified=True
        ),
        # Corrected System 2 confidence is saturated near 1.0, so a System 1 scale
        # cutoff flags nothing and any usable operating point sits in the
        # fourth decimal place. Both are reported in Chapter 4.7.
        "s2_corrected_conf_0_90": threshold_metrics(
            benchmark_rows["s2_baseline_corrected"], "conf", 0.90, exclude_unclassified=True
        ),
        "s2_corrected_conf_0_9999": threshold_metrics(
            benchmark_rows["s2_baseline_corrected"], "conf", 0.9999, exclude_unclassified=True
        ),
        # §4.7.2: the agreement ratio takes four values, so exactly three
        # cutoffs exist; scored on the vote verdict like the AUROCs.
        "s2_cons_agreement_0_5": threshold_metrics(
            benchmark_rows["s2_consistency"], "agreement_ratio", 0.5, "vote_safe", "vote_node"
        ),
        "s2_cons_agreement_0_7": threshold_metrics(
            benchmark_rows["s2_consistency"], "agreement_ratio", 0.7, "vote_safe", "vote_node"
        ),
        "s2_cons_agreement_0_9": threshold_metrics(
            benchmark_rows["s2_consistency"], "agreement_ratio", 0.9, "vote_safe", "vote_node"
        ),
    }
    # §4.7.2 prose guards: strictest agreement cutoff selects 25 cases (14
    # node errors, two of the seven safety errors); the other two select
    # 107 and 200.
    agree_thresholds = [
        uq["thresholds"][key] for key in
        ("s2_cons_agreement_0_5", "s2_cons_agreement_0_7", "s2_cons_agreement_0_9")
    ]
    if (
        [t["n_flagged"] for t in agree_thresholds] != [25, 107, 200]
        or agree_thresholds[0]["n_node_errors_flagged"] != 14
        or agree_thresholds[0]["n_safety_errors_caught"] != 2
        or agree_thresholds[0]["n_safety_errors"] != 7
    ):
        raise SystemExit(
            "s2_cons_agreement thresholds: §4.7.2 cites 25/107/200 flagged, "
            "with 14 node errors and two of seven safety errors at the strictest"
        )

    # §4.7.1 prose: 29 of the 42 correct-safety flags at the S1 0.90 cutoff
    # carry a wrong primary node (30 node errors among all 43 flagged).
    s1_090 = uq["thresholds"]["s1_conf_0_90"]
    if (
        s1_090["n_correct_safety_flagged_node_errors"] != 29
        or s1_090["n_node_errors_flagged"] != 30
    ):
        raise SystemExit(
            "s1_conf_0_90: §4.7.1 cites 29 of the 42 correct-safety flags as "
            "node errors (30 node errors among the 43 flagged)"
        )
    if uq["thresholds"]["s1_conf_0_95"]["n_correct_safety_flagged_node_errors"] != 40:
        raise SystemExit(
            "s1_conf_0_95: §4.7.1 cites 40 of the 67 correct-safety flags as node errors"
        )
    if uq["thresholds"]["s2_corrected_conf_0_9999"]["n_correct_safety_flagged_node_errors"] != 50:
        raise SystemExit(
            "s2_corrected_conf_0_9999: §4.7.1 cites 50 further flagged cases "
            "with a wrong primary node"
        )

    uq_dist = uncertainty_distributions(
        benchmark_rows["s1_baseline"],
        benchmark_rows["s2_baseline_corrected"],
        benchmark_rows["s2_consistency"],
    )
    # Guards: the distribution figure must keep matching the threshold and
    # agreement values the §4.7 prose cites.
    for key in ("s1_baseline_conf", "s2_corrected_conf"):
        hist = uq_dist[key]
        if sum(hist["counts"]) + len(hist["below_range_values"]) != hist["n"]:
            raise SystemExit(f"{key}: histogram counts do not partition the values")
    if uq_dist["s1_baseline_conf"]["n"] != uq["s1_baseline"]["logprob"]["n"]:
        raise SystemExit("s1_baseline_conf: histogram n diverged from AUROC n")
    if uq_dist["s1_baseline_conf"]["n_below_0_90"] != uq["thresholds"]["s1_conf_0_90"]["n_flagged"]:
        raise SystemExit("s1_baseline_conf: sub-0.90 count diverged from the threshold table")
    s2_dist = uq_dist["s2_corrected_conf"]
    if s2_dist["n"] != uq["s2_baseline_corrected"]["logprob"]["n"]:
        raise SystemExit("s2_corrected_conf: histogram n diverged from AUROC n")
    if s2_dist["n_excluded_unclassified"] != 1:
        raise SystemExit(
            "s2_corrected_conf: §4.7.1 states exactly one UNCLASSIFIED sentinel "
            "row is excluded from the confidence analysis"
        )
    if s2_dist["below_range_values"]:
        raise SystemExit(
            "s2_corrected_conf: no measured confidence should sit below the "
            "histogram range once the sentinel row is excluded"
        )
    if s2_dist["n"] - s2_dist["n_in_0_999_1"] != 8:
        raise SystemExit(
            "s2_corrected_conf: §4.7.1 states confidence sits in [0.9990, 1.0000] "
            "for all but eight of the measured cases"
        )
    if s2_dist["n_below_0_90"] != uq["thresholds"]["s2_corrected_conf_0_90"]["n_flagged"]:
        raise SystemExit("s2_corrected_conf: sub-0.90 count diverged from the threshold table")
    if s2_dist["n_below_0_9999"] != uq["thresholds"]["s2_corrected_conf_0_9999"]["n_flagged"]:
        raise SystemExit("s2_corrected_conf: sub-0.9999 count diverged from the threshold table")
    if sum(s2_dist["zoom"]["counts"]) != s2_dist["n_in_0_999_1"]:
        raise SystemExit("s2_corrected_conf: zoom histogram does not cover the saturated band")
    # §4.7.1 held-out replication: mean System 1 confidence split by
    # primary-node correctness on the scored held-out set (`heldout` above:
    # buried-alive suite and boundary cases excluded, 231 rows; two META_ONLY
    # rows carry no confidence). Guards pin the two prose-cited means.
    def _mean_conf(rows_subset: list[dict[str, Any]], key: str = "classifier_confidence") -> float:
        return rounded(sum(float(row[key]) for row in rows_subset) / len(rows_subset), 3)

    heldout_with_conf = [row for row in heldout if str(row.get("conf", "")).strip()]
    heldout_node_right = [r for r in heldout_with_conf if r.get("pred_node") == r.get("gt_node")]
    heldout_node_wrong = [r for r in heldout_with_conf if r.get("pred_node") != r.get("gt_node")]
    uq["heldout_confidence_by_node_correctness"] = {
        "basis": "scored held-out cases (buried-alive suite and boundary cases excluded) with a confidence value",
        "n_scored": len(heldout),
        "n_with_conf": len(heldout_with_conf),
        "n_without_conf": len(heldout) - len(heldout_with_conf),
        "node_correct": {"n": len(heldout_node_right), "mean_conf": _mean_conf(heldout_node_right, "conf")},
        "node_wrong": {"n": len(heldout_node_wrong), "mean_conf": _mean_conf(heldout_node_wrong, "conf")},
    }
    ho_conf = uq["heldout_confidence_by_node_correctness"]
    if (
        ho_conf["n_scored"] != 231
        or ho_conf["n_with_conf"] != 229
        or ho_conf["node_wrong"]["mean_conf"] != 0.937
        or ho_conf["node_correct"]["mean_conf"] != 0.982
    ):
        raise SystemExit(
            "heldout_confidence_by_node_correctness: §4.7.1 cites 0.937 (wrong node) / "
            "0.982 (right node) on 229 of 231 scored held-out cases"
        )

    # §4.7.1 KL descriptive support: mean classifier confidence split by
    # node agreement on KL's complete NON-BOUNDARY spot-check rows. Boundary
    # rows carry the author's internal-review annotation in `reference_node`
    # (build_spotcheck_app.py), not a benchmark reference label, and §4.6.2
    # states they have no reference node — so they are excluded here as in
    # every other reference-agreement figure (analyze_expert_annotations
    # row_slice). 98 complete rows, 81 non-boundary, 80 carry a confidence
    # value. Guards pin the four prose-cited means.
    if expert_available:
        kl_complete_all = [
            row for row in kl_rows
            if str(row.get("expert_node", "")).strip() and str(row.get("expert_safe", "")).strip()
            and str(row.get("classifier_confidence", "")).strip()
        ]
        kl_complete = [row for row in kl_complete_all if row.get("source_bucket") != "boundary"]
        cls_ref_agree = [r for r in kl_complete if as_bool(r.get("classifier_reference_node_agreement"))]
        cls_ref_dis = [r for r in kl_complete if as_bool(r.get("classifier_reference_node_agreement")) is False]
        kl_ref_agree = [r for r in kl_complete if as_bool(r.get("primary_node_agreement"))]
        kl_ref_dis = [r for r in kl_complete if as_bool(r.get("primary_node_agreement")) is False]
        uq["kl_spotcheck_confidence"] = {
            "basis": "KL complete non-boundary spot-check rows with a confidence value",
            "n_complete_with_conf": len(kl_complete),
            "n_boundary_excluded": len(kl_complete_all) - len(kl_complete),
            "classifier_reference_agree": {"n": len(cls_ref_agree), "mean_conf": _mean_conf(cls_ref_agree)},
            "classifier_reference_disagree": {"n": len(cls_ref_dis), "mean_conf": _mean_conf(cls_ref_dis)},
            "kl_reference_agree": {"n": len(kl_ref_agree), "mean_conf": _mean_conf(kl_ref_agree)},
            "kl_reference_disagree": {"n": len(kl_ref_dis), "mean_conf": _mean_conf(kl_ref_dis)},
        }
        kl_conf = uq["kl_spotcheck_confidence"]
        if (
            kl_conf["n_complete_with_conf"] != 80
            or kl_conf["n_boundary_excluded"] != 17
            or kl_conf["classifier_reference_disagree"]["mean_conf"] != 0.939
            or kl_conf["classifier_reference_agree"]["mean_conf"] != 0.984
            or kl_conf["kl_reference_disagree"]["mean_conf"] != 0.958
            or kl_conf["kl_reference_agree"]["mean_conf"] != 0.985
        ):
            raise SystemExit(
                "kl_spotcheck_confidence: §4.7.1 cites 0.939/0.984 (classifier-"
                "reference) and 0.958/0.985 (KL-reference) on 80 complete "
                "non-boundary rows (17 boundary rows excluded)"
            )
    else:
        uq["kl_spotcheck_confidence"] = canonical["uncertainty"]["kl_spotcheck_confidence"]

    jitter = s2_confidence_run_jitter(
        benchmark_rows["s2_baseline_corrected"],
        read_csv(other_paths["s2_logprob_diagnostic"]),
    )
    # §4.7.1 threshold-stability facts: pin the values the prose cites.
    if jitter["n_matched_segments"] != 91 or jitter["n_delta_ge_1e_4"] != 13:
        raise SystemExit(
            "s2_confidence_run_jitter: §4.7.1 cites 13 of 91 matched segments "
            "moving by at least one unit in the fourth decimal"
        )
    uq_dist["s2_confidence_run_jitter"] = jitter

    # §4.7.2 complementarity observation: the two S2-consistency missed
    # unsafe cases and their agreement values, plus the percentile anchors
    # the prose cites. Two-case observation; guards pin every cited value.
    miss_rows = [
        row for row in scored_rows(benchmark_rows["s2_consistency"])
        if as_bool(row.get("gt_safe")) is False and as_bool(row.get("vote_safe")) is True
    ]
    if len(miss_rows) != 2:
        raise SystemExit("s2_consistency: §4.7.2 describes exactly two missed unsafe cases")
    miss_agreement = {row["gt_node"]: float(row["agreement_ratio"]) for row in miss_rows}
    counts_by_value = uq_dist["s2_consistency_agreement"]["value_counts"]
    at_or_below_0_6 = counts_by_value.get("0.4", 0) + counts_by_value.get("0.6", 0)
    at_or_below_0_8 = at_or_below_0_6 + counts_by_value.get("0.8", 0)
    if (
        miss_agreement.get("TRAUMA_REPLAY") != 0.6
        or miss_agreement.get("VIOLENT_REVENGE") != 0.8
        or at_or_below_0_6 != 107
        or at_or_below_0_8 != 200
    ):
        raise SystemExit(
            "s2_consistency_miss_agreement: §4.7.2 cites agreement 0.6 (trauma "
            "replay, 107/417 at or below) and 0.8 (violent revenge, 200/417 at "
            "or below)"
        )
    uq["s2_consistency_miss_agreement"] = {
        "trauma_replay_agreement": miss_agreement["TRAUMA_REPLAY"],
        "violent_revenge_agreement": miss_agreement["VIOLENT_REVENGE"],
        "n_at_or_below_0_6": at_or_below_0_6,
        "n_at_or_below_0_8": at_or_below_0_8,
        "n_scored": uq_dist["s2_consistency_agreement"]["n"],
    }

    # §4.7.2 unanimous-error facts: agreement assigned its best value to 35
    # of the S1 consistency condition's 94 node errors (sharing the top
    # score with 231 correct cases). Guards pin the prose-cited counts.
    s1_agree_rows = [
        row for row in scored_rows(benchmark_rows["s1_consistency"])
        if str(row.get("agreement_ratio", "")).strip() not in {"", "nan", "None"}
    ]
    s1_node_errors = [row for row in s1_agree_rows if row.get("gt_node") != row.get("vote_node")]
    s1_unanimous_errors = [row for row in s1_node_errors if float(row["agreement_ratio"]) == 1.0]
    s1_unanimous_total = sum(float(row["agreement_ratio"]) == 1.0 for row in s1_agree_rows)
    # §4.7.2 caveat enumeration: eleven S1-consistency safety errors.
    s1_cons_safety_errors = sum(
        as_bool(row.get("gt_safe")) != as_bool(row.get("vote_safe")) for row in s1_agree_rows
    )
    if s1_cons_safety_errors != 11:
        raise SystemExit(
            "s1_consistency: §4.7.2's caveat enumeration cites eleven safety errors"
        )
    if (
        len(s1_node_errors) != 94
        or len(s1_unanimous_errors) != 35
        or s1_unanimous_total != 266
    ):
        raise SystemExit(
            "s1_consistency: §4.7.2 cites 35 of 94 node errors with unanimous "
            "votes, sharing the top score with 231 correct cases"
        )
    uq["s1_consistency_unanimous_errors"] = {
        "n_node_errors": len(s1_node_errors),
        "n_unanimous_node_errors": len(s1_unanimous_errors),
        "n_unanimous_total": s1_unanimous_total,
        "n_unanimous_node_correct": s1_unanimous_total - len(s1_unanimous_errors),
    }

    agree_dist = agreement_distribution(benchmark_rows["s2_consistency"])
    agree_panel = uq_dist["s2_consistency_agreement"]
    if agree_panel["n"] != agree_dist["n"]:
        raise SystemExit("s2_consistency_agreement: n diverged from the agreement distribution")
    if agree_panel["value_counts"].get("1.0", 0) != agree_dist["unanimous"]:
        raise SystemExit("s2_consistency_agreement: unanimous count diverged")

    # §4.8.2 prose guards: on the 41 shared blinded response rows, the prose
    # cites the four exact-agreement counts and reads the chance-corrected
    # values as clearly above zero only for clinical acceptability (kappa
    # 0.416) with the other three dimensions close to zero. Pin all cited
    # values plus the ordering claim.
    if expert_available:
        kl_system_rows = read_csv(other_paths["kl_system_merged"])
        ag_system_rows = read_csv(other_paths["ag_system_merged"])
        system_interexpert = system_eval_interexpert(kl_system_rows, ag_system_rows)
        if system_interexpert["n_shared_complete"] != 41:
            raise SystemExit(
                "system-eval inter-expert: §4.8.2 reports 41 shared response rows"
            )
        expected_interexpert = {
            "clinical_acceptability": (33, 0.416),
            "redirect_quality": (19, 0.023),
            "maladaptive_endorsement": (28, 0.1),
            "distress_validation": (12, 0.036),
        }
        for dimension, (expected_agree, expected_kappa) in expected_interexpert.items():
            got = system_interexpert["dimensions"][dimension]
            if got["n"] != 41 or got["exact_agree"] != expected_agree or got["kappa"] != expected_kappa:
                raise SystemExit(
                    f"system-eval inter-expert {dimension}: §4.8.2 cites "
                    f"{expected_agree}/41 exact agreement and kappa {expected_kappa}"
                )
        acceptability_kappa = system_interexpert["dimensions"]["clinical_acceptability"]["kappa"]
        if any(
            acceptability_kappa <= metrics["kappa"]
            for dimension, metrics in system_interexpert["dimensions"].items()
            if dimension != "clinical_acceptability"
        ):
            raise SystemExit(
                "system-eval inter-expert: clinical acceptability no longer has the "
                "highest chance-corrected agreement -- §4.8.2 states it does"
            )

        # §4.8.2 prose guards: the opposite movement of distress validation and
        # maladaptive endorsement between the two experts survives on identical
        # material -- on the 19 cases both completed as pairs, the sign-adjusted
        # mean changes still run in opposite directions on both dimensions.
        shared_deltas = system_eval_shared_paired_deltas(kl_system_rows, ag_system_rows)
        if shared_deltas["n_shared_pairs"] != 19:
            raise SystemExit("system-eval shared pairs: §4.8.2 reports 19 shared pairs")
        expected_shared = {
            # dimension: (kl n, kl mean, ag n, ag mean) -- paired-table sign
            # convention (positive favors safety augmentation).
            "distress_validation": (19, 0.158, 19, -0.158),
            "maladaptive_endorsement": (19, -0.368, 17, 0.118),
        }
        for dimension, (kl_n, kl_mean, ag_n, ag_mean) in expected_shared.items():
            got = shared_deltas["dimensions"][dimension]
            if (
                got["kl"]["n_valid_pairs"] != kl_n
                or got["kl"]["mean_change"] != kl_mean
                or got["ag"]["n_valid_pairs"] != ag_n
                or got["ag"]["mean_change"] != ag_mean
            ):
                raise SystemExit(
                    f"system-eval shared pairs {dimension}: values diverged from "
                    "the §4.8.2 like-for-like check"
                )
            if got["kl"]["mean_change"] * got["ag"]["mean_change"] >= 0:
                raise SystemExit(
                    f"system-eval shared pairs {dimension}: the two experts no "
                    "longer move in opposite directions on identical material -- "
                    "§4.8.2 states they do"
                )

        # §4.8 expert-ratings figure guards: pin every distribution the figure
        # draws. Clinical acceptability, redirect quality, and maladaptive
        # endorsement (levels + n/a) must match tab:expert-system-condition-
        # summary (counts + its n/a note); distress validation is absent from
        # that table, so its values are figure-cited only — pinned from the
        # merged CSVs (verified 2026-09-16).
        expert_dist = expert_rating_distributions(kl_system_rows, ag_system_rows)
        expected_rating_distributions: dict[tuple[str, str, int], dict[str, tuple[tuple[int, int, int], int]]] = {
            # (rater, condition, n): {dimension: ((worst, middle, best), n_na)}
            ("kl", "control", 43): {
                "clinical_acceptability": ((5, 9, 29), 0),
                "redirect_quality": ((11, 16, 16), 0),
                "distress_validation": ((9, 27, 0), 7),
                "maladaptive_endorsement": ((0, 0, 39), 4),
            },
            ("kl", "treatment", 42): {
                "clinical_acceptability": ((9, 12, 21), 0),
                "redirect_quality": ((15, 8, 18), 1),
                "distress_validation": ((14, 12, 9), 7),
                "maladaptive_endorsement": ((3, 2, 32), 5),
            },
            ("ag", "control", 21): {
                "clinical_acceptability": ((1, 1, 19), 0),
                "redirect_quality": ((1, 2, 18), 0),
                "distress_validation": ((3, 3, 15), 0),
                "maladaptive_endorsement": ((2, 5, 14), 0),
            },
            ("ag", "treatment", 20): {
                "clinical_acceptability": ((3, 2, 15), 0),
                "redirect_quality": ((2, 2, 16), 0),
                "distress_validation": ((0, 8, 12), 0),
                "maladaptive_endorsement": ((1, 2, 15), 2),
            },
        }
        for (rater, condition, expected_n), dims in expected_rating_distributions.items():
            block = expert_dist[rater][condition]
            if block["n"] != expected_n:
                raise SystemExit(
                    f"expert rating distributions {rater}/{condition}: "
                    f"n={block['n']} diverged from the condition-summary table ({expected_n})"
                )
            for dimension, (expected_levels, expected_na) in dims.items():
                got = block[dimension]
                if tuple(got["levels"].values()) != expected_levels or got["n_na"] != expected_na:
                    raise SystemExit(
                        f"expert rating distributions {rater}/{condition}/{dimension}: "
                        f"{got} diverged from the pinned figure values "
                        f"{expected_levels} + n/a {expected_na}"
                    )
                if sum(got["levels"].values()) + got["n_na"] != block["n"]:
                    raise SystemExit(
                        f"expert rating distributions {rater}/{condition}/{dimension}: "
                        "level counts + n/a do not partition the condition's rows"
                    )
    else:
        kl_system_rows = ag_system_rows = None
        system_interexpert = canonical["system_expert_evaluation"]["interexpert_shared"]
        shared_deltas = canonical["system_expert_evaluation"]["shared_paired_deltas"]
        expert_dist = canonical["system_expert_evaluation"]["rating_distributions"]

    if expert_available:
        spotcheck_queue_test_types = dict(
            Counter(row.get("test_type", "") for row in kl_rows)
        )
        expert_spotcheck_block = {
            "kl": spotcheck_slices(kl_rows),
            "ag": spotcheck_slices(ag_rows),
            "ag_kl_pairwise": pairwise,
        }
        system_expert_block = {
            "kl": system_eval_metrics(kl_system_rows),
            "ag": system_eval_metrics(ag_system_rows),
            "interexpert_shared": system_interexpert,
            "shared_paired_deltas": shared_deltas,
            # Figure-ready per-condition distributions (fixed worst -> best
            # level order); drawn by figures/generate_expert_ratings.py.
            "rating_distributions": expert_dist,
        }
    else:
        spotcheck_queue_test_types = canonical["dataset_composition"]["spotcheck_queue_test_types"]
        expert_spotcheck_block = canonical["expert_spotcheck"]
        system_expert_block = canonical["system_expert_evaluation"]

    stats = {
        "schema_version": 1,
        "main_benchmark": {
            name: (
                benchmark_metrics(rows, "vote_safe", "vote_node")
                if name in consistency_derivation
                else benchmark_metrics(rows)
            )
            for name, rows in benchmark_rows.items()
        },
        # Same rows scored on the temperature-0 pipeline output the consistency
        # run also produced. Compared against the matching baseline condition,
        # this isolates run-to-run variation of the deterministic pipeline.
        "consistency_pipeline_rerun": {
            name: benchmark_metrics(benchmark_rows[name])
            for name in consistency_derivation
        },
        "consistency_vote_derivation": consistency_derivation,
        # Two temperature-0 runs of the same config compared row by row
        # (baseline CSV vs the consistency run's recorded pipeline prediction).
        "temp0_rerun_divergence": {
            "s1": rerun_divergence(benchmark_rows["s1_baseline"], benchmark_rows["s1_consistency"]),
            "s2": rerun_divergence(benchmark_rows["s2_baseline"], benchmark_rows["s2_consistency"]),
            # The confidence-fix re-run is a third System 2 execution recorded
            # just over three months after the first two. Elapsed-time provider
            # drift mixes into these long-gap pairs, so they corroborate the
            # scale of the short-gap pair rather than measure pure
            # nondeterminism (thesis §4.4.5).
            "s2_baseline_vs_corrected": rerun_divergence(
                benchmark_rows["s2_baseline"], benchmark_rows["s2_baseline_corrected"]
            ),
            "s2_consistency_vs_corrected": rerun_divergence(
                benchmark_rows["s2_consistency"], benchmark_rows["s2_baseline_corrected"]
            ),
            "s2_three_executions": multi_run_safety_stability(
                [
                    benchmark_rows["s2_baseline"],
                    benchmark_rows["s2_consistency"],
                    benchmark_rows["s2_baseline_corrected"],
                ]
            ),
            "s2_segment_identity_on_flips": segment_identity_on_flips(
                benchmark_rows["s2_baseline"], benchmark_rows["s2_consistency"]
            ),
        },
        "consistency_agreement_distribution": {
            name: agreement_distribution(benchmark_rows[name])
            for name in consistency_derivation
        },
        "consistency_vote_counts": {
            name: vote_count_stats(benchmark_rows[name])
            for name in consistency_derivation
        },
        "ablations": {name: benchmark_metrics(rows) for name, rows in ablation_rows.items()},
        # §4.5.1 row-level miss comparison: the naive splitter against the
        # no-segmentation condition ("recovers six of the seven, introduces
        # one further miss" is not derivable from the aggregate counts).
        "ablation_miss_overlap": {
            "no_segmentation_vs_naive_splitter": missed_unsafe_overlap(
                ablation_rows["no_segmentation"], ablation_rows["naive_splitter"]
            ),
        },
        "segmentation_strata": {
            name: stratified_metrics(ablation_rows[name], dev_index)
            for name in ("full_s1", "no_segmentation")
        },
        "multi_maladaptive": {
            name: mal_mal_metrics(read_csv(path), mal_index)
            for name, path in mal_mal_paths.items()
        },
        "heldout_test_postfix": benchmark_metrics(heldout),
        "heldout_confusion_families": confusion_families(heldout),
        # §4.6.1 works the dev S1 baseline, where the error set is densest.
        "s1_baseline_confusion_families": confusion_families(
            benchmark_rows["s1_baseline"]
        ),
        "s1_baseline_directional_confusions": directional_confusions(
            benchmark_rows["s1_baseline"]
        ),
        "s1_baseline_node_flow": node_flow(benchmark_rows["s1_baseline"]),
        # The keyed entries check the temperature-0 pipeline columns every CSV
        # carries; the ``*_votes`` entries additionally check the majority-vote
        # verdict the consistency conditions are actually reported on, so the
        # §4.6.1 guarantee is guarded on both decision rules.
        "verdict_partition": {
            name: verdict_partition(rows) for name, rows in benchmark_rows.items()
        } | {
            f"{name}_votes": verdict_partition(
                benchmark_rows[name], "vote_safe", "vote_node"
            )
            for name in consistency_derivation
        } | {"heldout_test_postfix": verdict_partition(heldout)},
        # §4.6.1: the leading confusion of every reported condition, scored on
        # the decision rule each condition is reported on (majority vote for
        # the consistency conditions). Guarded in render_tex: the recompute
        # hard-fails if affect expression -> emotional mastery stops being
        # strictly most frequent anywhere.
        "leading_confusions": {
            "s1_baseline": leading_confusion(benchmark_rows["s1_baseline"]),
            "s1_consistency_votes": leading_confusion(
                benchmark_rows["s1_consistency"], "vote_node"
            ),
            "s2_baseline": leading_confusion(benchmark_rows["s2_baseline"]),
            "s2_baseline_corrected": leading_confusion(
                benchmark_rows["s2_baseline_corrected"]
            ),
            "s2_consistency_votes": leading_confusion(
                benchmark_rows["s2_consistency"], "vote_node"
            ),
            "heldout_test_postfix": leading_confusion(heldout),
        },
        "dataset_composition": {
            "dev": dataset_composition(other_paths["dev_dataset"]),
            # Final held-out set: the truncated buried-alive suite is excluded.
            "test_final": dataset_composition(
                other_paths["test_dataset"], exclude_suites=("buried alive",)
            ),
            # The full 100-row expert spot-check queue by test type (both
            # raters saw the same queue; KL's merged file carries all rows).
            "spotcheck_queue_test_types": spotcheck_queue_test_types,
        },
        "uncertainty": uq,
        "uncertainty_distributions": uq_dist,
        "expert_spotcheck": expert_spotcheck_block,
        "system_expert_evaluation": system_expert_block,
        "internal_constraint_redirect_pilot": integration_metrics(read_csv(other_paths["integration_eval"])),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    all_sources = {
        **{f"benchmark.{name}": path for name, path in benchmark_paths.items()},
        **{f"ablation.{name}": path for name, path in ablation_paths.items()},
        **{f"multi_maladaptive.{name}": path for name, path in mal_mal_paths.items()},
        **{f"other.{name}": path for name, path in other_paths.items()},
    }
    provenance = {
        "code": {
            "generator": {
                "path": "experiments/evaluation_statistics/recompute_chapter4.py",
                "sha256": sha256(Path(__file__)),
            },
            "consistency_votes": {
                "path": "experiments/evaluation_statistics/consistency_votes.py",
                "sha256": sha256(REPO_ROOT / "experiments/evaluation_statistics/consistency_votes.py"),
            },
            "expert_annotation_analysis": {
                "path": "experiments/expert_analysis/analyze_expert_annotations.py",
                "sha256": sha256(REPO_ROOT / "experiments/expert_analysis/analyze_expert_annotations.py"),
            },
            "independent_verifier": {
                "path": "experiments/evaluation_statistics/verify_chapter4_statistics.py",
                "sha256": sha256(REPO_ROOT / "experiments/evaluation_statistics/verify_chapter4_statistics.py"),
            },
        },
        "inputs": source_manifest(all_sources),
    }
    if not expert_available:
        provenance["withheld_inputs"] = {
            "inputs": [f"other.{key}" for key in EXPERT_INPUT_KEYS],
            "reason": (
                "The row-level expert annotation tables are personal data of "
                "the two raters and are distributed only with their consent."
            ),
            "carried_from_canonical": list(EXPERT_DERIVED_BLOCKS),
        }
    (output_dir / "chapter4_statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "chapter4_statistics.md").write_text(render_markdown(stats), encoding="utf-8")
    (output_dir / "chapter4_stats.tex").write_text(render_tex(stats), encoding="utf-8")
    if expert_available:
        write_csv(output_dir / "expert_spotcheck_ag_kl_pairwise.csv", pairwise_rows)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = recompute(args.output_dir)
    print(f"Wrote Chapter 4 statistics to {args.output_dir}")
    if expert_inputs_available():
        ag = stats["expert_spotcheck"]["ag"]["all"]
        pair = stats["expert_spotcheck"]["ag_kl_pairwise"]
        print(f"AG complete: {ag['n_complete']}; AG-reference raw safety agreement: {ag['raw_safety_agree']}")
        print(f"AG-KL shared complete: {pair['n_shared_complete']}; strict node agreement: {pair['strict_node_agree']}")
    else:
        print(
            "Expert annotation inputs absent: expert-derived blocks were carried over "
            "from the canonical statistics (see 'withheld_inputs' in provenance.json)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
