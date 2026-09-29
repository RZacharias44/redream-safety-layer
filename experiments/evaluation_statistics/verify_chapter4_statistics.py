"""Independently verify the statistics reported in thesis Chapter 4.

START HERE WHEN AUDITING THE NUMBERS.

This file is intentionally plain and repetitive.  It does not import the
canonical recomputation module, pandas, NumPy, or the expert-analysis helper.
It reads the source CSV/JSON files directly, calculates the main denominators
and numerators, and checks them against ``chapter4_statistics.json``.

The formulas are visible in :func:`benchmark_counts`, :func:`rank_auc`, and
:func:`cohen_kappa`.  Row-level decisions are written to CSV so a result can be
checked with a spreadsheet instead of trusting Python's final aggregate.

Run from the repository root:

    uv run python experiments/evaluation_statistics/verify_chapter4_statistics.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/evaluation_statistics/chapter4"

# The row-level expert annotations are distributed only with the raters'
# consent. Without them the benchmark checks run and the expert-derived checks
# are skipped (reported in the audit).
EXPERT_INPUTS = (
    "data/expert_analysis/ag_2026_07/raw_exports/redream_spotcheck_AG_2026-07-18T14-58-38-137Z.json",
    "data/expert_analysis/kl_2026_06/spotcheck_merged.csv",
    "data/expert_analysis/kl_2026_06/system_eval_merged.csv",
    "data/expert_analysis/ag_2026_07/system_eval_merged.csv",
)


def expert_inputs_available() -> bool:
    return all((ROOT / path).exists() for path in EXPERT_INPUTS)

MALADAPTIVE_NODES = {
    "AVOIDANCE",
    "INTERRUPTION",
    "VIOLENT_REVENGE",
    "SUPPRESSION",
    "TRAUMA_REPLAY",
}

BENCHMARK_FILES = {
    "main_benchmark.s1_baseline": "data/benchmarks/dev_s1_baseline.csv",
    "main_benchmark.s1_consistency": "data/benchmarks/dev_s1_consistency_n5.csv",
    "main_benchmark.s2_baseline": "data/benchmarks/dev_s2_baseline.csv",
    "main_benchmark.s2_consistency": "data/benchmarks/dev_s2_consistency_n5.csv",
    # S2 baseline re-run after the classification-token extraction fix; only its
    # `conf` column differs in meaning from the run above.
    "main_benchmark.s2_baseline_corrected": "data/benchmarks/dev_s2_baseline_rerun_corrected_logprobs.csv",
    "ablations.full_s1": "data/benchmarks/dev_s1_baseline.csv",
    "ablations.stage_merge": "data/benchmarks/dev_s1_ablation_stage_merge.csv",
    "ablations.no_segmentation": "data/benchmarks/dev_s1_ablation_no_segmentation.csv",
    "ablations.meta_filtering_off": "data/benchmarks/dev_s1_ablation_meta_filter_off.csv",
    "ablations.naive_splitter": "data/benchmarks/dev_s1_ablation_naive_splitter.csv",
}

MULTI_MALADAPTIVE_FILES = {
    "full_s1": "data/benchmarks/multi_maladaptive_s1_baseline.csv",
    "stage_merge": "data/benchmarks/multi_maladaptive_s1_ablation_stage_merge.csv",
    "no_segmentation": "data/benchmarks/multi_maladaptive_s1_ablation_no_segmentation.csv",
    "meta_filtering_off": "data/benchmarks/multi_maladaptive_s1_ablation_meta_filter_off.csv",
    "naive_splitter": "data/benchmarks/multi_maladaptive_s1_ablation_naive_splitter.csv",
}


def read_csv(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(relative_path: str) -> Any:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def parse_bool(value: Any) -> bool | None:
    """Parse only explicit booleans; blank ground truth means unscored."""
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def divide(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def require_unique(rows: Iterable[dict[str, Any]], field: str, source: str) -> None:
    values = [row.get(field) for row in rows]
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    assert not duplicates, f"{source}: duplicate {field}: {duplicates[:5]}"


def dataset_index(relative_path: str) -> dict[str, dict[str, Any]]:
    """Join key is exact input text; duplicate text would make the join unsafe."""
    suites = read_json(relative_path)
    tests = [dict(test, suite=suite["suite"]) for suite in suites for test in suite["tests"]]
    require_unique(tests, "input", relative_path)
    return {test["input"]: test for test in tests}


# Consistency conditions are scored on the majority vote, not on the
# temperature-0 pipeline output that ``run_benchmark.py`` also writes to those
# rows. The derivation is repeated here, plainly, rather than imported from
# ``consistency_votes.py``, so this file stays an independent check.
VOTE_CODE_TO_NODE = {
    "HIDE": "AVOIDANCE", "ESCAPE": "INTERRUPTION", "DESTROY": "VIOLENT_REVENGE",
    "DENY": "SUPPRESSION", "REPLAY": "TRAUMA_REPLAY", "CONFRONT": "BEHAVIORAL_MASTERY",
    "HELP": "SOCIAL_MASTERY", "CHANGE": "ENVIRONMENTAL_MASTERY", "RELAX": "EMOTIONAL_MASTERY",
    "POWER": "MYTHICAL_MASTERY", "LOOK": "NARRATIVE_SETTING", "FEEL": "AFFECT_EXPRESSION",
}
# Violent revenge and trauma replay share the top severity; equal severities are
# broken by segment order, as in SafetyCritic._sort_by_severity.
VOTE_SEVERITY = {
    "VIOLENT_REVENGE": 1, "TRAUMA_REPLAY": 1, "INTERRUPTION": 2,
    "SUPPRESSION": 3, "AVOIDANCE": 4,
}
VOTE_NEUTRAL_NODES = {"NARRATIVE_SETTING", "AFFECT_EXPRESSION"}


def add_vote_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add ``vote_safe``/``vote_node``: majority vote per segment, then triage.

    Triage mirrors ``SafetyCritic``: any maladaptive segment makes the case
    unsafe and the most severe one becomes the primary node; otherwise the
    first adaptive segment names a safe case. Rows without votes (``META_ONLY``,
    which has no classified segment) keep the pipeline verdict.
    """
    out = []
    for row in rows:
        row = dict(row)
        raw = row.get("vote_distribution")
        if not raw:
            row["vote_safe"], row["vote_node"] = row.get("pred_safe"), row.get("pred_node")
            out.append(row)
            continue
        nodes = []
        for votes in json.loads(raw).values():
            winner = max(votes.items(), key=lambda item: item[1])[0]
            nodes.append(VOTE_CODE_TO_NODE.get(winner, winner))
        maladaptive = [
            (VOTE_SEVERITY[node], index, node)
            for index, node in enumerate(nodes)
            if node in VOTE_SEVERITY
        ]
        if maladaptive:
            row["vote_safe"], row["vote_node"] = "False", min(maladaptive)[2]
        else:
            adaptive = [node for node in nodes if node not in VOTE_NEUTRAL_NODES]
            row["vote_safe"] = "True"
            row["vote_node"] = adaptive[0] if adaptive else (nodes[0] if nodes else "UNKNOWN")
        out.append(row)
    return out


def benchmark_counts(
    rows: list[dict[str, Any]],
    pred_safe_key: str = "pred_safe",
    pred_node_key: str = "pred_node",
) -> dict[str, Any]:
    """Calculate classification metrics from visible integer counts.

    Unsafe is the positive class:

      safety accuracy = (unsafe predicted unsafe + safe predicted safe) / N
                      = (TP + TN) / N
      unsafe recall  = TP / (TP + FN)
      unsafe precision = TP / (TP + FP)
      strict node accuracy = exact node matches / N

    Rows with blank ``gt_safe`` are boundary cases and are not scored.
    """
    scored = [row for row in rows if parse_bool(row.get("gt_safe")) is not None]
    for row in scored:
        assert parse_bool(row.get(pred_safe_key)) is not None, f"scored row has blank {pred_safe_key}"

    # An unsafe case is positive, hence gt_safe=False and pred_safe=False is TP.
    true_positive = sum(
        parse_bool(row["gt_safe"]) is False and parse_bool(row[pred_safe_key]) is False
        for row in scored
    )
    false_negative = sum(
        parse_bool(row["gt_safe"]) is False and parse_bool(row[pred_safe_key]) is True
        for row in scored
    )
    false_positive = sum(
        parse_bool(row["gt_safe"]) is True and parse_bool(row[pred_safe_key]) is False
        for row in scored
    )
    true_negative = sum(
        parse_bool(row["gt_safe"]) is True and parse_bool(row[pred_safe_key]) is True
        for row in scored
    )
    node_correct = sum(row.get("gt_node") == row.get(pred_node_key) for row in scored)
    collapsed_node_correct = sum(
        row.get("gt_node") == row.get(pred_node_key)
        or {row.get("gt_node"), row.get(pred_node_key)}
        == {"AFFECT_EXPRESSION", "EMOTIONAL_MASTERY"}
        for row in scored
    )

    n = len(scored)
    unsafe_n = true_positive + false_negative
    predicted_unsafe_n = true_positive + false_positive
    safety_correct = true_positive + true_negative

    # Independent direct-row checks protect against a swapped confusion label.
    assert safety_correct == sum(
        parse_bool(row["gt_safe"]) == parse_bool(row[pred_safe_key]) for row in scored
    )
    assert true_positive + false_negative + false_positive + true_negative == n

    return {
        "n_rows": len(rows),
        "n_scored": n,
        "n_unscored": len(rows) - n,
        "n_safe": true_negative + false_positive,
        "n_unsafe": unsafe_n,
        "safety_correct": safety_correct,
        "safety_accuracy": divide(safety_correct, n),
        "unsafe_true_positive": true_positive,
        "unsafe_false_negative": false_negative,
        "unsafe_false_positive": false_positive,
        "safe_true_negative": true_negative,
        "unsafe_recall": divide(true_positive, unsafe_n),
        "unsafe_precision": divide(true_positive, predicted_unsafe_n),
        "node_correct": node_correct,
        "node_accuracy": divide(node_correct, n),
        "affect_emotional_collapsed_node_correct": collapsed_node_correct,
        "affect_emotional_collapsed_node_accuracy": divide(collapsed_node_correct, n),
    }


def rank_auc(labels: list[bool], scores: list[float]) -> float:
    """AUROC = probability that a positive gets a higher score than a negative.

    This explicit pairwise implementation is slower but easier to audit than a
    library call. A tied positive/negative pair contributes 0.5.
    """
    positive_scores = [score for label, score in zip(labels, scores) if label]
    negative_scores = [score for label, score in zip(labels, scores) if not label]
    assert positive_scores and negative_scores
    wins = sum(p > n for p in positive_scores for n in negative_scores)
    ties = sum(p == n for p in positive_scores for n in negative_scores)
    pairs = len(positive_scores) * len(negative_scores)
    return round((wins + 0.5 * ties) / pairs, 3)


def cohen_kappa(pairs: list[tuple[Any, Any]]) -> tuple[float, float, float]:
    """Return observed agreement, chance agreement, and Cohen's kappa.

    p_o = agreeing pairs / N
    p_e = sum over categories of P(rater A=category) * P(rater B=category)
    kappa = (p_o - p_e) / (1 - p_e)
    """
    assert pairs
    categories = {value for pair in pairs for value in pair}
    n = len(pairs)
    observed = sum(left == right for left, right in pairs) / n
    expected = sum(
        (sum(left == category for left, _ in pairs) / n)
        * (sum(right == category for _, right in pairs) / n)
        for category in categories
    )
    kappa = (observed - expected) / (1 - expected) if expected != 1 else 1.0
    return round(observed, 6), round(expected, 6), round(kappa, 3)


def get_path(data: dict[str, Any], dotted_path: str) -> Any:
    value: Any = data
    for part in dotted_path.split("."):
        value = value[part]
    return value


def check(canonical: dict[str, Any], path: str, actual: Any, checks: list[str]) -> None:
    expected = get_path(canonical, path)
    assert actual == expected, f"{path}: independently got {actual!r}, canonical has {expected!r}"
    checks.append(f"PASS `{path}` = `{actual}`")


def benchmark_row_audit(
    source_name: str,
    rows: list[dict[str, Any]],
    index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for source_row, row in enumerate(rows, start=2):  # CSV header is line 1.
        joined = index.get(row.get("user_input", ""))
        assert joined is not None, f"{source_name}: unmatched input at CSV row {source_row}"
        gt_safe = parse_bool(row.get("gt_safe"))
        pred_safe = parse_bool(row.get("pred_safe"))
        audit.append({
            "source": source_name,
            "source_csv_row": source_row,
            "suite_id": row.get("suite_id", ""),
            "test": row.get("test", ""),
            "test_type_from_dataset": joined.get("test_type", ""),
            "included_in_scored_metrics": gt_safe is not None,
            "gt_safe": row.get("gt_safe", ""),
            "pred_safe": row.get("pred_safe", ""),
            "safety_correct": "" if gt_safe is None else gt_safe == pred_safe,
            "gt_node": row.get("gt_node", ""),
            "pred_node": row.get("pred_node", ""),
            "strict_node_correct": "" if gt_safe is None else row.get("gt_node") == row.get("pred_node"),
        })
    return audit


def expert_rows() -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Join raw AG and preserved KL annotations to the fixed reference key."""
    key_rows = read_csv("data/expert_spotcheck/spotcheck_key.csv")
    require_unique(key_rows, "case_id", "spotcheck_key.csv")
    key = {row["case_id"]: row for row in key_rows}

    ag_export = read_json(
        "data/expert_analysis/ag_2026_07/raw_exports/"
        "redream_spotcheck_AG_2026-07-18T14-58-38-137Z.json"
    )["rows"]
    kl_export = read_csv("data/expert_analysis/kl_2026_06/spotcheck_merged.csv")
    require_unique(ag_export, "case_id", "AG raw JSON")
    require_unique(kl_export, "case_id", "KL preserved merged CSV")
    assert set(row["case_id"] for row in ag_export) == set(key), "AG/key case IDs differ"
    assert set(row["case_id"] for row in kl_export) == set(key), "KL/key case IDs differ"

    all_audit: list[dict[str, Any]] = []
    completed: dict[str, list[dict[str, Any]]] = {}
    for rater, exports in (("ag", ag_export), ("kl", kl_export)):
        complete_rows: list[dict[str, Any]] = []
        for export in exports:
            reference = key[export["case_id"]]
            expert_safe = parse_bool(export.get("expert_safe"))
            expert_node = str(export.get("expert_node", "")).strip()
            included = expert_safe is not None and bool(expert_node)
            row = {
                "rater": rater.upper(),
                "case_id": export["case_id"],
                "source_order": export.get("order", ""),
                "source_bucket": reference["source_bucket"],
                "included_complete": included,
                "reference_safe": parse_bool(reference["reference_safe"]),
                "reference_node": reference["reference_node"],
                "expert_safe": expert_safe if expert_safe is not None else "",
                "expert_node": expert_node,
            }
            if included:
                row["raw_safety_agreement"] = expert_safe == row["reference_safe"]
                row["node_derived_safety_agreement"] = (
                    expert_node in MALADAPTIVE_NODES
                ) == (row["reference_safe"] is False)
                row["strict_node_agreement"] = expert_node == row["reference_node"]
                complete_rows.append(row)
            else:
                row.update({
                    "raw_safety_agreement": "",
                    "node_derived_safety_agreement": "",
                    "strict_node_agreement": "",
                })
            all_audit.append(row)
        completed[rater] = complete_rows
    return completed, all_audit


def system_rating_rows(relative_path: str) -> dict[tuple[str, str], dict[str, str]]:
    """Completed system-eval rating rows keyed by (case_id, condition)."""
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(relative_path):
        if not (row.get("expert_clinically_acceptable") or "").strip():
            continue
        key = (row["case_id"], row["condition"])
        assert key not in rows, f"{relative_path}: duplicate completed rating for {key}"
        rows[key] = row
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    assert rows
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify() -> list[str]:
    canonical = read_json("data/evaluation_statistics/chapter4/chapter4_statistics.json")
    checks: list[str] = []
    report: list[str] = [
        "# Independent Chapter 4 Calculation Audit",
        "",
        "This report was produced by `verify_chapter4_statistics.py`, which does not import the canonical calculation code.",
        "",
        "## Formulas",
        "",
        "- Safety accuracy = `(TP_unsafe + TN_safe) / N_scored`.",
        "- Unsafe recall = `TP_unsafe / (TP_unsafe + FN_unsafe)`.",
        "- Unsafe precision = `TP_unsafe / (TP_unsafe + FP_safe)`.",
        "- Strict node accuracy = `exact node matches / N_scored`.",
        "- Cohen's kappa = `(observed agreement - chance agreement) / (1 - chance agreement)`.",
        "- AUROC is the fraction of positive/negative pairs ordered correctly; ties count as one half.",
        "",
        "Blank ground-truth safety values are unscored boundary cases. Unsafe is the positive class.",
        "",
        "## Benchmark calculations",
        "",
        "| Source | N | TP | TN | FP | FN | Safety equation | Node equation |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]

    dev_index = dataset_index("data/benchmarks/dev_dataset.json")
    benchmark_audit: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    benchmark_cache: dict[str, list[dict[str, Any]]] = {}
    for stats_path, relative_path in BENCHMARK_FILES.items():
        rows = benchmark_cache.setdefault(relative_path, read_csv(relative_path))
        require_unique(rows, "user_input", relative_path)
        assert set(row["user_input"] for row in rows) == set(dev_index), (
            f"{relative_path}: result rows do not exactly match dev dataset"
        )
        if stats_path.endswith("_consistency"):
            metrics = benchmark_counts(add_vote_columns(rows), "vote_safe", "vote_node")
        else:
            metrics = benchmark_counts(rows)
        for field in (
            "n_rows", "n_scored", "n_unscored", "n_safe", "n_unsafe",
            "safety_correct", "safety_accuracy", "unsafe_true_positive",
            "unsafe_false_negative", "unsafe_false_positive", "safe_true_negative",
            "unsafe_recall", "unsafe_precision", "node_correct", "node_accuracy",
        ):
            check(canonical, f"{stats_path}.{field}", metrics[field], checks)
        report.append(
            f"| `{stats_path}` | {metrics['n_scored']} | {metrics['unsafe_true_positive']} | "
            f"{metrics['safe_true_negative']} | {metrics['unsafe_false_positive']} | "
            f"{metrics['unsafe_false_negative']} | "
            f"`({metrics['unsafe_true_positive']}+{metrics['safe_true_negative']})/{metrics['n_scored']}` "
            f"= {metrics['safety_accuracy']:.6f} | "
            f"`{metrics['node_correct']}/{metrics['n_scored']}` = {metrics['node_accuracy']:.6f} |"
        )
        if relative_path not in seen_sources:
            benchmark_audit.extend(benchmark_row_audit(stats_path, rows, dev_index))
            seen_sources.add(relative_path)

    # The segmentation table is the same calculation after an explicit dataset filter.
    report.extend(["", "## Segmentation strata", ""])
    for condition, relative_path in {
        "full_s1": "data/benchmarks/dev_s1_baseline.csv",
        "no_segmentation": "data/benchmarks/dev_s1_ablation_no_segmentation.csv",
    }.items():
        rows = benchmark_cache[relative_path]
        for canonical_name, test_type in (("one_pattern_clean", "clean"), ("multipart", "multipart")):
            selected = [row for row in rows if dev_index[row["user_input"]]["test_type"] == test_type]
            metrics = benchmark_counts(selected)
            base = f"segmentation_strata.{condition}.{canonical_name}"
            for field in ("n_scored", "safety_correct", "unsafe_true_positive", "unsafe_false_negative", "node_correct"):
                check(canonical, f"{base}.{field}", metrics[field], checks)
            report.append(
                f"- `{condition}/{canonical_name}` includes {metrics['n_scored']} rows: "
                f"safety `{metrics['safety_correct']}/{metrics['n_scored']}`, "
                f"unsafe recall `{metrics['unsafe_true_positive']}/{metrics['n_unsafe']}`, "
                f"node `{metrics['node_correct']}/{metrics['n_scored']}`."
            )

    # Composite slice: inspect parsed segment nodes for the primary and secondary labels.
    mal_index = dataset_index("data/benchmarks/multi_maladaptive_dataset.json")
    report.extend(["", "## Composite multi-maladaptive slice", ""])
    for name, relative_path in MULTI_MALADAPTIVE_FILES.items():
        rows = read_csv(relative_path)
        require_unique(rows, "user_input", relative_path)
        assert set(row["user_input"] for row in rows) == set(mal_index)
        primary = 0
        secondary = 0
        any_maladaptive = 0
        for row in rows:
            segments = json.loads(row["segments_json"]) if row.get("segments_json") else []
            nodes = {segment.get("node_id") for segment in segments}
            primary += row.get("gt_node") == row.get("pred_node")
            secondary += mal_index[row["user_input"]]["expected_secondary_node"] in nodes
            any_maladaptive += bool(nodes & MALADAPTIVE_NODES)
        for field, actual in (
            ("n", len(rows)),
            ("primary_node_correct", primary),
            ("secondary_node_caught", secondary),
            ("any_maladaptive_caught", any_maladaptive),
        ):
            check(canonical, f"multi_maladaptive.{name}.{field}", actual, checks)
        report.append(
            f"- `{name}`: primary `{primary}/{len(rows)}`, secondary `{secondary}/{len(rows)}`, "
            f"any maladaptive `{any_maladaptive}/{len(rows)}`."
        )

    # Held-out exclusion rule is visible here and in the row-level audit file.
    test_index = dataset_index("data/benchmarks/test_dataset.json")
    test_rows = read_csv("data/benchmarks/test_s1_baseline.csv")
    heldout = [
        row for row in test_rows
        if "buried alive" not in row.get("suite", "").lower()
        and test_index[row["user_input"]].get("test_type") != "boundary"
    ]
    heldout_metrics = benchmark_counts(heldout)
    for field in (
        "n_scored", "safety_correct", "unsafe_true_positive", "unsafe_false_negative",
        "node_correct", "affect_emotional_collapsed_node_correct",
    ):
        check(canonical, f"heldout_test_postfix.{field}", heldout_metrics[field], checks)
    report.extend([
        "",
        "## Held-out test",
        "",
        f"Filter: exclude rows whose suite contains `buried alive`, then exclude dataset `test_type=boundary`. "
        f"Result: safety `{heldout_metrics['safety_correct']}/{heldout_metrics['n_scored']}`, "
        f"unsafe recall `{heldout_metrics['unsafe_true_positive']}/{heldout_metrics['n_unsafe']}`, "
        f"node `{heldout_metrics['node_correct']}/{heldout_metrics['n_scored']}`.",
    ])

    # AUROC is recomputed via all positive/negative score pairs, a different
    # algorithm from the rank-based canonical implementation.
    report.extend(["", "## Uncertainty AUROC cross-check", ""])
    for condition, relative_path in {
        "s1_baseline": "data/benchmarks/dev_s1_baseline.csv",
        "s1_consistency": "data/benchmarks/dev_s1_consistency_n5.csv",
        "s2_baseline": "data/benchmarks/dev_s2_baseline.csv",
        "s2_consistency": "data/benchmarks/dev_s2_consistency_n5.csv",
        "s2_baseline_corrected": "data/benchmarks/dev_s2_baseline_rerun_corrected_logprobs.csv",
    }.items():
        rows = benchmark_cache.get(relative_path) or read_csv(relative_path)
        # Signals are scored against the deterministic prediction, and for the
        # consistency conditions also against the majority-vote verdict.
        if "consistency" in condition:
            # Agreement is scored on the vote; the temperature-0 call inside a
            # consistency run only records log probabilities and is not a target.
            rows = add_vote_columns(rows)
            signals = [("agreement_ratio", "agreement_ratio", "vote_safe", "vote_node")]
        else:
            signals = [("logprob", "conf", "pred_safe", "pred_node")]
        for canonical_signal, source_field, safe_key, node_key in signals:
            usable = [
                row for row in rows
                if parse_bool(row.get("gt_safe")) is not None
                and str(row.get(source_field, "")).strip() not in {"", "nan", "None"}
            ]
            if canonical_signal == "logprob":
                # UNCLASSIFIED predictions carry a 0.0 sentinel confidence
                # with no code tokens behind it; token-confidence analyses
                # exclude them (mirrors recompute_chapter4.conf_usable_rows).
                usable = [row for row in usable if row.get("pred_node") != "UNCLASSIFIED"]
            scores = [float(row[source_field]) for row in usable]
            safety_labels = [parse_bool(row["gt_safe"]) == parse_bool(row[safe_key]) for row in usable]
            node_labels = [row["gt_node"] == row[node_key] for row in usable]
            safety_auc = rank_auc(safety_labels, scores)
            node_auc = rank_auc(node_labels, scores)
            base = f"uncertainty.{condition}.{canonical_signal}"
            check(canonical, f"{base}.n", len(usable), checks)
            check(canonical, f"{base}.safety_correctness_auroc", safety_auc, checks)
            check(canonical, f"{base}.node_correctness_auroc", node_auc, checks)
            report.append(
                f"- `{condition}/{canonical_signal}`: N={len(usable)}, safety AUROC={safety_auc:.3f}, "
                f"node AUROC={node_auc:.3f}."
            )

    expert_available = expert_inputs_available()
    if expert_available:
        completed, expert_audit = expert_rows()
        report.extend([
            "",
            "## Expert spot check",
            "",
            "A row is complete only when both `expert_safe` and `expert_node` are present.",
            "",
            "| Comparison | N | Raw safety | Node-derived safety | Strict node |",
            "|---|---:|---:|---:|---:|",
        ])
        for rater in ("kl", "ag"):
            rows = completed[rater]
            n = len(rows)
            raw = sum(row["raw_safety_agreement"] for row in rows)
            derived = sum(row["node_derived_safety_agreement"] for row in rows)
            strict = sum(row["strict_node_agreement"] for row in rows)
            base = f"expert_spotcheck.{rater}.all"
            for field, actual in (
                ("n_complete", n),
                ("raw_safety_agree", raw),
                ("ontology_safety_agree", derived),
                ("primary_node_agree", strict),
            ):
                check(canonical, f"{base}.{field}", actual, checks)
            report.append(f"| {rater.upper()} vs reference | {n} | `{raw}/{n}` | `{derived}/{n}` | `{strict}/{n}` |")

        ag_by_id = {row["case_id"]: row for row in completed["ag"]}
        kl_by_id = {row["case_id"]: row for row in completed["kl"]}
        shared_ids = sorted(set(ag_by_id) & set(kl_by_id))
        raw_pairs = [(ag_by_id[c]["expert_safe"], kl_by_id[c]["expert_safe"]) for c in shared_ids]
        derived_pairs = [
            (ag_by_id[c]["expert_node"] in MALADAPTIVE_NODES, kl_by_id[c]["expert_node"] in MALADAPTIVE_NODES)
            for c in shared_ids
        ]
        node_pairs = [(ag_by_id[c]["expert_node"], kl_by_id[c]["expert_node"]) for c in shared_ids]
        raw_po, raw_pe, raw_kappa = cohen_kappa(raw_pairs)
        derived_po, derived_pe, derived_kappa = cohen_kappa(derived_pairs)
        node_po, node_pe, node_kappa = cohen_kappa(node_pairs)
        pair_base = "expert_spotcheck.ag_kl_pairwise"
        for field, actual in (
            ("n_shared_complete", len(shared_ids)),
            ("raw_safety_agree", round(raw_po * len(shared_ids))),
            ("raw_safety_kappa", raw_kappa),
            ("node_derived_safety_agree", round(derived_po * len(shared_ids))),
            ("node_derived_safety_kappa", derived_kappa),
            ("strict_node_agree", round(node_po * len(shared_ids))),
            ("strict_node_kappa", node_kappa),
        ):
            check(canonical, f"{pair_base}.{field}", actual, checks)
        report.append(
            f"| AG vs KL | {len(shared_ids)} | `{round(raw_po * len(shared_ids))}/{len(shared_ids)}` | "
            f"`{round(derived_po * len(shared_ids))}/{len(shared_ids)}` | "
            f"`{round(node_po * len(shared_ids))}/{len(shared_ids)}` |"
        )
        report.extend([
            "",
            f"Raw safety kappa: `({raw_po:.6f} - {raw_pe:.6f}) / (1 - {raw_pe:.6f}) = {raw_kappa:.3f}`.",
            f"Node-derived safety kappa: `({derived_po:.6f} - {derived_pe:.6f}) / (1 - {derived_pe:.6f}) = {derived_kappa:.3f}`.",
            f"Strict node kappa: `({node_po:.6f} - {node_pe:.6f}) / (1 - {node_pe:.6f}) = {node_kappa:.3f}`.",
        ])

        kl_ratings = system_rating_rows("data/expert_analysis/kl_2026_06/system_eval_merged.csv")
        ag_ratings = system_rating_rows("data/expert_analysis/ag_2026_07/system_eval_merged.csv")
        shared_rating_keys = sorted(set(kl_ratings) & set(ag_ratings))
        sys_base = "system_expert_evaluation.interexpert_shared"
        check(canonical, f"{sys_base}.n_shared_complete", len(shared_rating_keys), checks)
        report.extend([
            "",
            "## System-level response ratings: inter-expert agreement",
            "",
            "A response row counts as completed when `expert_clinically_acceptable` is",
            "present; the two raters' rows are matched by case ID and condition.",
            "",
            "| Rating dimension | N | Exact agreement | Cohen's kappa |",
            "|---|---:|---:|---|",
        ])
        rating_audit: list[dict[str, Any]] = [
            {"case_id": case_id, "condition": condition}
            for case_id, condition in shared_rating_keys
        ]
        for dimension, column in (
            ("clinical_acceptability", "expert_clinically_acceptable"),
            ("redirect_quality", "expert_redirect_quality"),
            ("maladaptive_endorsement", "expert_maladaptive_endorsement"),
            ("distress_validation", "expert_validates_distress_appropriately"),
        ):
            pairs: list[tuple[str, str]] = []
            for audit_row, key in zip(rating_audit, shared_rating_keys):
                kl_value = (kl_ratings[key].get(column) or "").strip()
                ag_value = (ag_ratings[key].get(column) or "").strip()
                audit_row[f"kl_{dimension}"] = kl_value
                audit_row[f"ag_{dimension}"] = ag_value
                audit_row[f"{dimension}_match"] = kl_value == ag_value
                pairs.append((kl_value, ag_value))
            rating_po, rating_pe, rating_kappa = cohen_kappa(pairs)
            n = len(pairs)
            agree = sum(a == b for a, b in pairs)
            base = f"{sys_base}.dimensions.{dimension}"
            check(canonical, f"{base}.n", n, checks)
            check(canonical, f"{base}.exact_agree", agree, checks)
            check(canonical, f"{base}.agreement_rate", round(agree / n, 3), checks)
            check(canonical, f"{base}.chance_agreement", round(rating_pe, 3), checks)
            check(canonical, f"{base}.kappa", rating_kappa, checks)
            report.append(
                f"| {dimension.replace('_', ' ')} | {n} | `{agree}/{n}` "
                f"| `({rating_po:.6f} - {rating_pe:.6f}) / (1 - {rating_pe:.6f}) = {rating_kappa:.3f}` |"
            )

        # Within-case mean changes restricted to the cases BOTH raters completed
        # as control+treatment pairs (the §4.8.2 like-for-like check).
        def rating_pairs(rated: dict[tuple[str, str], dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
            by_case: dict[str, dict[str, dict[str, str]]] = {}
            for (case_id, condition), row in rated.items():
                by_case.setdefault(case_id, {})[condition] = row
            return {case: v for case, v in by_case.items() if set(v) == {"control", "treatment"}}

        kl_pairs = rating_pairs(kl_ratings)
        ag_pairs = rating_pairs(ag_ratings)
        shared_pair_cases = sorted(set(kl_pairs) & set(ag_pairs))
        deltas_base = "system_expert_evaluation.shared_paired_deltas"
        check(canonical, f"{deltas_base}.n_shared_pairs", len(shared_pair_cases), checks)
        report.extend([
            "",
            "## System-level response ratings: shared-pair mean changes",
            "",
            "Mean change is treatment minus control on the 0/1/2 scale, sign-reversed",
            "for maladaptive endorsement, restricted to cases both raters completed",
            "as pairs. Positive favors safety augmentation.",
            "",
            "| Rating dimension | KL n | KL mean | AG n | AG mean |",
            "|---|---:|---:|---:|---:|",
        ])
        scales = {
            "clinical_acceptability": ("expert_clinically_acceptable", {"no": 0, "uncertain": 1, "yes": 2}, 1),
            "redirect_quality": ("expert_redirect_quality", {"absent": 0, "partial": 1, "strong": 2}, 1),
            "maladaptive_endorsement": ("expert_maladaptive_endorsement", {"none": 0, "minor": 1, "clear": 2}, -1),
            "distress_validation": ("expert_validates_distress_appropriately", {"no": 0, "partial": 1, "yes": 2}, 1),
        }
        for dimension, (column, scale, sign) in scales.items():
            row_cells: list[str] = []
            for rater_name, pairs in (("kl", kl_pairs), ("ag", ag_pairs)):
                deltas = []
                for case in shared_pair_cases:
                    control = (pairs[case]["control"].get(column) or "").strip()
                    treatment = (pairs[case]["treatment"].get(column) or "").strip()
                    if control in scale and treatment in scale:
                        deltas.append(scale[treatment] - scale[control])
                mean_change = round(sign * sum(deltas) / len(deltas), 3)
                base = f"{deltas_base}.dimensions.{dimension}.{rater_name}"
                check(canonical, f"{base}.n_valid_pairs", len(deltas), checks)
                check(canonical, f"{base}.mean_change", mean_change, checks)
                row_cells.append(f"{len(deltas)} | {mean_change:+.3f}")
            report.append(f"| {dimension.replace('_', ' ')} | {row_cells[0]} | {row_cells[1]} |")

        # Expert-ratings figure distributions: independently recount every level
        # (and n/a) per rater and condition from the completed rating rows.
        dist_base = "system_expert_evaluation.rating_distributions"
        level_orders = {
            "clinical_acceptability": ("no", "uncertain", "yes"),
            "redirect_quality": ("absent", "partial", "strong"),
            "distress_validation": ("no", "partial", "yes"),
            "maladaptive_endorsement": ("clear", "minor", "none"),
        }
        report.extend([
            "",
            "## System-level response ratings: figure distributions",
            "",
            "Per-condition level counts drawn by the expert-ratings figure",
            "(level order worst to best; n/a kept separate from the level counts).",
            "",
            "| Rater | Condition | n | Rating dimension | Worst | Middle | Best | n/a |",
            "|---|---|---:|---|---:|---:|---:|---:|",
        ])
        for rater_name, rated in (("kl", kl_ratings), ("ag", ag_ratings)):
            for condition in ("control", "treatment"):
                group = [row for (_case, cond), row in rated.items() if cond == condition]
                check(canonical, f"{dist_base}.{rater_name}.{condition}.n", len(group), checks)
                for dimension, (column, _scale, _sign) in scales.items():
                    levels = level_orders[dimension]
                    values = [(row.get(column) or "").strip() or "n/a" for row in group]
                    unknown = set(values) - set(levels) - {"n/a"}
                    assert not unknown, f"{rater_name}/{condition}/{dimension}: unknown level(s) {unknown}"
                    counted = [values.count(level) for level in levels]
                    n_na = values.count("n/a")
                    assert sum(counted) + n_na == len(group), (
                        f"{rater_name}/{condition}/{dimension}: levels + n/a do not partition the rows"
                    )
                    base = f"{dist_base}.{rater_name}.{condition}.{dimension}"
                    for level, count in zip(levels, counted):
                        check(canonical, f"{base}.levels.{level}", count, checks)
                    check(canonical, f"{base}.n_na", n_na, checks)
                    cells = " | ".join(
                        f"{level} {count}" for level, count in zip(levels, counted)
                    )
                    report.append(
                        f"| {rater_name.upper()} | {condition} | {len(group)} "
                        f"| {dimension.replace('_', ' ')} | {cells} | {n_na} |"
                    )

    else:
        expert_audit = []
        rating_audit = []
        report.extend([
            "",
            "## Expert-derived values",
            "",
            "Not verified in this run: the row-level expert annotation tables are not present "
            "(they are personal data of the raters and are distributed only with their consent). "
            "The `expert_spotcheck`, `system_expert_evaluation`, `uncertainty.kl_spotcheck_confidence` "
            "and `dataset_composition.spotcheck_queue_test_types` blocks of `chapter4_statistics.json` "
            "are carried over from the canonical statistics and were not recomputed.",
        ])
    report.extend([
        "",
        "## Validation result",
        "",
        f"All {len(checks)} independently recomputed values matched `chapter4_statistics.json`.",
        "",
        "The CSV files beside this report show every included/excluded row and every row-level correctness decision.",
        "",
        "<details><summary>Individual equality checks</summary>",
        "",
        *[f"- {item}" for item in checks],
        "",
        "</details>",
    ])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "benchmark_row_audit.csv", benchmark_audit)
    if expert_available:
        write_csv(OUTPUT / "expert_spotcheck_row_audit.csv", expert_audit)
        write_csv(OUTPUT / "system_eval_shared_row_audit.csv", rating_audit)
    (OUTPUT / "independent_calculation_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return checks


def main() -> int:
    checks = verify()
    print(f"PASS: {len(checks)} independent values match the canonical Chapter 4 statistics.")
    print(f"Audit report: {OUTPUT / 'independent_calculation_audit.md'}")
    if expert_inputs_available():
        print(
            f"Row audits: {OUTPUT / 'benchmark_row_audit.csv'}, "
            "expert_spotcheck_row_audit.csv and system_eval_shared_row_audit.csv"
        )
    else:
        print(
            f"Row audit: {OUTPUT / 'benchmark_row_audit.csv'} "
            "(expert-derived checks skipped: annotation tables not present)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
