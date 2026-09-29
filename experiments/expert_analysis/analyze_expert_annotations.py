"""Analyze expert annotation exports for the ReDream thesis evaluation.

The analysis deliberately separates three related but distinct quantities:

1. Raw expert safety/acceptability fields as entered in the rater UI.
2. Ontology safety derived from the selected clinical node.
3. Adequacy penalties, where a rater marks a neutral/adaptive node as unsafe
   because the rescripting move was not sufficiently adaptive.

Usage (the defaults reproduce the KL analysis from the tracked raw exports):
    uv run python experiments/expert_analysis/analyze_expert_annotations.py

    uv run python experiments/expert_analysis/analyze_expert_annotations.py \
        --spotcheck-export data/expert_analysis/ag_2026_07/raw_exports/redream_spotcheck_AG_2026-07-18T14-58-38-137Z.json \
        --system-export data/expert_analysis/ag_2026_07/raw_exports/redream_system_eval_AG_2026-07-18T14-59-13-387Z.json \
        --rater-id AG --output-dir data/expert_analysis/ag_2026_07

Raters are identified by their initials only. ``--rater-name`` overrides the
label used in the generated prose.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.expert_spotcheck.build_spotcheck_app import (
    ADAPTIVE_NODES,
    MALADAPTIVE_NODES,
    NEUTRAL_NODES,
    REPO_ROOT,
)


MALADAPTIVE_NODE_SET = set(MALADAPTIVE_NODES)
NON_MALADAPTIVE_NODE_SET = set(ADAPTIVE_NODES) | set(NEUTRAL_NODES)

CLINICAL_ACCEPTABILITY_SCORE = {"no": 0, "uncertain": 1, "yes": 2}
REDIRECT_QUALITY_SCORE = {"absent": 0, "partial": 1, "strong": 2}
VALIDATION_SCORE = {"no": 0, "partial": 1, "yes": 2}
MALADAPTIVE_ENDORSEMENT_SCORE = {"none": 0, "minor": 1, "clear": 2}

ADEQUACY_NOTE_PATTERNS = (
    "already adaptive",
    "attempt was adaptive",
    "attempt was already adaptive",
    "patient's attempt was adaptive",
    "patient's attempt was already adaptive",
    "patient´s attempt was already adaptive",
    "patients attempt was already adaptive",
    "participant´s attempt was already adaptive",
    "not adaptive",
    "not explicit enough",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_export(path: Path) -> list[dict[str, Any]]:
    """Read an annotation export without changing the rater's recorded values.

    The browser app exports equivalent CSV and JSON artifacts. Supporting both
    formats lets the checked-in primary artifact be used directly and avoids a
    manual conversion step in the audit trail.
    """
    if path.suffix.lower() == ".csv":
        return read_csv(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"Expected a JSON object with a 'rows' list: {path}")
        return rows
    raise ValueError(f"Unsupported annotation export format: {path.suffix}")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "safe"}:
        return True
    if text in {"false", "0", "no", "unsafe"}:
        return False
    return None


def as_lower(value: Any) -> str:
    return str(value or "").strip().lower()


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100 * numerator / denominator:.1f}%"


def fmt_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.3f}"


def node_is_maladaptive(node: Any) -> bool | None:
    node_text = str(node or "").strip().upper()
    if not node_text:
        return None
    if node_text in MALADAPTIVE_NODE_SET:
        return True
    if node_text in NON_MALADAPTIVE_NODE_SET:
        return False
    return None


def reference_is_maladaptive(row: dict[str, Any]) -> bool | None:
    from_node = node_is_maladaptive(row.get("reference_node"))
    if from_node is not None:
        return from_node
    reference_safe = as_bool(row.get("reference_safe"))
    if reference_safe is None:
        return None
    return not reference_safe


def classifier_is_maladaptive(row: dict[str, Any]) -> bool | None:
    from_node = node_is_maladaptive(row.get("classifier_node"))
    if from_node is not None:
        return from_node
    classifier_safe = as_bool(row.get("classifier_safe"))
    if classifier_safe is None:
        return None
    return not classifier_safe


def raw_expert_unsafe(row: dict[str, Any]) -> bool | None:
    safe_value = as_bool(row.get("expert_safe"))
    if safe_value is None:
        return None
    return not safe_value


def bool_text(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def label_agreement(a: Any, b: Any) -> bool | None:
    if a is None or b is None:
        return None
    return a == b


def cohen_kappa(pairs: list[tuple[Any, Any]]) -> float | None:
    complete = [(a, b) for a, b in pairs if a not in ("", None) and b not in ("", None)]
    if not complete:
        return None
    total = len(complete)
    observed = sum(a == b for a, b in complete) / total
    a_counts = Counter(a for a, _b in complete)
    b_counts = Counter(b for _a, b in complete)
    expected = sum((a_counts[label] / total) * (b_counts[label] / total) for label in set(a_counts) | set(b_counts))
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def derive_spotcheck_fields(row: dict[str, Any]) -> dict[str, Any]:
    ref_maladaptive = reference_is_maladaptive(row)
    classifier_maladaptive = classifier_is_maladaptive(row)
    classifier_safe = as_bool(row.get("classifier_safe"))
    classifier_unsafe = None if classifier_safe is None else not classifier_safe
    expert_node_maladaptive = node_is_maladaptive(row.get("expert_node"))
    expert_unsafe = raw_expert_unsafe(row)
    raw_safety_agree = label_agreement(expert_unsafe, ref_maladaptive)
    ontology_safety_agree = label_agreement(expert_node_maladaptive, ref_maladaptive)
    primary_node_agree = bool(row.get("expert_node")) and row.get("expert_node") == row.get("reference_node")
    classifier_reference_safety_agree = label_agreement(classifier_maladaptive, ref_maladaptive)
    classifier_reference_node_agree = (
        bool(row.get("classifier_node")) and row.get("classifier_node") == row.get("reference_node")
    )
    classifier_expert_raw_safety_agree = label_agreement(classifier_unsafe, expert_unsafe)
    classifier_expert_ontology_safety_agree = label_agreement(classifier_maladaptive, expert_node_maladaptive)
    classifier_expert_node_agree = (
        bool(row.get("classifier_node")) and bool(row.get("expert_node")) and row.get("classifier_node") == row.get("expert_node")
    )
    adequacy_penalty = bool(
        expert_unsafe is True
        and expert_node_maladaptive is False
        and ref_maladaptive is False
    )

    if not row.get("expert_safe") or not row.get("expert_node") or not row.get("classifier_node"):
        three_way_node_pattern = "incomplete"
    elif row.get("reference_node") == row.get("expert_node") == row.get("classifier_node"):
        three_way_node_pattern = "all_agree"
    elif row.get("reference_node") == row.get("expert_node"):
        three_way_node_pattern = "reference_expert_agree_classifier_differs"
    elif row.get("reference_node") == row.get("classifier_node"):
        three_way_node_pattern = "reference_classifier_agree_expert_differs"
    elif row.get("expert_node") == row.get("classifier_node"):
        three_way_node_pattern = "expert_classifier_agree_reference_differs"
    else:
        three_way_node_pattern = "all_differ"

    if expert_unsafe is None or ref_maladaptive is None or classifier_maladaptive is None:
        three_way_safety_pattern = "incomplete"
    elif ref_maladaptive == expert_unsafe == classifier_maladaptive:
        three_way_safety_pattern = "all_agree"
    elif ref_maladaptive == expert_unsafe:
        three_way_safety_pattern = "reference_expert_agree_classifier_differs"
    elif ref_maladaptive == classifier_maladaptive:
        three_way_safety_pattern = "reference_classifier_agree_expert_differs"
    elif expert_unsafe == classifier_maladaptive:
        three_way_safety_pattern = "expert_classifier_agree_reference_differs"
    else:
        three_way_safety_pattern = "all_differ"

    if not row.get("expert_safe") or not row.get("expert_node"):
        disagreement_type = "incomplete"
    elif adequacy_penalty:
        disagreement_type = "adequacy_penalty"
    elif ontology_safety_agree is False and expert_node_maladaptive is True and ref_maladaptive is False:
        disagreement_type = "expert_ontology_more_conservative"
    elif ontology_safety_agree is False and expert_node_maladaptive is False and ref_maladaptive is True:
        if expert_unsafe is True:
            disagreement_type = "raw_safety_caught_node_boundary"
        else:
            disagreement_type = "expert_ontology_less_conservative"
    elif ontology_safety_agree is False:
        disagreement_type = "ontology_safety_disagreement"
    elif raw_safety_agree is False:
        disagreement_type = "raw_safety_only_disagreement"
    elif not primary_node_agree:
        disagreement_type = "node_boundary_disagreement"
    else:
        disagreement_type = "full_agreement"

    return {
        **row,
        "reference_is_maladaptive": bool_text(ref_maladaptive),
        "classifier_is_maladaptive": bool_text(classifier_maladaptive),
        "classifier_raw_unsafe": bool_text(classifier_unsafe),
        "raw_expert_unsafe": bool_text(expert_unsafe),
        "expert_node_is_maladaptive": bool_text(expert_node_maladaptive),
        "raw_safety_agreement": bool_text(raw_safety_agree),
        "ontology_safety_agreement": bool_text(ontology_safety_agree),
        "primary_node_agreement": bool_text(primary_node_agree),
        "classifier_reference_safety_agreement": bool_text(classifier_reference_safety_agree),
        "classifier_reference_node_agreement": bool_text(classifier_reference_node_agree),
        "classifier_expert_raw_safety_agreement": bool_text(classifier_expert_raw_safety_agree),
        "classifier_expert_ontology_safety_agreement": bool_text(classifier_expert_ontology_safety_agree),
        "classifier_expert_node_agreement": bool_text(classifier_expert_node_agree),
        "three_way_node_pattern": three_way_node_pattern,
        "three_way_safety_pattern": three_way_safety_pattern,
        "adequacy_penalty": bool_text(adequacy_penalty),
        "disagreement_type": disagreement_type,
    }


def merge_spotcheck(
    export_path: Path,
    key_path: Path,
    blinded_path: Path,
    predictions_path: Path | None = None,
) -> list[dict[str, Any]]:
    exports = {row["case_id"]: row for row in read_export(export_path)}
    key_rows = read_csv(key_path)
    blinded_payload = json.loads(blinded_path.read_text(encoding="utf-8"))
    blinded_rows = {row["case_id"]: row for row in blinded_payload["cases"]}
    predictions_by_input = {}
    if predictions_path:
        predictions_by_input = {row["user_input"]: row for row in read_csv(predictions_path)}

    merged = []
    for key_row in key_rows:
        case_id = key_row["case_id"]
        blinded_row = blinded_rows.get(case_id, {})
        prediction = predictions_by_input.get(blinded_row.get("user_input", ""), {})
        prediction_subset = {
            "classifier_safe": prediction.get("pred_safe", ""),
            "classifier_node": prediction.get("pred_node", ""),
            "classifier_confidence": prediction.get("conf", ""),
            "classifier_severity": prediction.get("severity", ""),
            "classifier_contexts": prediction.get("contexts", ""),
            "classifier_segments": prediction.get("segments", ""),
        }
        row = {
            **key_row,
            **{f"case_{k}": v for k, v in blinded_row.items() if k not in {"order", "case_id"}},
            **prediction_subset,
            **exports.get(case_id, {}),
        }
        merged.append(derive_spotcheck_fields(row))
    return merged


def completed_spotcheck_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("expert_safe") and row.get("expert_node")]


def count_true(rows: list[dict[str, Any]], field: str) -> int:
    return sum(row.get(field) == "true" for row in rows)


def row_slice(rows: list[dict[str, Any]], include_boundary: bool) -> list[dict[str, Any]]:
    if include_boundary:
        return [row for row in rows if row.get("source_bucket") == "boundary"]
    return [row for row in rows if row.get("source_bucket") != "boundary"]


def spotcheck_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = completed_spotcheck_rows(rows)
    raw_pairs = [
        (row.get("raw_expert_unsafe"), row.get("reference_is_maladaptive"))
        for row in complete
    ]
    ontology_pairs = [
        (row.get("expert_node_is_maladaptive"), row.get("reference_is_maladaptive"))
        for row in complete
    ]
    node_pairs = [(row.get("expert_node"), row.get("reference_node")) for row in complete]
    classifier_expert_raw_pairs = [
        (row.get("classifier_raw_unsafe"), row.get("raw_expert_unsafe"))
        for row in complete
    ]
    classifier_expert_ontology_pairs = [
        (row.get("classifier_is_maladaptive"), row.get("expert_node_is_maladaptive"))
        for row in complete
    ]
    classifier_expert_node_pairs = [(row.get("classifier_node"), row.get("expert_node")) for row in complete]
    return {
        "n_total": len(rows),
        "n_complete": len(complete),
        "raw_safety_agree": count_true(complete, "raw_safety_agreement"),
        "ontology_safety_agree": count_true(complete, "ontology_safety_agreement"),
        "primary_node_agree": count_true(complete, "primary_node_agreement"),
        "classifier_reference_safety_agree": count_true(complete, "classifier_reference_safety_agreement"),
        "classifier_reference_node_agree": count_true(complete, "classifier_reference_node_agreement"),
        "classifier_expert_raw_safety_agree": count_true(complete, "classifier_expert_raw_safety_agreement"),
        "classifier_expert_ontology_safety_agree": count_true(complete, "classifier_expert_ontology_safety_agreement"),
        "classifier_expert_node_agree": count_true(complete, "classifier_expert_node_agreement"),
        "adequacy_penalties": count_true(complete, "adequacy_penalty"),
        "raw_safety_kappa": cohen_kappa(raw_pairs),
        "ontology_safety_kappa": cohen_kappa(ontology_pairs),
        "primary_node_kappa": cohen_kappa(node_pairs),
        "classifier_expert_raw_safety_kappa": cohen_kappa(classifier_expert_raw_pairs),
        "classifier_expert_ontology_safety_kappa": cohen_kappa(classifier_expert_ontology_pairs),
        "classifier_expert_node_kappa": cohen_kappa(classifier_expert_node_pairs),
    }


def metric_line(label: str, rows: list[dict[str, Any]]) -> str:
    metrics = spotcheck_metrics(rows)
    n = metrics["n_complete"]
    return (
        f"| {label} | {metrics['n_total']} | {n} | "
        f"{metrics['raw_safety_agree']}/{n} ({pct(metrics['raw_safety_agree'], n)}) | "
        f"{metrics['ontology_safety_agree']}/{n} ({pct(metrics['ontology_safety_agree'], n)}) | "
        f"{metrics['primary_node_agree']}/{n} ({pct(metrics['primary_node_agree'], n)}) | "
        f"{metrics['adequacy_penalties']}/{n} ({pct(metrics['adequacy_penalties'], n)}) | "
        f"{fmt_float(metrics['ontology_safety_kappa'])} |"
    )


def classifier_metric_line(label: str, rows: list[dict[str, Any]]) -> str:
    metrics = spotcheck_metrics(rows)
    n = metrics["n_complete"]
    return (
        f"| {label} | {metrics['n_total']} | {n} | "
        f"{metrics['classifier_reference_safety_agree']}/{n} ({pct(metrics['classifier_reference_safety_agree'], n)}) | "
        f"{metrics['classifier_reference_node_agree']}/{n} ({pct(metrics['classifier_reference_node_agree'], n)}) | "
        f"{metrics['classifier_expert_raw_safety_agree']}/{n} ({pct(metrics['classifier_expert_raw_safety_agree'], n)}) | "
        f"{metrics['classifier_expert_ontology_safety_agree']}/{n} ({pct(metrics['classifier_expert_ontology_safety_agree'], n)}) | "
        f"{metrics['classifier_expert_node_agree']}/{n} ({pct(metrics['classifier_expert_node_agree'], n)}) | "
        f"{fmt_float(metrics['classifier_expert_ontology_safety_kappa'])} |"
    )


def grouped_count_table(rows: list[dict[str, Any]], group_field: str) -> str:
    complete = completed_spotcheck_rows(rows)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in complete:
        groups[row.get(group_field, "")].append(row)

    lines = [
        f"| {group_field} | N | Raw Safety Agree | Node-Derived Safety Agree | Node Agree | Adequacy Penalty |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in sorted(groups):
        group_rows = groups[group]
        n = len(group_rows)
        lines.append(
            f"| {group or 'n/a'} | {n} | "
            f"{count_true(group_rows, 'raw_safety_agreement')} ({pct(count_true(group_rows, 'raw_safety_agreement'), n)}) | "
            f"{count_true(group_rows, 'ontology_safety_agreement')} ({pct(count_true(group_rows, 'ontology_safety_agreement'), n)}) | "
            f"{count_true(group_rows, 'primary_node_agreement')} ({pct(count_true(group_rows, 'primary_node_agreement'), n)}) | "
            f"{count_true(group_rows, 'adequacy_penalty')} ({pct(count_true(group_rows, 'adequacy_penalty'), n)}) |"
        )
    return "\n".join(lines)


def classifier_grouped_count_table(rows: list[dict[str, Any]], group_field: str) -> str:
    complete = completed_spotcheck_rows(rows)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in complete:
        groups[row.get(group_field, "")].append(row)

    lines = [
        f"| {group_field} | N | Classifier-Reference Safety | Classifier-Reference Node | Classifier-Expert Raw Safety | Classifier-Expert Node-Derived Safety | Classifier-Expert Node |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in sorted(groups):
        group_rows = groups[group]
        n = len(group_rows)
        lines.append(
            f"| {group or 'n/a'} | {n} | "
            f"{count_true(group_rows, 'classifier_reference_safety_agreement')} ({pct(count_true(group_rows, 'classifier_reference_safety_agreement'), n)}) | "
            f"{count_true(group_rows, 'classifier_reference_node_agreement')} ({pct(count_true(group_rows, 'classifier_reference_node_agreement'), n)}) | "
            f"{count_true(group_rows, 'classifier_expert_raw_safety_agreement')} ({pct(count_true(group_rows, 'classifier_expert_raw_safety_agreement'), n)}) | "
            f"{count_true(group_rows, 'classifier_expert_ontology_safety_agreement')} ({pct(count_true(group_rows, 'classifier_expert_ontology_safety_agreement'), n)}) | "
            f"{count_true(group_rows, 'classifier_expert_node_agreement')} ({pct(count_true(group_rows, 'classifier_expert_node_agreement'), n)}) |"
        )
    return "\n".join(lines)


def confidence_count_table(rows: list[dict[str, Any]]) -> str:
    complete = completed_spotcheck_rows(rows)
    confidence_order = ["high", "medium", "low", ""]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in complete:
        groups[row.get("expert_confidence", "")].append(row)

    lines = [
        "| Expert Confidence | N | Raw Safety Agree | Node-Derived Safety Agree | Node Agree | Adequacy Penalty |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for confidence in confidence_order:
        group_rows = groups.get(confidence, [])
        if not group_rows:
            continue
        label = confidence or "blank"
        n = len(group_rows)
        lines.append(
            f"| {label} | {n} | "
            f"{count_true(group_rows, 'raw_safety_agreement')} ({pct(count_true(group_rows, 'raw_safety_agreement'), n)}) | "
            f"{count_true(group_rows, 'ontology_safety_agreement')} ({pct(count_true(group_rows, 'ontology_safety_agreement'), n)}) | "
            f"{count_true(group_rows, 'primary_node_agreement')} ({pct(count_true(group_rows, 'primary_node_agreement'), n)}) | "
            f"{count_true(group_rows, 'adequacy_penalty')} ({pct(count_true(group_rows, 'adequacy_penalty'), n)}) |"
        )
    return "\n".join(lines)


def pattern_table(rows: list[dict[str, Any]], field: str) -> str:
    complete = completed_spotcheck_rows(rows)
    counts = Counter(row.get(field, "") for row in complete)
    lines = [f"| {field} | Count | Share |", "|---|---:|---:|"]
    for key, value in counts.most_common():
        lines.append(f"| {key or 'n/a'} | {value} | {pct(value, len(complete))} |")
    return "\n".join(lines)


def disagreement_table(rows: list[dict[str, Any]]) -> str:
    complete = completed_spotcheck_rows(rows)
    counts = Counter(row["disagreement_type"] for row in complete)
    lines = ["| Disagreement Type | Count | Share |", "|---|---:|---:|"]
    for key, value in counts.most_common():
        lines.append(f"| {key} | {value} | {pct(value, len(complete))} |")
    return "\n".join(lines)


def render_spotcheck_summary(
    rows: list[dict[str, Any]],
    export_path: Path,
    rater_id: str = "expert",
    rater_name: str | None = None,
) -> str:
    rater_name = rater_name or rater_id
    rater_label = rater_name if rater_name == rater_id else f"{rater_name} ({rater_id})"
    complete = completed_spotcheck_rows(rows)
    non_boundary = row_slice(rows, include_boundary=False)
    boundary = row_slice(rows, include_boundary=True)
    metrics = spotcheck_metrics(rows)
    md = [
        f"# {rater_id} Expert Spotcheck Analysis",
        "",
        f"Source export: `{export_path}`",
        "",
        "## Headline Slices",
        "",
        "| Slice | Rows | Complete | Raw Safety Agreement | Node-Derived Safety Agreement | Strict Node Agreement | Adequacy Penalties | Node-Derived Kappa |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        metric_line("All completed prefix rows", rows),
        metric_line("Primary non-boundary rows", non_boundary),
        metric_line("Boundary rows", boundary),
        "",
        "Raw safety agreement uses `expert_safe`. Node-derived safety agreement uses the selected expert node only.",
        f"Adequacy penalties are non-maladaptive reference rows where {rater_id} selected a neutral/adaptive node but marked the case unsafe/not safe.",
        "",
        "## Classifier Agreement with Reference and Expert",
        "",
        "| Slice | Rows | Complete | Classifier-Reference Safety | Classifier-Reference Node | Classifier-Expert Raw Safety | Classifier-Expert Node-Derived Safety | Classifier-Expert Strict Node | Classifier-Expert Node-Derived Kappa |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        classifier_metric_line("All completed prefix rows", rows),
        classifier_metric_line("Primary non-boundary rows", non_boundary),
        classifier_metric_line("Boundary rows", boundary),
        "",
        f"Classifier-expert raw safety compares the classifier's binary safety verdict with {rater_id}'s `expert_safe` field.",
        f"Classifier-expert node-derived safety compares the classifier's predicted node type with the type of {rater_id}'s selected node.",
        "",
        "## Three-Way Pattern: Reference, Expert, Classifier",
        "",
        f"Safety pattern uses {rater_id}'s raw safety field. Node pattern uses strict primary-node labels.",
        "",
        "### Safety Pattern",
        "",
        pattern_table(rows, "three_way_safety_pattern"),
        "",
        "### Node Pattern",
        "",
        pattern_table(rows, "three_way_node_pattern"),
        "",
        "## Agreement by Source Bucket",
        "",
        grouped_count_table(rows, "source_bucket"),
        "",
        "## Classifier Agreement by Source Bucket",
        "",
        classifier_grouped_count_table(rows, "source_bucket"),
        "",
        "## Confidence Sensitivity",
        "",
        f"Primary non-boundary rows stratified by {rater_id}'s confidence rating:",
        "",
        confidence_count_table(non_boundary),
        "",
        "## Agreement by Reference Node",
        "",
        grouped_count_table(rows, "reference_node"),
        "",
        "## Disagreement Taxonomy",
        "",
        disagreement_table(rows),
        "",
        "## Thesis Interpretation",
        "",
        (
            f"{rater_label} completed {len(complete)}/{len(rows)} spotcheck rows. "
            f"Raw safety agreement was {metrics['raw_safety_agree']}/{metrics['n_complete']} "
            f"({pct(metrics['raw_safety_agree'], metrics['n_complete'])}), while node-derived safety agreement was "
            f"{metrics['ontology_safety_agree']}/{metrics['n_complete']} "
            f"({pct(metrics['ontology_safety_agree'], metrics['n_complete'])}). "
            "This distinction is reported because the raw safety field sometimes captures clinical adequacy "
            "rather than presence of a maladaptive ontology node."
        ),
        "",
    ]
    return "\n".join(md)


def note_has_adequacy_signal(note: Any) -> bool:
    note_text = as_lower(note).replace("’", "'").replace("´", "'")
    return any(pattern.replace("´", "'") in note_text for pattern in ADEQUACY_NOTE_PATTERNS)


def score_value(value: Any, mapping: dict[str, int]) -> int | None:
    text = as_lower(value)
    if text in mapping:
        return mapping[text]
    return None


def derive_system_fields(row: dict[str, Any]) -> dict[str, Any]:
    expected_maladaptive = node_is_maladaptive(row.get("expected_node"))
    detected_maladaptive = node_is_maladaptive(row.get("detected_node"))
    clinical_score = score_value(row.get("expert_clinically_acceptable"), CLINICAL_ACCEPTABILITY_SCORE)
    redirect_score = score_value(row.get("expert_redirect_quality"), REDIRECT_QUALITY_SCORE)
    validation_score = score_value(row.get("expert_validates_distress_appropriately"), VALIDATION_SCORE)
    endorsement_score = score_value(row.get("expert_maladaptive_endorsement"), MALADAPTIVE_ENDORSEMENT_SCORE)
    adequacy_note = note_has_adequacy_signal(row.get("expert_notes"))
    false_positive_note = "false positive" in as_lower(row.get("expert_notes"))

    return {
        **row,
        "expected_is_maladaptive": bool_text(expected_maladaptive),
        "detected_is_maladaptive": bool_text(detected_maladaptive),
        "clinical_acceptability_score": "" if clinical_score is None else clinical_score,
        "redirect_quality_score": "" if redirect_score is None else redirect_score,
        "validation_score": "" if validation_score is None else validation_score,
        "maladaptive_endorsement_score": "" if endorsement_score is None else endorsement_score,
        "adequacy_note_signal": bool_text(adequacy_note),
        "false_positive_note_signal": bool_text(false_positive_note),
    }


def merge_system_eval(
    export_path: Path,
    key_path: Path,
    source_results_path: Path,
) -> list[dict[str, Any]]:
    exports = {str(row["shuffled_row_id"]): row for row in read_export(export_path)}
    key_rows = read_csv(key_path)
    source_rows = {
        (row["case_id"], row["condition"]): row
        for row in read_csv(source_results_path)
    }

    merged = []
    for key_row in key_rows:
        source_row = source_rows.get((key_row["case_id"], key_row["condition"]), {})
        source_subset = {
            "nightmare": source_row.get("nightmare", ""),
            "user_input": source_row.get("user_input", ""),
            "response": source_row.get("response", ""),
            "critic_classified_unsafe": source_row.get("critic_classified_unsafe", ""),
            "severity": source_row.get("severity", ""),
            "constraint": source_row.get("constraint", ""),
        }
        row = {**key_row, **source_subset, **exports.get(key_row["shuffled_row_id"], {})}
        merged.append(derive_system_fields(row))
    return merged


def completed_system_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("expert_clinically_acceptable")]


def average_score(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [int(row[field]) for row in rows if str(row.get(field, "")).strip() != ""]
    if not values:
        return None
    return sum(values) / len(values)


def score_distribution(rows: list[dict[str, Any]], field: str) -> str:
    counts = Counter(row.get(field, "") for row in rows if row.get(field, "") != "")
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) or "n/a"


def condition_summary_table(rows: list[dict[str, Any]]) -> str:
    complete = completed_system_rows(rows)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in complete:
        by_condition[row["condition"]].append(row)

    lines = [
        "| Condition | N | Clinically Acceptable | Mean Clinical Score | Redirect Quality | Mean Redirect Score | Maladaptive Endorsement | Validation |",
        "|---|---:|---|---:|---|---:|---|---|",
    ]
    for condition in sorted(by_condition):
        group = by_condition[condition]
        lines.append(
            f"| {condition} | {len(group)} | "
            f"{score_distribution(group, 'expert_clinically_acceptable')} | "
            f"{fmt_float(average_score(group, 'clinical_acceptability_score'))} | "
            f"{score_distribution(group, 'expert_redirect_quality')} | "
            f"{fmt_float(average_score(group, 'redirect_quality_score'))} | "
            f"{score_distribution(group, 'expert_maladaptive_endorsement')} | "
            f"{score_distribution(group, 'expert_validates_distress_appropriately')} |"
        )
    return "\n".join(lines)


def system_node_table(rows: list[dict[str, Any]]) -> str:
    complete = completed_system_rows(rows)
    by_node_condition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in complete:
        by_node_condition[(row["expected_node"], row["condition"])].append(row)

    lines = [
        "| Expected Node | Condition | N | Clinical Acceptability | Redirect Quality | Maladaptive Endorsement |",
        "|---|---|---:|---|---|---|",
    ]
    for node, condition in sorted(by_node_condition):
        group = by_node_condition[(node, condition)]
        lines.append(
            f"| {node} | {condition} | {len(group)} | "
            f"{score_distribution(group, 'expert_clinically_acceptable')} | "
            f"{score_distribution(group, 'expert_redirect_quality')} | "
            f"{score_distribution(group, 'expert_maladaptive_endorsement')} |"
        )
    return "\n".join(lines)


def build_paired_system_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in completed_system_rows(rows):
        by_case[row["case_id"]][row["condition"]] = row

    paired = []
    score_fields = [
        "clinical_acceptability_score",
        "redirect_quality_score",
        "validation_score",
        "maladaptive_endorsement_score",
    ]
    for case_id in sorted(by_case):
        pair = by_case[case_id]
        if "control" not in pair or "treatment" not in pair:
            continue
        control = pair["control"]
        treatment = pair["treatment"]
        row = {
            "case_id": case_id,
            "tier": control.get("tier", treatment.get("tier", "")),
            "expected_node": control.get("expected_node", treatment.get("expected_node", "")),
            "detected_node": control.get("detected_node", treatment.get("detected_node", "")),
            "control_row_id": control.get("shuffled_row_id", ""),
            "treatment_row_id": treatment.get("shuffled_row_id", ""),
            "control_clinically_acceptable": control.get("expert_clinically_acceptable", ""),
            "treatment_clinically_acceptable": treatment.get("expert_clinically_acceptable", ""),
            "control_redirect_quality": control.get("expert_redirect_quality", ""),
            "treatment_redirect_quality": treatment.get("expert_redirect_quality", ""),
            "control_maladaptive_endorsement": control.get("expert_maladaptive_endorsement", ""),
            "treatment_maladaptive_endorsement": treatment.get("expert_maladaptive_endorsement", ""),
            "control_validation": control.get("expert_validates_distress_appropriately", ""),
            "treatment_validation": treatment.get("expert_validates_distress_appropriately", ""),
            "control_notes": control.get("expert_notes", ""),
            "treatment_notes": treatment.get("expert_notes", ""),
        }
        for field in score_fields:
            c_value = control.get(field, "")
            t_value = treatment.get(field, "")
            row[f"control_{field}"] = c_value
            row[f"treatment_{field}"] = t_value
            if str(c_value) != "" and str(t_value) != "":
                row[f"delta_{field}"] = int(t_value) - int(c_value)
            else:
                row[f"delta_{field}"] = ""
        paired.append(row)
    return paired


def paired_delta_table(paired_rows: list[dict[str, Any]]) -> str:
    fields = [
        ("Clinical Acceptability", "delta_clinical_acceptability_score", "higher is better"),
        ("Redirect Quality", "delta_redirect_quality_score", "higher is better"),
        ("Validation", "delta_validation_score", "higher is better"),
        ("Maladaptive Endorsement", "delta_maladaptive_endorsement_score", "lower is better"),
    ]
    lines = ["| Metric | Direction | Better | Same | Worse | Mean Delta |", "|---|---|---:|---:|---:|---:|"]
    for label, field, direction in fields:
        values = [int(row[field]) for row in paired_rows if str(row.get(field, "")) != ""]
        if not values:
            lines.append(f"| {label} | {direction} | 0 | 0 | 0 | n/a |")
            continue
        if "lower" in direction:
            better = sum(value < 0 for value in values)
            worse = sum(value > 0 for value in values)
        else:
            better = sum(value > 0 for value in values)
            worse = sum(value < 0 for value in values)
        same = sum(value == 0 for value in values)
        mean_delta = sum(values) / len(values)
        lines.append(f"| {label} | {direction} | {better} | {same} | {worse} | {mean_delta:.2f} |")
    return "\n".join(lines)


VERDICT_TIER_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("Correct-verdict tiers (main_unsafe, safe_regression, multipart)", ("main_unsafe", "safe_regression", "multipart")),
    ("Error tier (robustness)", ("robustness",)),
]


def paired_delta_table_by_tier(paired_rows: list[dict[str, Any]]) -> str:
    """Paired deltas split by whether the safety layer's verdict was correct.

    The pooled table mixes the correct-verdict tiers with the error tier, whose
    share of wrong verdicts is far above the held-out error rate. The split
    shows whether the pooled result depends on the error tier.
    """
    known = {tier for _, tiers in VERDICT_TIER_GROUPS for tier in tiers}
    lines = []
    for label, tiers in VERDICT_TIER_GROUPS:
        subset = [row for row in paired_rows if row.get("tier", "") in tiers]
        lines.append(f"### {label}: {len(subset)} pairs")
        lines.append("")
        lines.append(paired_delta_table(subset))
        lines.append("")
    unassigned = sorted({row.get("tier", "") for row in paired_rows} - known)
    if unassigned:
        lines.append("Unassigned tiers (in neither group): " + ", ".join(repr(t) for t in unassigned))
        lines.append("")
    return "\n".join(lines).rstrip()


def render_system_summary(
    rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    export_path: Path,
    rater_id: str = "expert",
    rater_name: str | None = None,
) -> str:
    rater_name = rater_name or rater_id
    complete = completed_system_rows(rows)
    notes = [row for row in complete if row.get("expert_notes")]
    adequacy_notes = [row for row in complete if row.get("adequacy_note_signal") == "true"]
    false_positive_notes = [row for row in complete if row.get("false_positive_note_signal") == "true"]
    md = [
        f"# {rater_id} System-Level Expert Evaluation",
        "",
        f"Source export: `{export_path}`",
        "",
        f"Completed rows: {len(complete)}/{len(rows)}",
        f"Paired case rows: {len(paired_rows)}",
        f"Rows with notes: {len(notes)}",
        f"Rows with adequacy-note signal: {len(adequacy_notes)}",
        f"Rows with false-positive-note signal: {len(false_positive_notes)}",
        "",
        "## Condition Summary",
        "",
        condition_summary_table(rows),
        "",
        "## Paired Treatment-Control Deltas",
        "",
        paired_delta_table(paired_rows),
        "",
        "## Paired Deltas by Verdict Tier",
        "",
        paired_delta_table_by_tier(paired_rows),
        "",
        "## By Expected Node",
        "",
        system_node_table(rows),
        "",
        "## Interpretation Note",
        "",
        (
            "The paired deltas use ordinal scores only as descriptive summaries. "
            f"They should be interpreted alongside the free-text notes, especially when {rater_name} flagged "
            "responses as false-positive redirects because the user's original attempt was already adaptive."
        ),
        "",
    ]
    return "\n".join(md)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spotcheck-export",
        type=Path,
        default=REPO_ROOT / "data/expert_analysis/kl_2026_06/raw_exports/redream_spotcheck_KL_2026-06-25T10-45-11-923Z.csv",
    )
    parser.add_argument(
        "--system-export",
        type=Path,
        default=REPO_ROOT / "data/expert_analysis/kl_2026_06/raw_exports/redream_system_eval_KL_2026-06-29T11-11-01-082Z.csv",
    )
    parser.add_argument("--spotcheck-key", type=Path, default=REPO_ROOT / "data/expert_spotcheck/spotcheck_key.csv")
    parser.add_argument(
        "--spotcheck-blinded",
        type=Path,
        default=REPO_ROOT / "data/expert_spotcheck/spotcheck_cases_blinded.json",
    )
    parser.add_argument(
        "--spotcheck-predictions",
        type=Path,
        default=REPO_ROOT / "data/benchmarks/test_s1_baseline.csv",
    )
    parser.add_argument("--system-key", type=Path, default=REPO_ROOT / "data/system_eval/system_eval_rater_key.csv")
    parser.add_argument("--system-source", type=Path, default=REPO_ROOT / "data/system_eval/system_eval_results.csv")
    parser.add_argument("--rater-id", default="KL", help="Short rater identifier used in generated reports")
    parser.add_argument(
        "--rater-name",
        default=None,
        help="Rater label used in generated prose (default: the rater id, i.e. initials only)",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/expert_analysis/kl_2026_06")
    return parser.parse_args()


def run_analysis(args: argparse.Namespace) -> dict[str, Path]:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    spotcheck_rows = merge_spotcheck(
        args.spotcheck_export,
        args.spotcheck_key,
        args.spotcheck_blinded,
        args.spotcheck_predictions,
    )
    system_rows = merge_system_eval(args.system_export, args.system_key, args.system_source)
    paired_rows = build_paired_system_rows(system_rows)

    spotcheck_disagreements = [
        row for row in completed_spotcheck_rows(spotcheck_rows)
        if row.get("disagreement_type") != "full_agreement"
    ]
    spotcheck_classifier_disagreements = [
        row for row in completed_spotcheck_rows(spotcheck_rows)
        if row.get("classifier_expert_raw_safety_agreement") == "false"
        or row.get("classifier_expert_ontology_safety_agreement") == "false"
        or row.get("classifier_expert_node_agreement") == "false"
    ]
    spotcheck_boundary = [row for row in spotcheck_rows if row.get("source_bucket") == "boundary"]
    system_notes = [
        row for row in completed_system_rows(system_rows)
        if row.get("expert_notes") or row.get("adequacy_note_signal") == "true" or row.get("false_positive_note_signal") == "true"
    ]

    paths = {
        "spotcheck_merged": out_dir / "spotcheck_merged.csv",
        "spotcheck_disagreements": out_dir / "spotcheck_disagreements.csv",
        "spotcheck_classifier_disagreements": out_dir / "spotcheck_classifier_disagreements.csv",
        "spotcheck_boundary_cases": out_dir / "spotcheck_boundary_cases.csv",
        "spotcheck_summary": out_dir / "spotcheck_summary.md",
        "system_eval_merged": out_dir / "system_eval_merged.csv",
        "system_eval_paired_by_case": out_dir / "system_eval_paired_by_case.csv",
        "system_eval_note_flags": out_dir / "system_eval_note_flags.csv",
        "system_eval_summary": out_dir / "system_eval_summary.md",
    }

    write_csv(paths["spotcheck_merged"], spotcheck_rows)
    write_csv(paths["spotcheck_disagreements"], spotcheck_disagreements)
    write_csv(paths["spotcheck_classifier_disagreements"], spotcheck_classifier_disagreements)
    write_csv(paths["spotcheck_boundary_cases"], spotcheck_boundary)
    write_text(
        paths["spotcheck_summary"],
        render_spotcheck_summary(
            spotcheck_rows,
            args.spotcheck_export,
            rater_id=args.rater_id,
            rater_name=args.rater_name,
        ),
    )

    write_csv(paths["system_eval_merged"], system_rows)
    write_csv(paths["system_eval_paired_by_case"], paired_rows)
    write_csv(paths["system_eval_note_flags"], system_notes)
    write_text(
        paths["system_eval_summary"],
        render_system_summary(
            system_rows,
            paired_rows,
            args.system_export,
            rater_id=args.rater_id,
            rater_name=args.rater_name,
        ),
    )
    return paths


def main() -> int:
    args = parse_args()
    paths = run_analysis(args)
    print(f"Wrote expert analysis outputs to {args.output_dir}")
    for label, path in paths.items():
        print(f"- {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
