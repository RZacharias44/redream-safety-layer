"""Reproduce every quantitative uncertainty result reported in Chapter 4.7.

Run from the repository root:

    uv run python experiments/evaluation_statistics/chapter_4/section_4_7_uncertainty_quantification.py

AUROC is calculated directly from pairwise rankings (ties count as one half),
which keeps the implementation inspectable and independent of ML libraries.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from experiments.evaluation_statistics.consistency_votes import derive_vote_verdicts


ROOT = Path(__file__).resolve().parents[3]
FILES = {
    "S1 baseline": ROOT / "data/benchmarks/dev_s1_baseline.csv",
    "S1 consistency": ROOT / "data/benchmarks/dev_s1_consistency_n5.csv",
    "S2 baseline": ROOT / "data/benchmarks/dev_s2_baseline.csv",
    "S2 consistency": ROOT / "data/benchmarks/dev_s2_consistency_n5.csv",
    # S2 baseline re-run after the classification-token extraction fix. The `conf`
    # column of the two S2 runs above came from a defective extractor that fell back
    # to the mean log probability of the whole JSON response on 87.1% of calls, so
    # those values measure rationale fluency rather than label certainty
    # (docs/planning/S2_logprob_diagnostic_findings.md). They are kept here only so
    # the superseded figures remain reproducible. Classification outcomes are not
    # affected by the fix: nothing gates on confidence at threshold 0.0.
    "S2 baseline corrected": ROOT / "data/benchmarks/dev_s2_baseline_rerun_corrected_logprobs.csv",
}
KL_FILE = ROOT / "data/expert_analysis/kl_2026_06/spotcheck_merged.csv"

EXPERT_DATA_NOTICE = (
    "The row-level expert annotation tables under data/expert_analysis/ are not "
    "present: they are personal data of the raters and are distributed only with "
    "their consent. The aggregate results are in "
    "data/evaluation_statistics/chapter4/chapter4_statistics.json ('uncertainty.kl_spotcheck_confidence')."
)
# Held-out System 1 execution on the final test set (section 4.3). Rows with a
# blank `gt_safe` are boundary cases and are not scored; the truncated
# buried-alive suite is excluded from the final held-out set.
HELDOUT_FILE = ROOT / "data/benchmarks/test_s1_baseline.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def as_bool(value: str) -> bool | None:
    text = value.strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def scored(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if as_bool(row["gt_safe"]) is not None]


def pairwise_auc(labels: list[bool], scores: list[float]) -> float:
    """Probability that a correct case scores above an incorrect case."""
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    assert positives and negatives
    ordered = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return ordered / (len(positives) * len(negatives))


def uncertainty_metrics(
    rows: list[dict[str, str]],
    signal: str,
    pred_safe_key: str = "pred_safe",
    pred_node_key: str = "pred_node",
) -> dict[str, int | float]:
    """AUROC of one signal against the correctness of one prediction.

    The prediction keys select which decision the signal is judged on. The
    default is the deterministic pipeline prediction, i.e. the signal treated as
    a flag on the label the system would deploy. Passing the vote columns
    instead judges the signal on the majority-vote verdict that Chapter 4.4
    reports for the consistency conditions. Both are reported because they
    answer different questions and the answers differ for System 2.
    """
    scored_rows = scored(rows)
    usable = [row for row in scored_rows if row.get(signal, "").strip()]
    scores = [float(row[signal]) for row in usable]
    safety_correct = [
        as_bool(row["gt_safe"]) == as_bool(row[pred_safe_key]) for row in usable
    ]
    node_correct = [row["gt_node"] == row[pred_node_key] for row in usable]
    return {
        "n_scored": len(scored_rows),
        "n_usable": len(usable),
        "n_missing": len(scored_rows) - len(usable),
        "safety_errors": sum(not value for value in safety_correct),
        "node_errors": sum(not value for value in node_correct),
        "safety_auroc": pairwise_auc(safety_correct, scores),
        "node_auroc": pairwise_auc(node_correct, scores),
    }


def threshold_metrics(
    rows: list[dict[str, str]], signal: str, threshold: float
) -> dict[str, int | float]:
    usable = [row for row in scored(rows) if row.get(signal, "").strip()]
    flagged = [row for row in usable if float(row[signal]) < threshold]
    errors = [
        row
        for row in usable
        if as_bool(row["gt_safe"]) != as_bool(row["pred_safe"])
    ]
    caught = [row for row in flagged if row in errors]
    return {
        "threshold": threshold,
        "flagged": len(flagged),
        "safety_errors": len(errors),
        "caught": len(caught),
        "correct_flagged": len(flagged) - len(caught),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def signal_spread(rows: list[dict[str, str]], signal: str) -> dict[str, float | int]:
    """Descriptive spread of a signal, and its separation of correct from incorrect.

    Reported because AUROC is scale-free: a signal can rank cases while occupying
    a range too narrow for any defensible operating threshold.
    """
    usable = [row for row in scored(rows) if row.get(signal, "").strip()]
    values = [float(row[signal]) for row in usable]
    node_correct = [row["gt_node"] == row["pred_node"] for row in usable]
    correct = [value for value, ok in zip(values, node_correct) if ok]
    incorrect = [value for value, ok in zip(values, node_correct) if not ok]
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "node_correct_mean": mean(correct),
        "node_incorrect_mean": mean(incorrect),
    }


def kl_model_confidence(rows: list[dict[str, str]]) -> dict[str, Any]:
    # Boundary rows carry the author's internal-review annotation in
    # `reference_node` (build_spotcheck_app.py), not a benchmark reference
    # label; thesis section 4.6.2 states they have no reference node, so they
    # are excluded here as in every other reference-agreement figure.
    complete_all = [
        row
        for row in rows
        if as_bool(row["expert_safe"]) is not None
        and row["expert_node"].strip()
        and row["classifier_confidence"].strip()
    ]
    complete = [row for row in complete_all if row.get("source_bucket") != "boundary"]

    def confidence(group: list[dict[str, str]]) -> float:
        return mean([float(row["classifier_confidence"]) for row in group])

    classifier_correct = [
        row for row in complete if row["classifier_node"] == row["reference_node"]
    ]
    classifier_wrong = [row for row in complete if row not in classifier_correct]
    expert_correct = [
        row for row in complete if row["expert_node"] == row["reference_node"]
    ]
    expert_wrong = [row for row in complete if row not in expert_correct]

    result: dict[str, Any] = {
        "n": len(complete),
        "n_boundary_excluded": len(complete_all) - len(complete),
        "classifier_wrong_mean": confidence(classifier_wrong),
        "classifier_correct_mean": confidence(classifier_correct),
        "expert_wrong_mean": confidence(expert_wrong),
        "expert_correct_mean": confidence(expert_correct),
    }
    for cutoff in (0.90, 0.95):
        flagged = [
            row for row in complete if float(row["classifier_confidence"]) < cutoff
        ]
        classifier_disagreement = [
            row for row in flagged if row["classifier_node"] != row["reference_node"]
        ]
        expert_disagreement = [
            row for row in flagged if row["expert_node"] != row["reference_node"]
        ]
        both = [
            row
            for row in flagged
            if row in classifier_disagreement and row in expert_disagreement
        ]
        result[f"threshold_{cutoff:.2f}"] = {
            "flagged": len(flagged),
            "classifier_reference_disagreements": len(classifier_disagreement),
            "expert_reference_disagreements": len(expert_disagreement),
            "both": len(both),
        }
    return result


def heldout_confidence_by_node_correctness(rows: list[dict[str, str]]) -> dict[str, Any]:
    scored = [
        row
        for row in rows
        if as_bool(row["gt_safe"]) is not None
        and row["gt_node"].strip()
        and "buried alive" not in row["suite"].lower()
    ]
    with_conf = [row for row in scored if row["conf"].strip()]
    correct = [row for row in with_conf if row["pred_node"] == row["gt_node"]]
    wrong = [row for row in with_conf if row["pred_node"] != row["gt_node"]]

    def confidence(group: list[dict[str, str]]) -> float:
        return mean([float(row["conf"]) for row in group])

    return {
        "n_scored": len(scored),
        "n_with_conf": len(with_conf),
        "node_correct_n": len(correct),
        "node_correct_mean": confidence(correct),
        "node_wrong_n": len(wrong),
        "node_wrong_mean": confidence(wrong),
    }


def calculate_all() -> dict[str, Any]:
    rows = {name: read_rows(path) for name, path in FILES.items()}
    # The consistency rows also carry the vote-derived verdict, so each signal
    # can be scored either as a flag on the deterministic prediction or on the
    # majority vote that Chapter 4.4 reports.
    voted = {
        name: derive_vote_verdicts(rows[name])[0]
        for name in ("S1 consistency", "S2 consistency")
    }
    return {
        "rows": rows,
        # Log-probability confidence is reported for the baseline conditions,
        # where it accompanies the emitted label. Agreement is reported for the
        # consistency conditions, scored on the majority-vote verdict they
        # report; the temperature-0 call inside those runs only records log
        # probabilities and is not used as a scoring target.
        "signals": {
            "S1 baseline log-probability": uncertainty_metrics(rows["S1 baseline"], "conf"),
            "S2 baseline log-probability": uncertainty_metrics(rows["S2 baseline"], "conf"),
            "S2 baseline corrected log-probability": uncertainty_metrics(
                rows["S2 baseline corrected"], "conf"
            ),
            "S1 consistency agreement": uncertainty_metrics(
                voted["S1 consistency"], "agreement_ratio", "vote_safe", "vote_node"
            ),
            "S2 consistency agreement": uncertainty_metrics(
                voted["S2 consistency"], "agreement_ratio", "vote_safe", "vote_node"
            ),
        },
        "thresholds": {
            "S1 0.90": threshold_metrics(rows["S1 baseline"], "conf", 0.90),
            "S1 0.95": threshold_metrics(rows["S1 baseline"], "conf", 0.95),
            "S2 0.80": threshold_metrics(rows["S2 baseline"], "conf", 0.80),
            # The corrected S2 signal is compressed into the last decimals, so a
            # cutoff on the same scale as System 1 flags nothing at all.
            "S2 corrected 0.90": threshold_metrics(
                rows["S2 baseline corrected"], "conf", 0.90
            ),
            "S2 corrected 0.9999": threshold_metrics(
                rows["S2 baseline corrected"], "conf", 0.9999
            ),
        },
        "s2_corrected_spread": signal_spread(rows["S2 baseline corrected"], "conf"),
        "s2_superseded_spread": signal_spread(rows["S2 baseline"], "conf"),
        "kl_model_confidence": kl_model_confidence(read_rows(KL_FILE)) if KL_FILE.exists() else None,
        "heldout_model_confidence": heldout_confidence_by_node_correctness(read_rows(HELDOUT_FILE)),
    }


def main() -> None:
    results = calculate_all()
    print("Chapter 4.7 -- Uncertainty Quantification Results")
    print("\n1. AUROC comparison")
    print("condition / signal | usable | missing | safety AUROC | node AUROC")
    for name, metrics in results["signals"].items():
        print(
            f"{name} | {metrics['n_usable']} | {metrics['n_missing']} | "
            f"{metrics['safety_auroc']:.2f} | {metrics['node_auroc']:.2f}"
        )


    print("\n2. Confidence-threshold review burden")
    for name, metrics in results["thresholds"].items():
        print(
            f"{name}: {metrics['flagged']} flagged; {metrics['caught']}/"
            f"{metrics['safety_errors']} safety errors caught; "
            f"{metrics['correct_flagged']} correct cases flagged"
        )

    print("\n2b. System 2 confidence spread, superseded vs corrected extraction")
    for label, key in (
        ("superseded", "s2_superseded_spread"),
        ("corrected", "s2_corrected_spread"),
    ):
        spread = results[key]
        print(
            f"{label}: n={spread['n']} range [{spread['min']:.4f}, {spread['max']:.4f}] "
            f"mean {spread['mean']:.4f}; node-correct {spread['node_correct_mean']:.6f} "
            f"vs node-incorrect {spread['node_incorrect_mean']:.6f}"
        )

    kl = results["kl_model_confidence"]
    if kl is None:
        print("\n3. Classifier confidence on KL's scored spot check: skipped")
        print(EXPERT_DATA_NOTICE)
        return
    print("\n3. Classifier confidence on KL's scored spot check")
    print(
        f"classifier wrong/correct reference node means: "
        f"{kl['classifier_wrong_mean']:.3f} / {kl['classifier_correct_mean']:.3f}"
    )
    print(
        f"KL wrong/correct reference node means: "
        f"{kl['expert_wrong_mean']:.3f} / {kl['expert_correct_mean']:.3f}"
    )
    for cutoff in (0.90, 0.95):
        metrics = kl[f"threshold_{cutoff:.2f}"]
        print(
            f"below {cutoff:.2f}: {metrics['flagged']} rows; "
            f"classifier/reference {metrics['classifier_reference_disagreements']}; "
            f"expert/reference {metrics['expert_reference_disagreements']}; "
            f"both {metrics['both']}"
        )


if __name__ == "__main__":
    main()
