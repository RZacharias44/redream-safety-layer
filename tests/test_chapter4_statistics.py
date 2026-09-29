import csv

from experiments.evaluation_statistics.recompute_chapter4 import (
    benchmark_metrics,
    expert_inputs_available,
    rank_auc,
    recompute,
)
from experiments.evaluation_statistics.verify_chapter4_statistics import (
    benchmark_counts as independent_benchmark_counts,
    cohen_kappa as independent_cohen_kappa,
    rank_auc as independent_rank_auc,
    verify,
)


def test_benchmark_metrics_separate_false_alarms_and_missed_unsafe():
    rows = [
        {"gt_safe": "True", "pred_safe": "True", "gt_node": "SOCIAL_MASTERY", "pred_node": "SOCIAL_MASTERY"},
        {"gt_safe": "True", "pred_safe": "False", "gt_node": "SOCIAL_MASTERY", "pred_node": "AVOIDANCE"},
        {"gt_safe": "False", "pred_safe": "False", "gt_node": "AVOIDANCE", "pred_node": "AVOIDANCE"},
        {"gt_safe": "False", "pred_safe": "True", "gt_node": "AVOIDANCE", "pred_node": "BEHAVIORAL_MASTERY"},
        {"gt_safe": "", "pred_safe": "True", "gt_node": "", "pred_node": "BEHAVIORAL_MASTERY"},
    ]

    metrics = benchmark_metrics(rows)

    assert metrics["n_scored"] == 4
    assert metrics["safety_correct"] == 2
    assert metrics["unsafe_false_positive"] == 1
    assert metrics["unsafe_false_negative"] == 1
    assert metrics["node_correct"] == 2


def test_rank_auc_handles_ties():
    assert rank_auc([False, True], [0.1, 0.9]) == 1.0
    assert rank_auc([False, True], [0.5, 0.5]) == 0.5


def test_independent_formulas_are_explicit_and_handle_ties():
    rows = [
        {"gt_safe": "False", "pred_safe": "False", "gt_node": "AVOIDANCE", "pred_node": "AVOIDANCE"},
        {"gt_safe": "False", "pred_safe": "True", "gt_node": "AVOIDANCE", "pred_node": "SOCIAL_MASTERY"},
        {"gt_safe": "True", "pred_safe": "False", "gt_node": "SOCIAL_MASTERY", "pred_node": "AVOIDANCE"},
        {"gt_safe": "True", "pred_safe": "True", "gt_node": "SOCIAL_MASTERY", "pred_node": "SOCIAL_MASTERY"},
        {"gt_safe": "", "pred_safe": "True", "gt_node": "", "pred_node": "SOCIAL_MASTERY"},
    ]
    metrics = independent_benchmark_counts(rows)
    assert metrics["n_scored"] == 4
    assert metrics["safety_correct"] == 2
    assert metrics["unsafe_recall"] == 0.5
    assert independent_rank_auc([False, True], [0.5, 0.5]) == 0.5
    assert independent_cohen_kappa([(True, True), (False, False)])[2] == 1.0


def test_canonical_chapter4_recomputation(tmp_path):
    stats = recompute(tmp_path)

    assert stats["main_benchmark"]["s1_baseline"]["safety_correct"] == 411
    assert stats["main_benchmark"]["s2_consistency"]["node_correct"] == 336
    assert stats["multi_maladaptive"]["full_s1"]["secondary_node_caught"] == 34
    assert stats["expert_spotcheck"]["ag"]["all"]["n_complete"] == 37
    assert stats["expert_spotcheck"]["ag_kl_pairwise"]["n_shared_complete"] == 35

    if expert_inputs_available():
        with (tmp_path / "expert_spotcheck_ag_kl_pairwise.csv").open(newline="", encoding="utf-8") as handle:
            pairwise_rows = list(csv.DictReader(handle))
        assert len(pairwise_rows) == 35
    else:
        assert not (tmp_path / "expert_spotcheck_ag_kl_pairwise.csv").exists()
    assert (tmp_path / "chapter4_stats.tex").is_file()


def test_independent_verifier_matches_canonical_outputs():
    checks = verify()
    assert len(checks) >= 150
