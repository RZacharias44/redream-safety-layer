# Benchmark datasets and result files

Naming scheme for result files: `<dataset>_<system>_<condition>.csv`, where
`dataset` is `dev` (development benchmark), `test` (held-out test set) or
`multi_maladaptive` (supplementary slice), `system` is `s1` (direct
classifier) or `s2` (structured-rationale classifier), and `condition` is the
pipeline configuration. All result files are written by
`experiments/run_benchmark.py` unless stated otherwise; run it with `--help`
for the flags. Every Chapter 4 statistic is recomputed from these files by
`experiments/evaluation_statistics/recompute_chapter4.py` and the
section scripts under `experiments/evaluation_statistics/chapter_4/`.

## Datasets

| File | Content | Produced by |
| --- | --- | --- |
| `dev_dataset.json` | Development benchmark: 10 nightmare suites, 480 cases, 12 ontology nodes | `experiments/generate_dataset.py` (GPT-5.4); tracked version last changed Apr 16, 2026 |
| `test_dataset.json` | Held-out test set: 7 nightmare suites, 315 cases | `experiments/generate_dataset.py`, Apr 16, 2026 |
| `multi_maladaptive_dataset.json` | Supplementary slice: 5 development nightmares × 7 maladaptive-pair patterns, 35 cases | `experiments/generate_multi_maladaptive_slice.py`, Apr 30, 2026 |
| `multi_maladaptive_dataset_review.csv` | Human certification of the 35 slice labels (all confirmed) | `experiments/generate_multi_maladaptive_review_csv.py` + manual review, May 2026 |
| `test_dataset_annotation_log.json` | Reference-label review of the test set (boundary, affect-expression, violent-revenge and stratified spot-check subsets) | manual annotation, Apr 2026 |
| `dev_dataset_spot_check_log.json`, `dev_dataset_spot_check_round2.json` | Reference-label spot checks of the development set | manual annotation, Mar–Apr 2026 |

## Result files used by the thesis

Run dates are the dates of the recorded runs. `n5` = classification
consistency with five stochastic classifier calls per segment
(`--consistency 5`); the temperature-0 pipeline verdict is kept in
`pred_safe`/`pred_node`, the votes in `vote_distribution`.

| File | Rows | Configuration | Run | Thesis use |
| --- | --- | --- | --- | --- |
| `dev_s1_baseline.csv` | 480 | full pipeline, S1 (defaults) | Apr 23–24, 2026 | §4.4 results, §4.5 reference cell, §4.6, §4.7 |
| `dev_s1_consistency_n5.csv` | 480 | S1, `--consistency 5` (segments lost to connection drops completed by `experiments/topup_s1_consistency.py`) | Apr 23–24, 2026 | §4.4, §4.6, §4.7 |
| `dev_s2_baseline.csv` | 480 | `--classifier s2` | Apr 23–24, 2026 | §4.4 results, §4.7 |
| `dev_s2_consistency_n5.csv` | 480 | `--classifier s2 --consistency 5` | Apr 23–24, 2026 | §4.4, §4.6, §4.7 |
| `dev_s2_baseline_rerun_corrected_logprobs.csv` | 480 | `--classifier s2`, rerun after the S2 log-probability extractor fix; only the confidence column is methodologically different | Aug 5, 2026 | §4.7 confidence analysis |
| `dev_s1_ablation_stage_merge.csv` | 480 | `--critic merged` (segmentation and classification in one call) | May 21, 2026 | §4.5 |
| `dev_s1_ablation_no_segmentation.csv` | 480 | `--no-segment` (whole turn classified once) | May 19, 2026 | §4.5, §4.6 |
| `dev_s1_ablation_meta_filter_off.csv` | 480 | `--no-meta-filter` | May 19, 2026 | §4.5 |
| `dev_s1_ablation_naive_splitter.csv` | 480 | `--segmenter naive` (sentence splitter instead of the LLM segmenter) | May 19, 2026 | §4.5 |
| `multi_maladaptive_s1_baseline.csv` | 35 | full pipeline, S1, `--input multi_maladaptive_dataset.json` | May 19, 2026 | §4.5 supplementary slice |
| `multi_maladaptive_s1_ablation_{stage_merge,no_segmentation,meta_filter_off,naive_splitter}.csv` | 35 | as the `dev_s1_ablation_*` files, on the slice | May 19/21, 2026 | §4.5 supplementary slice |
| `test_s1_baseline.csv` | 315 | full pipeline, S1, `--input test_dataset.json` | Apr 29, 2026 | §4.3, §4.4 held-out results, §4.7 |
| `dev_s2_logprob_diagnostic.csv` | 92 | `experiments/s2_logprob_diagnostic.py`: token-span check of the S2 confidence extractor on a stratified sample | Aug 5, 2026 | §4.7 |
| `dev_s2_rationale_check.csv` | 116 | `experiments/s2_rationale_check.py`: does the S2 rationale name the chosen code | Sep 4, 2026 | §4.7 |

## Result files not used by the thesis

Kept for completeness in the research repository only; they are not part of the public code release, and nothing in Chapter 4 depends on them.

- `multi_maladaptive_s1_consistency_n5.csv`, `multi_maladaptive_s2_baseline.csv`, `multi_maladaptive_s2_consistency_n5.csv` — further slice configurations (May 19, 2026); the thesis table reports S1 conditions only.
- `dev_s1_ablation_stage_merge_v1.csv`, `dev_s1_ablation_stage_merge_prompted_unclassified.csv` and their `multi_maladaptive_` counterparts — earlier stage-merge prompt variants (Jun 3, 2026), superseded.
- `benchmark_results_s1_ctx.csv`, `benchmark_results_s2_ctx.csv` — context-aware few-shot ablation (`--context-examples`), negative result, not reported.
- `consistency_errors_*.csv` — targeted consistency on an error/control slice (Apr 14, 2026), not reported.
- `s2_logprob_diagnostic_prefix.csv` — the diagnostic before the extractor fix (documents the defect).
- `test_benchmark_results.csv` — test-set run before the segmenter determinism fix (Apr 16, 2026), superseded by `test_s1_baseline.csv`.
- `benchmark_results.csv`-era files (`benchmark_results_1.csv`, `_2.csv`, `_2_small.csv`, `_3.csv`, `benchmark_margin.csv`, `benchmark_results_pre_meta_fix.csv`, `benchmark_test.csv`), `dev_dataset_1.json`, `dev_dataset_2.json`, `dev_dataset_pre_affect_fix.json`, `dev_dataset_pre_meta_fix.json` — development history before the final dataset and pipeline.
- `segmenter_determinism_probe.json`, `segmenter_determinism_probe.md`, `segmenter_sw_temp0_deep_probe.json` — segmenter determinism probes (`experiments/determinism_probe/`), background to the Apr 22–23 segmenter fix.
- `archive/pre_fix/` — runs before the segmenter determinism fix, provenance only (see its README).
