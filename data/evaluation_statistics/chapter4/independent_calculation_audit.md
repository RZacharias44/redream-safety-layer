# Independent Chapter 4 Calculation Audit

This report was produced by `verify_chapter4_statistics.py`, which does not import the canonical calculation code.

## Formulas

- Safety accuracy = `(TP_unsafe + TN_safe) / N_scored`.
- Unsafe recall = `TP_unsafe / (TP_unsafe + FN_unsafe)`.
- Unsafe precision = `TP_unsafe / (TP_unsafe + FP_safe)`.
- Strict node accuracy = `exact node matches / N_scored`.
- Cohen's kappa = `(observed agreement - chance agreement) / (1 - chance agreement)`.
- AUROC is the fraction of positive/negative pairs ordered correctly; ties count as one half.

Blank ground-truth safety values are unscored boundary cases. Unsafe is the positive class.

## Benchmark calculations

| Source | N | TP | TN | FP | FN | Safety equation | Node equation |
|---|---:|---:|---:|---:|---:|---|---|
| `main_benchmark.s1_baseline` | 420 | 179 | 232 | 8 | 1 | `(179+232)/420` = 0.978571 | `328/420` = 0.780952 |
| `main_benchmark.s1_consistency` | 420 | 177 | 232 | 8 | 3 | `(177+232)/420` = 0.973810 | `323/420` = 0.769048 |
| `main_benchmark.s2_baseline` | 420 | 179 | 233 | 7 | 1 | `(179+233)/420` = 0.980952 | `332/420` = 0.790476 |
| `main_benchmark.s2_consistency` | 420 | 178 | 235 | 5 | 2 | `(178+235)/420` = 0.983333 | `336/420` = 0.800000 |
| `main_benchmark.s2_baseline_corrected` | 420 | 175 | 235 | 5 | 5 | `(175+235)/420` = 0.976190 | `331/420` = 0.788095 |
| `ablations.full_s1` | 420 | 179 | 232 | 8 | 1 | `(179+232)/420` = 0.978571 | `328/420` = 0.780952 |
| `ablations.stage_merge` | 420 | 177 | 236 | 4 | 3 | `(177+236)/420` = 0.983333 | `365/420` = 0.869048 |
| `ablations.no_segmentation` | 420 | 173 | 240 | 0 | 7 | `(173+240)/420` = 0.983333 | `352/420` = 0.838095 |
| `ablations.meta_filtering_off` | 420 | 179 | 222 | 18 | 1 | `(179+222)/420` = 0.954762 | `318/420` = 0.757143 |
| `ablations.naive_splitter` | 420 | 178 | 231 | 9 | 2 | `(178+231)/420` = 0.973810 | `332/420` = 0.790476 |

## Segmentation strata

- `full_s1/one_pattern_clean` includes 360 rows: safety `351/360`, unsafe recall `149/150`, node `286/360`.
- `full_s1/multipart` includes 60 rows: safety `60/60`, unsafe recall `30/30`, node `42/60`.
- `no_segmentation/one_pattern_clean` includes 360 rows: safety `353/360`, unsafe recall `143/150`, node `310/360`.
- `no_segmentation/multipart` includes 60 rows: safety `60/60`, unsafe recall `30/30`, node `42/60`.

## Composite multi-maladaptive slice

- `full_s1`: primary `35/35`, secondary `34/35`, any maladaptive `35/35`.
- `stage_merge`: primary `35/35`, secondary `35/35`, any maladaptive `35/35`.
- `no_segmentation`: primary `25/35`, secondary `6/35`, any maladaptive `31/35`.
- `meta_filtering_off`: primary `35/35`, secondary `34/35`, any maladaptive `35/35`.
- `naive_splitter`: primary `25/35`, secondary `6/35`, any maladaptive `31/35`.

## Held-out test

Filter: exclude rows whose suite contains `buried alive`, then exclude dataset `test_type=boundary`. Result: safety `229/231`, unsafe recall `89/90`, node `186/231`.

## Uncertainty AUROC cross-check

- `s1_baseline/logprob`: N=416, safety AUROC=0.770, node AUROC=0.853.
- `s1_consistency/agreement_ratio`: N=417, safety AUROC=0.682, node AUROC=0.675.
- `s2_baseline/logprob`: N=417, safety AUROC=0.619, node AUROC=0.605.
- `s2_consistency/agreement_ratio`: N=417, safety AUROC=0.739, node AUROC=0.712.
- `s2_baseline_corrected/logprob`: N=416, safety AUROC=0.546, node AUROC=0.689.

## Expert-derived values

Not verified in this run: the row-level expert annotation tables are not present (they are personal data of the raters and are distributed only with their consent). The `expert_spotcheck`, `system_expert_evaluation`, `uncertainty.kl_spotcheck_confidence` and `dataset_composition.spotcheck_queue_test_types` blocks of `chapter4_statistics.json` are carried over from the canonical statistics and were not recomputed.

## Validation result

All 211 independently recomputed values matched `chapter4_statistics.json`.

The CSV files beside this report show every included/excluded row and every row-level correctness decision.

<details><summary>Individual equality checks</summary>

- PASS `main_benchmark.s1_baseline.n_rows` = `480`
- PASS `main_benchmark.s1_baseline.n_scored` = `420`
- PASS `main_benchmark.s1_baseline.n_unscored` = `60`
- PASS `main_benchmark.s1_baseline.n_safe` = `240`
- PASS `main_benchmark.s1_baseline.n_unsafe` = `180`
- PASS `main_benchmark.s1_baseline.safety_correct` = `411`
- PASS `main_benchmark.s1_baseline.safety_accuracy` = `0.978571`
- PASS `main_benchmark.s1_baseline.unsafe_true_positive` = `179`
- PASS `main_benchmark.s1_baseline.unsafe_false_negative` = `1`
- PASS `main_benchmark.s1_baseline.unsafe_false_positive` = `8`
- PASS `main_benchmark.s1_baseline.safe_true_negative` = `232`
- PASS `main_benchmark.s1_baseline.unsafe_recall` = `0.994444`
- PASS `main_benchmark.s1_baseline.unsafe_precision` = `0.957219`
- PASS `main_benchmark.s1_baseline.node_correct` = `328`
- PASS `main_benchmark.s1_baseline.node_accuracy` = `0.780952`
- PASS `main_benchmark.s1_consistency.n_rows` = `480`
- PASS `main_benchmark.s1_consistency.n_scored` = `420`
- PASS `main_benchmark.s1_consistency.n_unscored` = `60`
- PASS `main_benchmark.s1_consistency.n_safe` = `240`
- PASS `main_benchmark.s1_consistency.n_unsafe` = `180`
- PASS `main_benchmark.s1_consistency.safety_correct` = `409`
- PASS `main_benchmark.s1_consistency.safety_accuracy` = `0.97381`
- PASS `main_benchmark.s1_consistency.unsafe_true_positive` = `177`
- PASS `main_benchmark.s1_consistency.unsafe_false_negative` = `3`
- PASS `main_benchmark.s1_consistency.unsafe_false_positive` = `8`
- PASS `main_benchmark.s1_consistency.safe_true_negative` = `232`
- PASS `main_benchmark.s1_consistency.unsafe_recall` = `0.983333`
- PASS `main_benchmark.s1_consistency.unsafe_precision` = `0.956757`
- PASS `main_benchmark.s1_consistency.node_correct` = `323`
- PASS `main_benchmark.s1_consistency.node_accuracy` = `0.769048`
- PASS `main_benchmark.s2_baseline.n_rows` = `480`
- PASS `main_benchmark.s2_baseline.n_scored` = `420`
- PASS `main_benchmark.s2_baseline.n_unscored` = `60`
- PASS `main_benchmark.s2_baseline.n_safe` = `240`
- PASS `main_benchmark.s2_baseline.n_unsafe` = `180`
- PASS `main_benchmark.s2_baseline.safety_correct` = `412`
- PASS `main_benchmark.s2_baseline.safety_accuracy` = `0.980952`
- PASS `main_benchmark.s2_baseline.unsafe_true_positive` = `179`
- PASS `main_benchmark.s2_baseline.unsafe_false_negative` = `1`
- PASS `main_benchmark.s2_baseline.unsafe_false_positive` = `7`
- PASS `main_benchmark.s2_baseline.safe_true_negative` = `233`
- PASS `main_benchmark.s2_baseline.unsafe_recall` = `0.994444`
- PASS `main_benchmark.s2_baseline.unsafe_precision` = `0.962366`
- PASS `main_benchmark.s2_baseline.node_correct` = `332`
- PASS `main_benchmark.s2_baseline.node_accuracy` = `0.790476`
- PASS `main_benchmark.s2_consistency.n_rows` = `480`
- PASS `main_benchmark.s2_consistency.n_scored` = `420`
- PASS `main_benchmark.s2_consistency.n_unscored` = `60`
- PASS `main_benchmark.s2_consistency.n_safe` = `240`
- PASS `main_benchmark.s2_consistency.n_unsafe` = `180`
- PASS `main_benchmark.s2_consistency.safety_correct` = `413`
- PASS `main_benchmark.s2_consistency.safety_accuracy` = `0.983333`
- PASS `main_benchmark.s2_consistency.unsafe_true_positive` = `178`
- PASS `main_benchmark.s2_consistency.unsafe_false_negative` = `2`
- PASS `main_benchmark.s2_consistency.unsafe_false_positive` = `5`
- PASS `main_benchmark.s2_consistency.safe_true_negative` = `235`
- PASS `main_benchmark.s2_consistency.unsafe_recall` = `0.988889`
- PASS `main_benchmark.s2_consistency.unsafe_precision` = `0.972678`
- PASS `main_benchmark.s2_consistency.node_correct` = `336`
- PASS `main_benchmark.s2_consistency.node_accuracy` = `0.8`
- PASS `main_benchmark.s2_baseline_corrected.n_rows` = `480`
- PASS `main_benchmark.s2_baseline_corrected.n_scored` = `420`
- PASS `main_benchmark.s2_baseline_corrected.n_unscored` = `60`
- PASS `main_benchmark.s2_baseline_corrected.n_safe` = `240`
- PASS `main_benchmark.s2_baseline_corrected.n_unsafe` = `180`
- PASS `main_benchmark.s2_baseline_corrected.safety_correct` = `410`
- PASS `main_benchmark.s2_baseline_corrected.safety_accuracy` = `0.97619`
- PASS `main_benchmark.s2_baseline_corrected.unsafe_true_positive` = `175`
- PASS `main_benchmark.s2_baseline_corrected.unsafe_false_negative` = `5`
- PASS `main_benchmark.s2_baseline_corrected.unsafe_false_positive` = `5`
- PASS `main_benchmark.s2_baseline_corrected.safe_true_negative` = `235`
- PASS `main_benchmark.s2_baseline_corrected.unsafe_recall` = `0.972222`
- PASS `main_benchmark.s2_baseline_corrected.unsafe_precision` = `0.972222`
- PASS `main_benchmark.s2_baseline_corrected.node_correct` = `331`
- PASS `main_benchmark.s2_baseline_corrected.node_accuracy` = `0.788095`
- PASS `ablations.full_s1.n_rows` = `480`
- PASS `ablations.full_s1.n_scored` = `420`
- PASS `ablations.full_s1.n_unscored` = `60`
- PASS `ablations.full_s1.n_safe` = `240`
- PASS `ablations.full_s1.n_unsafe` = `180`
- PASS `ablations.full_s1.safety_correct` = `411`
- PASS `ablations.full_s1.safety_accuracy` = `0.978571`
- PASS `ablations.full_s1.unsafe_true_positive` = `179`
- PASS `ablations.full_s1.unsafe_false_negative` = `1`
- PASS `ablations.full_s1.unsafe_false_positive` = `8`
- PASS `ablations.full_s1.safe_true_negative` = `232`
- PASS `ablations.full_s1.unsafe_recall` = `0.994444`
- PASS `ablations.full_s1.unsafe_precision` = `0.957219`
- PASS `ablations.full_s1.node_correct` = `328`
- PASS `ablations.full_s1.node_accuracy` = `0.780952`
- PASS `ablations.stage_merge.n_rows` = `480`
- PASS `ablations.stage_merge.n_scored` = `420`
- PASS `ablations.stage_merge.n_unscored` = `60`
- PASS `ablations.stage_merge.n_safe` = `240`
- PASS `ablations.stage_merge.n_unsafe` = `180`
- PASS `ablations.stage_merge.safety_correct` = `413`
- PASS `ablations.stage_merge.safety_accuracy` = `0.983333`
- PASS `ablations.stage_merge.unsafe_true_positive` = `177`
- PASS `ablations.stage_merge.unsafe_false_negative` = `3`
- PASS `ablations.stage_merge.unsafe_false_positive` = `4`
- PASS `ablations.stage_merge.safe_true_negative` = `236`
- PASS `ablations.stage_merge.unsafe_recall` = `0.983333`
- PASS `ablations.stage_merge.unsafe_precision` = `0.977901`
- PASS `ablations.stage_merge.node_correct` = `365`
- PASS `ablations.stage_merge.node_accuracy` = `0.869048`
- PASS `ablations.no_segmentation.n_rows` = `480`
- PASS `ablations.no_segmentation.n_scored` = `420`
- PASS `ablations.no_segmentation.n_unscored` = `60`
- PASS `ablations.no_segmentation.n_safe` = `240`
- PASS `ablations.no_segmentation.n_unsafe` = `180`
- PASS `ablations.no_segmentation.safety_correct` = `413`
- PASS `ablations.no_segmentation.safety_accuracy` = `0.983333`
- PASS `ablations.no_segmentation.unsafe_true_positive` = `173`
- PASS `ablations.no_segmentation.unsafe_false_negative` = `7`
- PASS `ablations.no_segmentation.unsafe_false_positive` = `0`
- PASS `ablations.no_segmentation.safe_true_negative` = `240`
- PASS `ablations.no_segmentation.unsafe_recall` = `0.961111`
- PASS `ablations.no_segmentation.unsafe_precision` = `1.0`
- PASS `ablations.no_segmentation.node_correct` = `352`
- PASS `ablations.no_segmentation.node_accuracy` = `0.838095`
- PASS `ablations.meta_filtering_off.n_rows` = `480`
- PASS `ablations.meta_filtering_off.n_scored` = `420`
- PASS `ablations.meta_filtering_off.n_unscored` = `60`
- PASS `ablations.meta_filtering_off.n_safe` = `240`
- PASS `ablations.meta_filtering_off.n_unsafe` = `180`
- PASS `ablations.meta_filtering_off.safety_correct` = `401`
- PASS `ablations.meta_filtering_off.safety_accuracy` = `0.954762`
- PASS `ablations.meta_filtering_off.unsafe_true_positive` = `179`
- PASS `ablations.meta_filtering_off.unsafe_false_negative` = `1`
- PASS `ablations.meta_filtering_off.unsafe_false_positive` = `18`
- PASS `ablations.meta_filtering_off.safe_true_negative` = `222`
- PASS `ablations.meta_filtering_off.unsafe_recall` = `0.994444`
- PASS `ablations.meta_filtering_off.unsafe_precision` = `0.908629`
- PASS `ablations.meta_filtering_off.node_correct` = `318`
- PASS `ablations.meta_filtering_off.node_accuracy` = `0.757143`
- PASS `ablations.naive_splitter.n_rows` = `480`
- PASS `ablations.naive_splitter.n_scored` = `420`
- PASS `ablations.naive_splitter.n_unscored` = `60`
- PASS `ablations.naive_splitter.n_safe` = `240`
- PASS `ablations.naive_splitter.n_unsafe` = `180`
- PASS `ablations.naive_splitter.safety_correct` = `409`
- PASS `ablations.naive_splitter.safety_accuracy` = `0.97381`
- PASS `ablations.naive_splitter.unsafe_true_positive` = `178`
- PASS `ablations.naive_splitter.unsafe_false_negative` = `2`
- PASS `ablations.naive_splitter.unsafe_false_positive` = `9`
- PASS `ablations.naive_splitter.safe_true_negative` = `231`
- PASS `ablations.naive_splitter.unsafe_recall` = `0.988889`
- PASS `ablations.naive_splitter.unsafe_precision` = `0.951872`
- PASS `ablations.naive_splitter.node_correct` = `332`
- PASS `ablations.naive_splitter.node_accuracy` = `0.790476`
- PASS `segmentation_strata.full_s1.one_pattern_clean.n_scored` = `360`
- PASS `segmentation_strata.full_s1.one_pattern_clean.safety_correct` = `351`
- PASS `segmentation_strata.full_s1.one_pattern_clean.unsafe_true_positive` = `149`
- PASS `segmentation_strata.full_s1.one_pattern_clean.unsafe_false_negative` = `1`
- PASS `segmentation_strata.full_s1.one_pattern_clean.node_correct` = `286`
- PASS `segmentation_strata.full_s1.multipart.n_scored` = `60`
- PASS `segmentation_strata.full_s1.multipart.safety_correct` = `60`
- PASS `segmentation_strata.full_s1.multipart.unsafe_true_positive` = `30`
- PASS `segmentation_strata.full_s1.multipart.unsafe_false_negative` = `0`
- PASS `segmentation_strata.full_s1.multipart.node_correct` = `42`
- PASS `segmentation_strata.no_segmentation.one_pattern_clean.n_scored` = `360`
- PASS `segmentation_strata.no_segmentation.one_pattern_clean.safety_correct` = `353`
- PASS `segmentation_strata.no_segmentation.one_pattern_clean.unsafe_true_positive` = `143`
- PASS `segmentation_strata.no_segmentation.one_pattern_clean.unsafe_false_negative` = `7`
- PASS `segmentation_strata.no_segmentation.one_pattern_clean.node_correct` = `310`
- PASS `segmentation_strata.no_segmentation.multipart.n_scored` = `60`
- PASS `segmentation_strata.no_segmentation.multipart.safety_correct` = `60`
- PASS `segmentation_strata.no_segmentation.multipart.unsafe_true_positive` = `30`
- PASS `segmentation_strata.no_segmentation.multipart.unsafe_false_negative` = `0`
- PASS `segmentation_strata.no_segmentation.multipart.node_correct` = `42`
- PASS `multi_maladaptive.full_s1.n` = `35`
- PASS `multi_maladaptive.full_s1.primary_node_correct` = `35`
- PASS `multi_maladaptive.full_s1.secondary_node_caught` = `34`
- PASS `multi_maladaptive.full_s1.any_maladaptive_caught` = `35`
- PASS `multi_maladaptive.stage_merge.n` = `35`
- PASS `multi_maladaptive.stage_merge.primary_node_correct` = `35`
- PASS `multi_maladaptive.stage_merge.secondary_node_caught` = `35`
- PASS `multi_maladaptive.stage_merge.any_maladaptive_caught` = `35`
- PASS `multi_maladaptive.no_segmentation.n` = `35`
- PASS `multi_maladaptive.no_segmentation.primary_node_correct` = `25`
- PASS `multi_maladaptive.no_segmentation.secondary_node_caught` = `6`
- PASS `multi_maladaptive.no_segmentation.any_maladaptive_caught` = `31`
- PASS `multi_maladaptive.meta_filtering_off.n` = `35`
- PASS `multi_maladaptive.meta_filtering_off.primary_node_correct` = `35`
- PASS `multi_maladaptive.meta_filtering_off.secondary_node_caught` = `34`
- PASS `multi_maladaptive.meta_filtering_off.any_maladaptive_caught` = `35`
- PASS `multi_maladaptive.naive_splitter.n` = `35`
- PASS `multi_maladaptive.naive_splitter.primary_node_correct` = `25`
- PASS `multi_maladaptive.naive_splitter.secondary_node_caught` = `6`
- PASS `multi_maladaptive.naive_splitter.any_maladaptive_caught` = `31`
- PASS `heldout_test_postfix.n_scored` = `231`
- PASS `heldout_test_postfix.safety_correct` = `229`
- PASS `heldout_test_postfix.unsafe_true_positive` = `89`
- PASS `heldout_test_postfix.unsafe_false_negative` = `1`
- PASS `heldout_test_postfix.node_correct` = `186`
- PASS `heldout_test_postfix.affect_emotional_collapsed_node_correct` = `196`
- PASS `uncertainty.s1_baseline.logprob.n` = `416`
- PASS `uncertainty.s1_baseline.logprob.safety_correctness_auroc` = `0.77`
- PASS `uncertainty.s1_baseline.logprob.node_correctness_auroc` = `0.853`
- PASS `uncertainty.s1_consistency.agreement_ratio.n` = `417`
- PASS `uncertainty.s1_consistency.agreement_ratio.safety_correctness_auroc` = `0.682`
- PASS `uncertainty.s1_consistency.agreement_ratio.node_correctness_auroc` = `0.675`
- PASS `uncertainty.s2_baseline.logprob.n` = `417`
- PASS `uncertainty.s2_baseline.logprob.safety_correctness_auroc` = `0.619`
- PASS `uncertainty.s2_baseline.logprob.node_correctness_auroc` = `0.605`
- PASS `uncertainty.s2_consistency.agreement_ratio.n` = `417`
- PASS `uncertainty.s2_consistency.agreement_ratio.safety_correctness_auroc` = `0.739`
- PASS `uncertainty.s2_consistency.agreement_ratio.node_correctness_auroc` = `0.712`
- PASS `uncertainty.s2_baseline_corrected.logprob.n` = `416`
- PASS `uncertainty.s2_baseline_corrected.logprob.safety_correctness_auroc` = `0.546`
- PASS `uncertainty.s2_baseline_corrected.logprob.node_correctness_auroc` = `0.689`

</details>
