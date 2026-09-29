# Chapter 4 statistics

Statistics are added here in the order they appear while Chapter 4 is revised.
Section-specific filenames follow the current top-level thesis section numbers.
Each script reads row-level source files directly, prints visible numerators and
denominators, and has a corresponding regression test under `tests/`.

Current section-specific coverage:

- Sections 4.1--4.2: evaluation design and dataset/review coverage
- Section 4.3: held-out reference-label validation
- Section 4.4: safety-layer classification results
- Section 4.5: ablation and pipeline analysis
- Section 4.6: failure analysis and boundary cases
- Section 4.7: uncertainty quantification
- Section 4.8: system-level expert evaluation
- Supplementary claims: cross-section prose claims without another computing
  script (run-to-run variation and the three-execution S2 comparison with its
  segmentation decomposition, naive-splitter miss recovery, consistency-miss
  vote details, representative-failure attributions, boundary classifier--expert
  agreement, the S1/S2 confidence-spread comparison, and AUROC
  leave-one-error-out plus bootstrap intervals)

Run the complete section-by-section audit:

```bash
for script in experiments/evaluation_statistics/chapter_4/*.py; do
  uv run python "$script"
done
```

Each empirical number in the revised Chapter 4 is covered by one of these
scripts. Mathematical constants used to explain metrics (for example AUROC
endpoints) and enumerated score-code definitions are definitions rather than
dataset-derived statistics.

## 4.1--4.2 Evaluation Design and Dataset Rationale

Run:

```bash
uv run python experiments/evaluation_statistics/chapter_4/sections_4_1_2_evaluation_design.py
```

The script derives the development and held-out dataset compositions, the
truncated-suite exclusion, the supplementary slice size, and internal review
coverage from the dataset and annotation-log JSON files.

## 4.3 Held-Out Reference-Label Validation

Run:

```bash
uv run python experiments/evaluation_statistics/chapter_4/section_4_3_reference_label_validation.py
```

The script reads these row-level tables:

- `data/expert_analysis/ag_2026_07/spotcheck_merged.csv`
- `data/expert_analysis/kl_2026_06/spotcheck_merged.csv`

It recomputes the agreement values directly from the reference, expert, and
classifier columns. It does not use the precomputed agreement columns in those
CSV files.

## 4.4 Safety Layer Classification Results

Run:

```bash
uv run python experiments/evaluation_statistics/chapter_4/section_4_4_classification_results.py
```

The script reads the four row-level development benchmark result files directly
and prints the statistics in the order they appear in Section 4.4. For the
post-fix held-out evaluation in Section 4.4.5, it reproduces the headline
metrics, per-node table, complete strict-node confusion counts, the post-hoc
affect-expression/emotional-mastery sensitivity analysis, and the two binary
safety-error records.

## 4.5 Ablation and Pipeline Analysis

Run:

```bash
uv run python experiments/evaluation_statistics/chapter_4/section_4_5_ablation_analysis.py
```

The script reproduces the main five-condition ablation table, the clean versus
multipart stratification, the multipart dataset composition, the supplementary
multi-maladaptive table, and the illustrative case trace from row-level files.

## 4.6 Failure Analysis and Edge Cases

Run:

```bash
uv run python experiments/evaluation_statistics/chapter_4/section_4_6_failure_analysis.py
```

The script derives the reported safety-error totals, primary-node confusion
counts, false-alarm grouping, no-segmentation misses, boundary agreement,
four unchanged-replay cases, and the complete boundary-theme synthesis from
row-level CSV and JSON files.

## 4.7 Uncertainty Quantification

Run:

```bash
uv run python experiments/evaluation_statistics/chapter_4/section_4_7_uncertainty_quantification.py
```

The script independently computes AUROC by pairwise ranking, all reported
confidence thresholds, classifier-confidence means, and low-confidence expert
spot-check counts.

## 4.8 System-Level and Expert Evaluation

Run:

```bash
uv run python experiments/evaluation_statistics/chapter_4/section_4_8_system_expert_evaluation.py
```

The script reproduces completion counts, condition distributions, within-case
changes, shared-response agreement, and qualitative-note flags from both
experts' row-level response-rating files.

## Supplementary Claims (cross-section)

Run:

```bash
uv run python experiments/evaluation_statistics/chapter_4/section_4_supplementary_claims.py
```

The script covers chapter claims identified in the August 2026 number audit as
having no other computing script: the run-to-run variation of the deterministic
pipeline (4.4.6 and Chapter 5), the naive-splitter recovery of no-segmentation
misses (4.5.1), the per-segment votes behind the two S2-consistency unsafe
misses (4.6.3), the per-condition attribution of the representative failure
cases (4.6.4), boundary-row classifier--expert agreement for both raters
(4.6.2), KL's confidence on individually cited cases (4.4.5, 4.6.2), the
S1-versus-corrected-S2 code-token confidence spread (4.7.1), and
leave-one-error-out ranges plus seeded bootstrap 95% intervals for the reported
AUROC values (the 4.7 small-sample caveat).
