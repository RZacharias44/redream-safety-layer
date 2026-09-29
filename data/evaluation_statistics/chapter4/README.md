# Chapter 4 Statistics

This folder is the canonical audit bundle for the thesis evaluation chapter.
Every empirical number in Chapter 4 is computed from the row-level records by
the scripts below; nothing here is typed in by hand.

## Scripts

- `experiments/evaluation_statistics/recompute_chapter4.py` — canonical
  generator. Reads the benchmark result files, the expert annotation tables
  and the system-evaluation records, writes the files listed below.
- `experiments/evaluation_statistics/verify_chapter4_statistics.py` —
  independent verifier. Imports none of the project analysis code, recomputes
  the values with visible formulas and checks them against the canonical JSON.
- `experiments/evaluation_statistics/chapter_4/*.py` — one script per thesis
  section; each prints its statistics in thesis order with visible numerators
  and denominators (see the README next to them).

Regenerate and verify from the repository root:

```bash
uv run python experiments/evaluation_statistics/recompute_chapter4.py
uv run python experiments/evaluation_statistics/verify_chapter4_statistics.py
for script in experiments/evaluation_statistics/chapter_4/*.py; do uv run python "$script"; done
```

## Files

- `chapter4_statistics.json` — machine-readable source of truth for the
  benchmark, ablation, uncertainty, held-out, expert and system-level
  statistics. Rates are proportions (`0.838`); percentages are presentation
  formatting (`83.8%`).
- `chapter4_statistics.md` — compact human-readable headline report.
- `chapter4_stats.tex` — generated LaTeX macros used by the thesis tables.
- `benchmark_inventory_tables.tex` — generated LaTeX table of the benchmark
  material (`generate_benchmark_inventory.py`).
- `provenance.json` — every input path with size and SHA-256, plus the
  checksums of the generating scripts.
- `independent_calculation_audit.md` — the verifier's report: expanded
  formulas, integer numerators and denominators, and every equality check.
- `benchmark_row_audit.csv` — one row per benchmark case with its inclusion
  and correctness decisions.
- `expert_spotcheck_ag_kl_pairwise.csv`, `expert_spotcheck_row_audit.csv`,
  `system_eval_shared_row_audit.csv` — row-level expert audits. Development
  repository only: they require the expert annotation tables.

## Without the expert annotation tables

The row-level expert annotations under `data/expert_analysis/` are personal
data of the two raters and are distributed only with their consent. When they
are absent, `recompute_chapter4.py` recomputes every benchmark-derived value
and carries the four expert-derived blocks (`expert_spotcheck`,
`system_expert_evaluation`, `uncertainty.kl_spotcheck_confidence`,
`dataset_composition.spotcheck_queue_test_types`) over from the existing
canonical JSON; `provenance.json` then lists the withheld inputs under
`withheld_inputs`. The verifier skips the expert-derived checks and says so in
its report; the section scripts for §4.3 and §4.8 print a notice instead of
results, and §4.6/§4.7 skip their expert-dependent parts.

## Denominator rules

- Development headline metrics use the 420 scored cases and exclude the 60
  unscored boundary cases.
- Held-out test metrics exclude the truncated buried-alive suite and the
  unscored boundary cases.
- Expert/reference agreement includes only rows with both a binary safety
  judgment and a primary node.
- AG–KL inter-rater agreement includes only cases completed by both raters.
