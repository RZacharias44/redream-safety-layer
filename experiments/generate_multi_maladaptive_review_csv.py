#!/usr/bin/env python3
"""
MAL+MAL annotation review CSV generator (Deliverable 3).

Reads data/benchmarks/multi_maladaptive_dataset.json and emits
data/benchmarks/multi_maladaptive_dataset_review.csv with columns:

    case_id, nightmare, input, gpt5_label_primary, gpt5_label_secondary,
    gpt5_pattern, ramon_safe, ramon_primary_node, ramon_secondary_node, ramon_notes

`gpt5_*` columns are filled from the slice JSON. `ramon_*` columns are blank
for Ramon to fill (in a spreadsheet). After the annotation pass, run a merge
script to fold ramon_* back into the JSON, dropping cases Ramon flags as
malformed and relabeling cases where his primary/secondary differ.

Per HANDOFF_MAL_MAL_CONTINUATION.md §"Annotation CSV".

Usage:
    uv run python experiments/generate_multi_maladaptive_review_csv.py
"""

import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
SLICE_JSON = REPO_ROOT / "data" / "benchmarks" / "multi_maladaptive_dataset.json"
REVIEW_CSV = REPO_ROOT / "data" / "benchmarks" / "multi_maladaptive_dataset_review.csv"

COLUMNS = [
    "case_id",
    "pair_id",
    "nightmare",
    "input",
    "gpt5_label_primary",
    "gpt5_label_secondary",
    "gpt5_pattern",
    "ramon_safe",
    "ramon_primary_node",
    "ramon_secondary_node",
    "ramon_notes",
]


def main() -> None:
    if not SLICE_JSON.exists():
        print(f"✗ Slice JSON not found: {SLICE_JSON}")
        print("  Run experiments/generate_multi_maladaptive_slice.py first.")
        sys.exit(1)

    with open(SLICE_JSON, "r", encoding="utf-8") as f:
        suites = json.load(f)

    rows = []
    case_idx = 0
    for suite in suites:
        suite_name = suite["suite"]
        nightmare = suite["nightmare"]
        for case in suite["tests"]:
            case_idx += 1
            # case_id format: <suite-slug>_<pair_id>_<idx> for traceability.
            slug = "".join(ch if ch.isalnum() else "_" for ch in suite_name.lower())[:40].rstrip("_")
            pair_id = case.get("pair_id", "UNKNOWN")
            case_id = f"{case_idx:03d}_{slug}_{pair_id}"
            rows.append({
                "case_id": case_id,
                "pair_id": pair_id,
                "nightmare": nightmare,
                "input": case["input"],
                "gpt5_label_primary": case["expected_node"],
                "gpt5_label_secondary": case["expected_secondary_node"],
                "gpt5_pattern": case.get("pattern", ""),
                "ramon_safe": "",
                "ramon_primary_node": "",
                "ramon_secondary_node": "",
                "ramon_notes": "",
            })

    REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Wrote {len(rows)} rows to {REVIEW_CSV}")
    print()
    print("Annotation rubric for Ramon:")
    print("  - ramon_safe: 'true' or 'false' (false expected for all MAL+MAL cases —")
    print("    flag any case where the input is NOT clearly unsafe).")
    print("  - ramon_primary_node: which maladaptive node should triage prioritize?")
    print("    Use the 12-code set. Leave blank to confirm gpt5_label_primary.")
    print("  - ramon_secondary_node: the *other* maladaptive node present.")
    print("    Leave blank to confirm gpt5_label_secondary.")
    print("  - ramon_notes: free-text. Use 'DROP' to flag cases that aren't actually")
    print("    MAL+MAL (e.g. one part is adaptive, or only one part is recognizable).")
    print()
    print("Reference: data/benchmarks/test_dataset_annotation_log.json (previous annotation pattern).")


if __name__ == "__main__":
    main()
