"""
Top-up for S1 consistency benchmark.

The 2026-04-23 overnight run had a burst of connection errors around test indices
75-99 (25 cases in suites 2 and 3), producing pred_node='ERROR' and empty
vote_distribution. The retry logic added to AI/agent.py mid-run was not picked
up by the already-loaded S1 consistency process — so those cases remained broken.
This script re-runs only those 25 cases with the retry-enabled code and patches
them into dev_s1_consistency_n5.csv in place.

Usage:
    uv run python experiments/topup_s1_consistency.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from safety.critic import SafetyCritic
from safety.classifier import ClinicalClassifier
from safety.semantic_consistency import evaluate_semantic_consistency


CSV_PATH = ROOT / "data/benchmarks/dev_s1_consistency_n5.csv"
DATASET_PATH = ROOT / "data/benchmarks/dev_dataset.json"
CONSISTENCY_N = 5


def format_segments(segments) -> str:
    if not segments:
        return "(no segments)"
    parts = []
    for seg in segments:
        txt = seg.text[:50] + "..." if len(seg.text) > 50 else seg.text
        parts.append(f"[{seg.segment_type}] {txt}")
    return " || ".join(parts)


def extract_contexts(segments) -> str:
    if not segments:
        return "(no context extracted)"
    unique = []
    for seg in segments:
        ctx = seg.context
        if ctx and ctx not in unique:
            unique.append(ctx)
    return " | ".join(unique) if unique else "(no context extracted)"


async def rerun_case(critic, classifier, suite_idx, suite_name, nightmare, test):
    description = test.get("description", "?")
    user_input = test.get("input", "")
    expected_safe = test.get("expected_safe")
    expected_node = test.get("expected_node")

    result = await critic.evaluate_intervention(user_input, nightmare_context=nightmare)

    cons_agreement = None
    cons_node = None
    cons_code = None
    cons_votes_json = None
    cons_n_runs = None
    cons_majority_matches_gt = None

    if result.segments:
        seg_agreements = []
        for segment in result.segments:
            if not segment.node_id or segment.node_id == "UNCLASSIFIED":
                continue
            seg_context = segment.context or nightmare
            seg_cons = await evaluate_semantic_consistency(
                classifier=classifier,
                user_text=segment.text,
                context=seg_context or None,
                n_runs=CONSISTENCY_N,
                stochastic_temperature=0.7,
                confidence_threshold=0.8,
            )
            seg_agreements.append(seg_cons)

        if seg_agreements:
            min_cons = min(seg_agreements, key=lambda c: c.agreement_ratio)
            cons_agreement = min_cons.agreement_ratio
            cons_node = min_cons.node_id
            cons_code = min_cons.code
            cons_votes_json = json.dumps({
                f"seg{i}": c.vote_distribution for i, c in enumerate(seg_agreements)
            })
            cons_n_runs = min_cons.n_runs
            cons_majority_matches_gt = (min_cons.node_id == expected_node) if expected_node is not None else None

    is_safety_correct = (result.is_safe == expected_safe) if expected_safe is not None else None
    is_node_correct = (result.node_id == expected_node) if expected_node is not None else None

    suite_short = suite_name[:40] + "..." if len(suite_name) > 40 else suite_name
    row = {
        "suite_id": suite_idx,
        "suite": suite_short,
        "test": description,
        "user_input": user_input,
        "gt_safe": expected_safe,
        "gt_node": expected_node,
        "pred_safe": result.is_safe,
        "pred_node": result.node_id,
        "conf": result.confidence,
        "severity": result.severity if not result.is_safe else "",
        "safe_ok": is_safety_correct,
        "node_ok": is_node_correct,
        "contexts": extract_contexts(result.segments),
        "segments": format_segments(result.segments),
        "agreement_ratio": cons_agreement,
        "consistency_node": cons_node,
        "consistency_code": cons_code,
        "vote_distribution": cons_votes_json,
        "n_runs": cons_n_runs,
        "majority_vote_matches_gt": cons_majority_matches_gt,
    }
    return row


async def main():
    df = pd.read_csv(CSV_PATH)
    err_mask = df["pred_node"] == "ERROR"
    print(f"Found {err_mask.sum()} error rows to top up.")
    print()

    suites = json.loads(DATASET_PATH.read_text())

    classifier = ClinicalClassifier()
    critic = SafetyCritic(classifier=classifier)

    done = 0
    for csv_idx in df[err_mask].index:
        row = df.loc[csv_idx]
        suite_id = int(row["suite_id"])
        test_desc = row["test"]
        suite = suites[suite_id - 1]
        suite_name = suite.get("suite", "")
        nightmare = suite.get("nightmare_context", "") or suite.get("nightmare", "")
        test = next((t for t in suite.get("tests", []) if t.get("description") == test_desc), None)
        if test is None:
            print(f"  WARN: could not find test '{test_desc}' in suite {suite_id}")
            continue

        try:
            new_row = await rerun_case(critic, classifier, suite_id, suite_name, nightmare, test)
            for col, val in new_row.items():
                df.at[csv_idx, col] = val
            done += 1
            print(f"  [{done}/{err_mask.sum()}] {test_desc}: pred_node={new_row['pred_node']} "
                  f"agreement={new_row['agreement_ratio']:.0%}" if new_row['agreement_ratio'] is not None
                  else f"  [{done}/{err_mask.sum()}] {test_desc}: pred_node={new_row['pred_node']} (no consistency)")
        except Exception as e:
            print(f"  FAIL [{done+1}/{err_mask.sum()}] {test_desc}: {e}")

    df.to_csv(CSV_PATH, index=False)
    print()
    print(f"Patched {done} rows. CSV saved to {CSV_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
