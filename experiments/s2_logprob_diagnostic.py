"""
System 2 log-probability provenance diagnostic.

Question (docs/planning/HANDOFF_S2_LOGPROB_DIAGNOSTIC.md): System 2 computes its
classification confidence from the log-probabilities of the classification *code*
tokens. If it cannot locate that span it falls back to the mean log-probability of
the entire response — dominated by the fluency of the `context_summary` and
`clinical_rationale` prose. Nobody had measured how often that fallback fires, and
the saved benchmark CSVs record only the final `conf` value.

This script runs a stratified sample of development cases through the production
SafetyCritic pipeline with an instrumented System 2 classifier, and records for
every classification call:

  - which strategy produced the confidence ("key_scan", "tail_scan", "full_seq")
  - the code-token confidence actually used
  - the full-sequence confidence that the fallback would have produced
  - token counts for both spans

Usage:
    uv run python experiments/s2_logprob_diagnostic.py
    uv run python experiments/s2_logprob_diagnostic.py --n 50 --out data/benchmarks/s2_logprob_diagnostic.csv
"""

import argparse
import asyncio
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety.classifier_cot import ClinicalClassifierCoT
from safety.critic import SafetyCritic


def stratified_sample(suites: List[dict], n: int) -> List[dict]:
    """
    Pick n test cases spread as evenly as possible across expected_node.

    Round-robins over the node buckets so that rare nodes are represented, and
    walks each bucket in suite order so the sample also spans nightmare suites.
    """
    by_node = defaultdict(list)
    for suite_idx, suite in enumerate(suites, 1):
        for test in suite.get("tests", []):
            node = test.get("expected_node") or "(none)"
            by_node[node].append({
                "suite_id": suite_idx,
                "suite": suite.get("suite", ""),
                "nightmare": suite.get("nightmare", ""),
                "description": test.get("description", ""),
                "input": test.get("input", ""),
                "expected_safe": test.get("expected_safe"),
                "expected_node": test.get("expected_node"),
                "test_type": test.get("test_type"),
            })

    # Deterministic order: nodes alphabetically, cases in dataset order.
    buckets = [by_node[node] for node in sorted(by_node)]
    selected = []
    depth = 0
    while len(selected) < n and any(depth < len(b) for b in buckets):
        for bucket in buckets:
            if depth < len(bucket):
                selected.append(bucket[depth])
                if len(selected) == n:
                    break
        depth += 1
    return selected


async def run(input_file: Path, output_file: Path, n: int) -> pd.DataFrame:
    with open(input_file, "r", encoding="utf-8") as f:
        suites = json.load(f)

    cases = stratified_sample(suites, n)
    node_spread = Counter(c["expected_node"] or "(none)" for c in cases)
    print(f"✓ Sampled {len(cases)} cases across {len(node_spread)} expected nodes")
    for node, count in sorted(node_spread.items()):
        print(f"    {node:<24} {count}")
    print()

    classifier = ClinicalClassifierCoT()
    classifier.logprob_diagnostics = []
    critic = SafetyCritic(classifier=classifier)
    print(f"✓ System 2 classifier: {classifier.model_config.name}")
    print()

    rows = []
    for i, case in enumerate(cases, 1):
        before = len(classifier.logprob_diagnostics)
        try:
            result = await critic.evaluate_intervention(
                case["input"],
                nightmare_context=case["nightmare"],
            )
            pred_node = result.node_id
            pred_safe = result.is_safe
            error = ""
        except Exception as e:  # keep going; a dropped case is not a missing answer
            pred_node, pred_safe, error = None, None, str(e)

        calls = classifier.logprob_diagnostics[before:]
        for call_idx, call in enumerate(calls):
            rows.append({
                "case_idx": i,
                "suite_id": case["suite_id"],
                "test": case["description"],
                "gt_node": case["expected_node"],
                "pred_node": pred_node,
                "pred_safe": pred_safe,
                "call_idx": call_idx,
                "segment_text": call["user_text"],
                "code": call["code"],
                "seg_node": call["node_id"],
                "logprob_source": call["logprob_source"],
                "code_token_conf": call["code_token_conf"],
                "full_seq_conf": call["full_seq_conf"],
                "code_token_count": call["code_token_count"],
                "response_token_count": call["response_token_count"],
                "max_tokens": classifier.agent.max_tokens,
                "tail_tokens": json.dumps(call["tail_tokens"]),
                "error": error,
            })

        sources = ",".join(c["logprob_source"] or "?" for c in calls) or "(no calls)"
        status = "✗" if error else "✓"
        print(f"  {status} [{i}/{len(cases)}] {case['description'][:44]:<44} {sources}")

    df = pd.DataFrame(rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n✓ Wrote {len(df)} classification calls to {output_file}")
    return df


def summarize(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("LOGPROB PROVENANCE")
    print("=" * 70)

    total = len(df)
    if total == 0:
        print("No classification calls recorded.")
        return

    counts = df["logprob_source"].value_counts(dropna=False)
    for source, count in counts.items():
        print(f"  {str(source):<12} {count:>4} / {total}  ({count / total:.1%})")

    fallback = df[df["logprob_source"] == "full_seq"]
    print(f"\n  Fallback (full-response) rate: {len(fallback) / total:.1%}")

    # A truncated response would break JSON parsing and route to UNCLASSIFIED, so
    # confirm the generation limit is not being approached.
    limit = int(df["max_tokens"].iloc[0])
    longest = int(df["response_token_count"].max())
    at_limit = int((df["response_token_count"] >= limit).sum())
    print(f"\n  Response length: max {longest} tokens against a {limit}-token limit; "
          f"{at_limit} call(s) at the limit")

    hits = df[df["logprob_source"].isin(["span_scan", "key_scan", "tail_scan"])]
    if len(hits) > 0:
        gap = hits["code_token_conf"] - hits["full_seq_conf"]
        print("\n" + "=" * 70)
        print(f"CODE-TOKEN vs FULL-SEQUENCE CONFIDENCE (n={len(hits)} located spans)")
        print("=" * 70)
        print(f"  code-token conf   mean {hits['code_token_conf'].mean():.3f}   "
              f"median {hits['code_token_conf'].median():.3f}   "
              f"min {hits['code_token_conf'].min():.3f}   "
              f"max {hits['code_token_conf'].max():.3f}   "
              f"sd {hits['code_token_conf'].std():.3f}")
        print(f"  full-seq conf     mean {hits['full_seq_conf'].mean():.3f}   "
              f"median {hits['full_seq_conf'].median():.3f}   "
              f"min {hits['full_seq_conf'].min():.3f}   "
              f"max {hits['full_seq_conf'].max():.3f}   "
              f"sd {hits['full_seq_conf'].std():.3f}")
        print(f"  gap (code - full) mean {gap.mean():+.3f}   median {gap.median():+.3f}")
        in_band = hits["code_token_conf"].between(0.70, 0.85, inclusive="left").mean()
        print(f"\n  code-token conf inside [0.70, 0.85): {in_band:.1%}")
        full_band = hits["full_seq_conf"].between(0.70, 0.85, inclusive="left").mean()
        print(f"  full-seq  conf inside [0.70, 0.85): {full_band:.1%}")
        print(f"  code-token conf above 0.95:          "
              f"{(hits['code_token_conf'] > 0.95).mean():.1%}")
        print(f"  median tokens in code span: {hits['code_token_count'].median():.0f}   "
              f"median tokens in response: {hits['response_token_count'].median():.0f}")

    print("\n" + "=" * 70)
    print("FALLBACK RATE BY PREDICTED CODE")
    print("=" * 70)
    by_code = df.groupby("code").agg(
        n=("code", "size"),
        fallback=("logprob_source", lambda s: (s == "full_seq").sum()),
        mean_conf=("code_token_conf", "mean"),
    )
    by_code["rate"] = by_code["fallback"] / by_code["n"]
    for code, r in by_code.sort_values("rate", ascending=False).iterrows():
        print(f"  {str(code):<12} {int(r['fallback']):>3}/{int(r['n']):<3} "
              f"({r['rate']:>6.1%})   mean conf used {r['mean_conf']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("data/benchmarks/dev_dataset.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/benchmarks/s2_logprob_diagnostic.csv"))
    parser.add_argument("--n", type=int, default=40,
                        help="Number of development cases to sample (default 40)")
    args = parser.parse_args()

    df = asyncio.run(run(args.input, args.out, args.n))
    summarize(df)


if __name__ == "__main__":
    main()
