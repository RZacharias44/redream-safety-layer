#!/usr/bin/env python3
"""
Benchmark Evaluation Script for Safety Critic

Loads a hierarchical JSON dataset (like contextual_batch_tests.json format),
runs the Safety Critic against every test case, and generates a detailed CSV report.

Usage:
    python run_benchmark.py                              # Uses default dev_dataset.json
    python run_benchmark.py --input data/benchmarks/test_dataset.json
    python run_benchmark.py --input data/benchmarks/dev_dataset.json --output data/benchmarks/benchmark.csv
    python run_benchmark.py --classifier s2 --consistency 5  # Full consistency run
"""

import os
import sys
import argparse
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from safety.critic import SafetyCritic
from safety.stage_merge_critic import StageMergeCritic
from safety.classifier import ClinicalClassifier
from safety.classifier_cot import ClinicalClassifierCoT
from safety.semantic_consistency import evaluate_semantic_consistency



def format_segments(segments) -> str:
    """Format segment list into a readable summary string."""
    if not segments:
        return ""
    
    parts = []
    for seg in segments:
        seg_type = seg.segment_type
        node = seg.node_id or "?"
        # Truncate text for readability
        text = seg.text[:40] + "..." if len(seg.text) > 40 else seg.text
        conf = f"{seg.confidence:.0%}" if seg.confidence else "?"
        # Include context if present
        ctx = f" ctx:{seg.context[:25]}..." if seg.context and len(seg.context) > 25 else (f" ctx:{seg.context}" if seg.context else "")
        parts.append(f"[{seg_type}]{ctx} \"{text}\" → {node} ({conf})")
    
    return " | ".join(parts)


def serialize_segments(segments) -> str:
    """Serialize segment list as JSON for post-hoc per-segment analysis.

    Captures all SegmentResult fields (text, segment_type, context, node_id,
    node_type, category, confidence, mean_logprob) so downstream analysis can
    compute per-segment metrics (e.g. secondary-node recall, any-maladaptive
    recall) without parsing the human-readable format_segments string.
    """
    if not segments:
        return "[]"

    serialized = []
    for seg in segments:
        if hasattr(seg, "to_dict"):
            serialized.append(seg.to_dict())
        else:
            # SegmentResult is a dataclass — __dict__ gives all fields
            serialized.append(dict(seg.__dict__))

    return json.dumps(serialized, ensure_ascii=False)


def extract_contexts(segments) -> str:
    """Extract just the contexts from segments for a dedicated column."""
    if not segments:
        return ""
    
    contexts = []
    for seg in segments:
        if seg.context:
            contexts.append(seg.context)
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for ctx in contexts:
        if ctx not in seen:
            seen.add(ctx)
            unique.append(ctx)
    
    return " | ".join(unique) if unique else "(no context extracted)"


async def run_benchmark(input_file: Path, output_file: Path, limit_suites: Optional[int] = None, classifier_mode: str = "s1", use_context_examples: bool = False, consistency_n: int = 0, skip_segmentation: bool = False, skip_meta_filter: bool = False, naive_segmentation: bool = False, critic_mode: str = "structured") -> pd.DataFrame:
    """
    Run the benchmark evaluation on a dataset.

    Args:
        input_file: Path to the JSON dataset file
        output_file: Path to save the CSV results
        limit_suites: Only process first N suites
        classifier_mode: "s1" or "s2"
        use_context_examples: Enable context-aware few-shot examples
        consistency_n: If > 0, run semantic consistency with this many stochastic runs
            per test case instead of the full SafetyCritic pipeline.
        skip_segmentation: Ablation B (ablation_spec.md §B.1). Bypass the LLM
            segmenter; whole input forced as one ACTION segment. Single-shot
            only (do not combine with --consistency, which is out of ablation
            scope per spec §"Run mode").
        skip_meta_filter: Cell B (ablation_spec.md §"Cell B"). Keep META
            segments (no filtering); also implicitly disables the META sanity
            check (documented coupling). Single-shot only.
        naive_segmentation: Cell C (ablation_spec.md §"Cell C"). Replace the
            LLM segmenter with rule-based sentence splitting; all-ACTION
            segments → META filter inert. Single-shot only. Mutually
            exclusive with skip_segmentation.
        critic_mode: "structured" (production SafetyCritic, default) or
            "merged" = Ablation A (ablation_spec.md §"Ablation A"):
            StageMergeCritic — one LLM call segments+classifies, post-hoc
            triage. Single-shot S1 only; ignores the other ablation flags
            (it has no separate segmenter/META filter).

    Returns:
        DataFrame with all results
    """

    # Load dataset
    print("=" * 70)
    print("SAFETY CRITIC BENCHMARK EVALUATION")
    if consistency_n > 0:
        print(f"  MODE: Semantic Consistency (N={consistency_n})")
    print("=" * 70)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print()

    print("Loading dataset...")
    with open(input_file, 'r', encoding='utf-8') as f:
        suites = json.load(f)

    # Apply limit if specified
    if limit_suites:
        suites = suites[:limit_suites]
        print(f"✓ Limited to first {limit_suites} suite(s)")

    total_tests = sum(len(suite.get("tests", [])) for suite in suites)
    print(f"✓ Loaded {len(suites)} suites with {total_tests} total tests")
    print()

    # Initialize SafetyCritic with selected classifier
    ctx_label = " + context examples" if use_context_examples else ""
    print(f"Initializing SafetyCritic (classifier: {classifier_mode}{ctx_label})...")
    if classifier_mode == "s2":
        classifier = ClinicalClassifierCoT(use_context_examples=use_context_examples)
        print(f"  Using System 2 (CoT) classifier: {classifier.model_config.name}")
    else:
        classifier = ClinicalClassifier(use_context_examples=use_context_examples)
        print(f"  Using System 1 (Direct) classifier: {classifier.model_config.name}")
    if use_context_examples:
        print(f"  Context-aware examples: ENABLED")
    if consistency_n > 0:
        print(f"  Semantic consistency: N={consistency_n}, temp=0.7 (per-segment after pipeline)")
    if skip_segmentation:
        print("  ⚠ Ablation B: segmentation DISABLED (whole input → single ACTION segment)")
    if skip_meta_filter:
        print("  ⚠ Cell B: META filter DISABLED (META segments classified + sanity check off)")
    if naive_segmentation:
        print("  ⚠ Cell C: NAIVE sentence-splitter segmenter (no LLM, no META by absence)")
    if critic_mode == "merged":
        print("  ⚠ Ablation A: STAGE-MERGE critic (1 LLM call segments+classifies; "
              "post-hoc triage; fail-closed JSON)")
        critic = StageMergeCritic()
    else:
        critic = SafetyCritic(classifier=classifier, skip_segmentation=skip_segmentation,
                              skip_meta_filter=skip_meta_filter,
                              naive_segmentation=naive_segmentation)
    print("✓ Critic initialized")
    print()
    
    # Run evaluation
    results = []
    start_time = datetime.now()
    test_counter = 0
    
    for suite_idx, suite in enumerate(suites, 1):
        suite_name = suite.get("suite", f"Suite {suite_idx}")
        nightmare = suite.get("nightmare", "")
        tests = suite.get("tests", [])
        
        print(f"Processing Suite {suite_idx}/{len(suites)}: \"{suite_name[:50]}...\"")
        print(f"  Nightmare: \"{nightmare[:60]}...\"")
        print(f"  Tests: {len(tests)}")
        
        for test_idx, test in enumerate(tests, 1):
            test_counter += 1
            description = test.get("description", f"Test {test_idx}")
            user_input = test.get("input", "")
            expected_safe = test.get("expected_safe")
            expected_node = test.get("expected_node")  # May be None

            try:
                # --- Always run the full SafetyCritic pipeline first ---
                result = await critic.evaluate_intervention(
                    user_input,
                    nightmare_context=nightmare
                )
                predicted_safe = result.is_safe
                predicted_node = result.node_id
                confidence_score = result.confidence
                severity = result.severity
                segments_formatted = format_segments(result.segments)
                segments_serialized = serialize_segments(result.segments)
                contexts_extracted = extract_contexts(result.segments)

                # --- Consistency mode: run per-segment consistency after pipeline ---
                # Approach: segment once (via SafetyCritic above), then run N stochastic
                # classification passes on each segment. This isolates classifier
                # uncertainty from segmentation variation while testing the same text
                # the classifier actually sees in production.
                cons_agreement = None
                cons_node = None
                cons_code = None
                cons_votes = None
                cons_n_runs = None
                cons_majority_matches_gt = None
                cons_latency_ms = 0.0
                cons_input_tokens = 0
                cons_output_tokens = 0

                if consistency_n > 0 and result.segments:
                    # Run consistency on each classified segment
                    seg_agreements = []
                    for segment in result.segments:
                        if not segment.node_id or segment.node_id == "UNCLASSIFIED":
                            continue
                        seg_context = segment.context or nightmare
                        seg_cons = await evaluate_semantic_consistency(
                            classifier=classifier,
                            user_text=segment.text,
                            context=seg_context or None,
                            n_runs=consistency_n,
                            stochastic_temperature=0.7,
                            confidence_threshold=0.8,
                        )
                        seg_agreements.append(seg_cons)
                        cons_latency_ms += seg_cons.total_latency_ms
                        cons_input_tokens += seg_cons.total_input_tokens
                        cons_output_tokens += seg_cons.total_output_tokens

                    if seg_agreements:
                        # Use minimum agreement across segments (conservative: weakest
                        # link determines uncertainty). The primary segment's consistency
                        # is reported for node/code/votes.
                        min_cons = min(seg_agreements, key=lambda c: c.agreement_ratio)
                        cons_agreement = min_cons.agreement_ratio
                        cons_node = min_cons.node_id
                        cons_code = min_cons.code
                        # Aggregate vote distributions across all segments
                        cons_votes = json.dumps({
                            f"seg{i}": c.vote_distribution
                            for i, c in enumerate(seg_agreements)
                        })
                        cons_n_runs = min_cons.n_runs
                        cons_majority_matches_gt = (min_cons.node_id == expected_node) if expected_node is not None else None

                # Calculate correctness (shared by both modes)
                is_safety_correct = (predicted_safe == expected_safe) if expected_safe is not None else None
                if expected_node is not None:
                    is_node_correct = (predicted_node == expected_node)
                else:
                    is_node_correct = None  # NaN in pandas

                # Store result (use short suite ID instead of full name)
                suite_short = suite_name[:40] + "..." if len(suite_name) > 40 else suite_name
                row = {
                    "suite_id": suite_idx,
                    "suite": suite_short,
                    "test": description,
                    "user_input": user_input,
                    "gt_safe": expected_safe,
                    "gt_node": expected_node,
                    "pred_safe": predicted_safe,
                    "pred_node": predicted_node,
                    "conf": confidence_score,
                    "severity": severity if not predicted_safe else "",
                    "safe_ok": is_safety_correct,
                    "node_ok": is_node_correct,
                    "contexts": contexts_extracted,
                    "segments": segments_formatted,
                    "segments_json": segments_serialized,
                    "parse_failure": getattr(result, "parse_failure", False),
                }

                if consistency_n > 0:
                    row.update({
                        "agreement_ratio": cons_agreement,
                        "consistency_node": cons_node,
                        "consistency_code": cons_code,
                        "vote_distribution": cons_votes,
                        "n_runs": cons_n_runs,
                        "majority_vote_matches_gt": cons_majority_matches_gt,
                    })

                results.append(row)

                # Progress output
                if consistency_n > 0:
                    agreement_str = f"{cons_agreement:.0%}" if cons_agreement is not None else "n/a"
                    latency_s = cons_latency_ms / 1000
                    tok_str = f"{cons_input_tokens}+{cons_output_tokens} tok"
                    status = "✓" if is_safety_correct else "✗"
                    print(f"    {status} [{test_counter}/{total_tests}] {description}")
                    print(f"       agree={agreement_str}  {latency_s:.1f}s  {tok_str}")
                else:
                    status = "✓" if is_safety_correct else "✗"
                    print(f"    {status} [{test_counter}/{total_tests}] {description}")

            except Exception as e:
                print(f"    ✗ [{test_counter}/{total_tests}] {description} - ERROR: {e}")
                suite_short = suite_name[:40] + "..." if len(suite_name) > 40 else suite_name
                row = {
                    "suite_id": suite_idx,
                    "suite": suite_short,
                    "test": description,
                    "user_input": user_input,
                    "gt_safe": expected_safe,
                    "gt_node": expected_node,
                    "pred_safe": None,
                    "pred_node": "ERROR",
                    "conf": None,
                    "severity": "ERROR",
                    "safe_ok": False,
                    "node_ok": False,
                    "contexts": "ERROR",
                    "segments": f"ERROR: {str(e)}",
                    "segments_json": "[]",
                    "parse_failure": False,
                }
                if consistency_n > 0:
                    row.update({
                        "agreement_ratio": None,
                        "consistency_node": "ERROR",
                        "consistency_code": "ERROR",
                        "vote_distribution": "{}",
                        "n_runs": consistency_n,
                        "majority_vote_matches_gt": False,
                    })
                results.append(row)
        
        print()
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✓ Results saved to: {output_file}")
    
    # Calculate metrics
    elapsed = datetime.now() - start_time
    
    print()
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {len(df)}")
    print(f"Time Elapsed: {elapsed}")
    print()
    
    # Safety accuracy
    safety_correct = df["safe_ok"].sum()
    safety_total = df["safe_ok"].notna().sum()
    safety_accuracy = (safety_correct / safety_total * 100) if safety_total > 0 else 0
    print(f"Safety Accuracy: {safety_correct}/{safety_total} ({safety_accuracy:.1f}%)")
    
    # Node accuracy (only where expected_node was provided)
    node_df = df[df["gt_node"].notna()]
    if len(node_df) > 0:
        node_correct = node_df["node_ok"].sum()
        node_total = len(node_df)
        node_accuracy = (node_correct / node_total * 100) if node_total > 0 else 0
        print(f"Node Accuracy (strict): {node_correct}/{node_total} ({node_accuracy:.1f}%)")
    else:
        print("Node Accuracy: N/A (no expected_node labels)")
    
    # Average confidence
    avg_confidence = df["conf"].mean()
    if pd.notna(avg_confidence):
        print(f"Average Confidence: {avg_confidence*100:.1f}%")
    
    print()
    
    # Breakdown by ground truth
    print("Breakdown by Ground Truth:")
    safe_tests = df[df["gt_safe"] == True]
    unsafe_tests = df[df["gt_safe"] == False]
    
    if len(safe_tests) > 0:
        safe_correct = safe_tests["safe_ok"].sum()
        print(f"  Safe (expected): {safe_correct}/{len(safe_tests)} correct ({safe_correct/len(safe_tests)*100:.1f}%)")
    
    if len(unsafe_tests) > 0:
        unsafe_correct = unsafe_tests["safe_ok"].sum()
        print(f"  Unsafe (expected): {unsafe_correct}/{len(unsafe_tests)} correct ({unsafe_correct/len(unsafe_tests)*100:.1f}%)")
    
    print()
    
    # Breakdown by node type (where applicable)
    if len(node_df) > 0:
        print("Breakdown by Expected Node:")
        for node in node_df["gt_node"].unique():
            node_subset = node_df[node_df["gt_node"] == node]
            correct = node_subset["node_ok"].sum()
            total = len(node_subset)
            print(f"  {node}: {correct}/{total} ({correct/total*100:.1f}%)")
    
    print()
    
    # Confusion matrix for safety
    print("Safety Confusion Matrix:")
    print("                    Predicted Safe    Predicted Unsafe")
    
    tp = len(df[(df["gt_safe"] == True) & (df["pred_safe"] == True)])
    fp = len(df[(df["gt_safe"] == False) & (df["pred_safe"] == True)])
    tn = len(df[(df["gt_safe"] == False) & (df["pred_safe"] == False)])
    fn = len(df[(df["gt_safe"] == True) & (df["pred_safe"] == False)])
    
    print(f"  Actually Safe:    {tp:>10}          {fn:>10}")
    print(f"  Actually Unsafe:  {fp:>10}          {tn:>10}")
    
    # Precision, Recall, F1 for "Unsafe" class (more clinically important)
    print()
    print("Unsafe Detection Metrics (clinically important):")
    precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"  Precision (Unsafe): {precision*100:.1f}%")
    print(f"  Recall (Unsafe): {recall*100:.1f}%")
    print(f"  F1 (Unsafe): {f1*100:.1f}%")
    
    print()
    print("=" * 70)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run Safety Critic benchmark evaluation on a dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_benchmark.py
    python run_benchmark.py --input data/benchmarks/test_dataset.json
    python run_benchmark.py --input data/benchmarks/dev_dataset.json --output data/benchmarks/dev_results.csv
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        default='data/benchmarks/dev_dataset.json',
        help="Path to input JSON dataset (default: data/benchmarks/dev_dataset.json)"
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/benchmarks/dev_s1_baseline.csv',
        help="Path to output CSV results (default: data/benchmarks/dev_s1_baseline.csv)"
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help="Limit to first N suites (for quick testing)"
    )

    parser.add_argument(
        '--classifier',
        type=str,
        choices=["s1", "s2"],
        default="s1",
        help="Classifier mode: s1 (System 1 / Direct) or s2 (System 2 / CoT)"
    )

    parser.add_argument(
        '--context-examples',
        action='store_true',
        default=False,
        help="Enable context-aware few-shot examples in the classifier prompt"
    )

    parser.add_argument(
        '--consistency',
        type=int,
        default=0,
        metavar='N',
        help="Run semantic consistency UQ with N stochastic runs per case (default: 0 = off). "
             "When > 0, bypasses SafetyCritic segmentation and evaluates the raw classifier "
             "directly on each test input (matching run_consistency_errors.py)."
    )

    parser.add_argument(
        '--no-segment',
        action='store_true',
        default=False,
        help="Ablation B (ablation_spec.md §B.1): disable the LLM segmenter; "
             "force the whole input as one ACTION segment. Single-shot only — "
             "do not combine with --consistency (out of ablation scope)."
    )

    parser.add_argument(
        '--no-meta-filter',
        action='store_true',
        default=False,
        help="Cell B (ablation_spec.md §'Cell B'): keep META segments (no "
             "filtering); also disables the META sanity check (documented "
             "coupling). Single-shot only — do not combine with --consistency."
    )

    parser.add_argument(
        '--segmenter',
        type=str,
        choices=["llm", "naive"],
        default="llm",
        help="Segmenter: 'llm' (production LLM segmenter, default) or 'naive' "
             "= Cell C (ablation_spec.md §'Cell C'): rule-based sentence "
             "splitting, all-ACTION segments, META inert by absence. "
             "Single-shot only; mutually exclusive with --no-segment."
    )

    parser.add_argument(
        '--critic',
        type=str,
        choices=["structured", "merged"],
        default="structured",
        help="Critic: 'structured' (production SafetyCritic, default) or "
             "'merged' = Ablation A (ablation_spec.md §'Ablation A'): "
             "StageMergeCritic, one LLM call segments+classifies, post-hoc "
             "triage, fail-closed JSON. Single-shot S1 only; not combinable "
             "with the other ablation flags or --classifier s2."
    )

    args = parser.parse_args()

    naive_seg = (args.segmenter == "naive")
    merged = (args.critic == "merged")

    if (args.no_segment or args.no_meta_filter or naive_seg or merged) and args.consistency > 0:
        print("Error: ablation conditions are single-shot only (out of UQ "
              "scope, ablation_spec.md §'Run mode'). Drop --consistency.")
        sys.exit(1)

    if naive_seg and args.no_segment:
        print("Error: --segmenter naive and --no-segment are mutually exclusive "
              "(Cell C vs Ablation B are distinct conditions, ablation_spec.md).")
        sys.exit(1)

    if merged and (args.no_segment or args.no_meta_filter or naive_seg):
        print("Error: --critic merged (Ablation A) is a self-contained critic; "
              "it cannot be combined with --no-segment/--no-meta-filter/"
              "--segmenter naive (those are SafetyCritic flags).")
        sys.exit(1)

    if merged and args.classifier != "s1":
        print("Error: Ablation A is locked to S1 (ablation_spec.md "
              "§'Configuration'). Use --classifier s1 (default).")
        sys.exit(1)
    
    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    input_file = project_root / args.input
    output_file = project_root / args.output
    
    # Check input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print(f"Available files in data/:")
        data_dir = project_root / "data"
        if data_dir.exists():
            for f in data_dir.iterdir():
                print(f"  - {f.name}")
        sys.exit(1)
    
    # Run benchmark
    asyncio.run(run_benchmark(input_file, output_file, limit_suites=args.limit, classifier_mode=args.classifier, use_context_examples=args.context_examples, consistency_n=args.consistency, skip_segmentation=args.no_segment, skip_meta_filter=args.no_meta_filter, naive_segmentation=naive_seg, critic_mode=args.critic))


if __name__ == "__main__":
    main()

