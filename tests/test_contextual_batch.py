"""
Test runner for contextual batch tests.

This script runs all contextual tests (with nightmare context) from
data/scenarios/contextual_batch_tests.json and provides detailed results.
"""

import asyncio
import json
import sys
import os
from typing import Dict, List
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Add parent directory to path (go up from testing/ to code/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety.critic import SafetyCritic


async def run_contextual_batch_tests():
    """Run all contextual batch tests and report results."""
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load test data
    test_file = os.path.join(project_root, "data", "scenarios", "contextual_batch_tests.json")
    with open(test_file, 'r') as f:
        test_suites = json.load(f)
    
    # Initialize critic
    print("=" * 80)
    print("CONTEXTUAL BATCH TEST RUNNER")
    print("=" * 80)
    print(f"\nInitializing SafetyCritic...")
    critic = SafetyCritic()
    
    # Track overall statistics
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    results_by_suite = {}
    
    # Run each test suite
    for suite in test_suites:
        suite_name = suite["suite"]
        nightmare = suite["nightmare"]
        tests = suite["tests"]
        
        print(f"\n{'=' * 80}")
        print(f"SUITE: {suite_name}")
        print(f"{'=' * 80}")
        print(f"Nightmare Context: {nightmare[:100]}...")
        print(f"Tests: {len(tests)}")
        print()
        
        suite_passed = 0
        suite_failed = 0
        suite_details = []
        
        for i, test in enumerate(tests, 1):
            total_tests += 1
            description = test["description"]
            user_input = test["input"]
            expected_safe = test["expected_safe"]
            expected_node = test.get("expected_node")
            
            try:
                # Run evaluation with nightmare context
                result = await critic.evaluate_intervention(user_input, nightmare_context=nightmare)
                
                # Check if result matches expected
                safety_match = result.is_safe == expected_safe
                node_match = True
                if expected_node and not result.is_safe:
                    node_match = result.node_id == expected_node
                
                test_passed = safety_match and node_match
                
                if test_passed:
                    status = "✅ PASS"
                    passed_tests += 1
                    suite_passed += 1
                else:
                    status = "❌ FAIL"
                    failed_tests += 1
                    suite_failed += 1
                
                # Format output
                severity_str = f" [{result.severity}]" if result.severity else ""
                safety_str = "SAFE" if result.is_safe else f"UNSAFE{severity_str}"
                expected_str = "SAFE" if expected_safe else "UNSAFE"
                
                # Format confidence as percentage
                conf_str = f" | conf: {result.confidence*100:.1f}%" if result.confidence is not None else ""
                logprob_str = f" (logp: {result.mean_logprob:.3f})" if result.mean_logprob is not None else ""
                
                print(f"{status} Test {i}: {description}")
                print(f"  Input: \"{user_input[:60]}...\"" if len(user_input) > 60 else f"  Input: \"{user_input}\"")
                print(f"  Result: {safety_str} → {result.node_id}{conf_str}{logprob_str}")
                
                if expected_node:
                    print(f"  Expected: {expected_str} → {expected_node}")
                else:
                    print(f"  Expected: {expected_str}")
                
                if not test_passed:
                    if not safety_match:
                        print(f"  ⚠️  Safety mismatch!")
                    if not node_match:
                        print(f"  ⚠️  Node mismatch! Expected: {expected_node}")
                
                # Show per-segment confidence if multiple segments
                if len(result.segments) > 1:
                    print(f"  Segments ({len(result.segments)}):")
                    for seg in result.segments:
                        seg_conf = f" [{seg.confidence*100:.1f}%]" if seg.confidence is not None else ""
                        seg_text = seg.text[:40] + "..." if len(seg.text) > 40 else seg.text
                        print(f"    • {seg.segment_type}: \"{seg_text}\" → {seg.node_id}{seg_conf}")
                
                # Store detailed results
                suite_details.append({
                    "description": description,
                    "input": user_input,
                    "expected_safe": expected_safe,
                    "expected_node": expected_node,
                    "actual_safe": result.is_safe,
                    "actual_node": result.node_id,
                    "passed": test_passed,
                    "segments": len(result.segments),
                    "confidence": result.confidence,
                    "mean_logprob": result.mean_logprob,
                    "segment_confidences": [
                        {"text": s.text, "node_id": s.node_id, "confidence": s.confidence, "mean_logprob": s.mean_logprob}
                        for s in result.segments if s.node_id
                    ]
                })
                
            except Exception as e:
                status = "💥 ERROR"
                failed_tests += 1
                suite_failed += 1
                print(f"{status} Test {i}: {description}")
                print(f"  Error: {str(e)}")
                suite_details.append({
                    "description": description,
                    "input": user_input,
                    "error": str(e),
                    "passed": False
                })
        
        # Calculate average confidence for the suite
        confidences = [d.get("confidence") for d in suite_details if d.get("confidence") is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        
        # Store suite results
        results_by_suite[suite_name] = {
            "total": len(tests),
            "passed": suite_passed,
            "failed": suite_failed,
            "pass_rate": (suite_passed / len(tests) * 100) if len(tests) > 0 else 0,
            "avg_confidence": avg_confidence,
            "details": suite_details
        }
        
        # Print suite summary
        conf_summary = f" | Avg Confidence: {avg_confidence*100:.1f}%" if avg_confidence else ""
        print(f"\nSuite Summary: {suite_passed}/{len(tests)} passed ({suite_passed/len(tests)*100:.1f}%){conf_summary}")
    
    # Calculate overall average confidence
    all_confidences = []
    for stats in results_by_suite.values():
        for detail in stats["details"]:
            if detail.get("confidence") is not None:
                all_confidences.append(detail["confidence"])
    overall_avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else None
    
    # Print overall summary
    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
    if overall_avg_conf:
        print(f"Average Confidence: {overall_avg_conf*100:.1f}%")
    print()
    
    # Print suite breakdown
    print("Suite Breakdown:")
    for suite_name, stats in results_by_suite.items():
        status_emoji = "✅" if stats["failed"] == 0 else "⚠️"
        conf_str = f" | conf: {stats['avg_confidence']*100:.1f}%" if stats.get('avg_confidence') else ""
        print(f"  {status_emoji} {suite_name}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1f}%){conf_str}")
    
    # Save detailed results to JSON
    output_file = os.path.join(project_root, f"data/results/test_results_contextual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "pass_rate": passed_tests/total_tests*100 if total_tests > 0 else 0,
            "avg_confidence": overall_avg_conf,
            "suites": results_by_suite
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Return exit code
    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_contextual_batch_tests())
    sys.exit(exit_code)
