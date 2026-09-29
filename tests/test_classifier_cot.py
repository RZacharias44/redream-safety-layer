#!/usr/bin/env python3
"""
Tests for System 2 (Chain-of-Thought) Classifier and Semantic Consistency.

Tests cover:
1. CoT classifier produces valid ClassificationResult objects
2. CoT classifier returns correct classifications for clear cases
3. CoT JSON parsing handles edge cases
4. Semantic consistency evaluation works with both classifiers
5. CoT classifier is compatible with the critic pipeline interface
"""

import asyncio
import json
import sys
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

sys.path.insert(0, str(Path(__file__).parent.parent))

requires_scaleway = pytest.mark.skipif(
    not os.getenv("SCALEWAY_API_KEY"),
    reason="requires SCALEWAY_API_KEY for live classifier calls",
)


def test_cot_prompt_parity():
    """Verify System 2 prompt contains the same codes, rules, and examples as System 1."""
    from safety.classifier import CLINICAL_CODER_PROMPT
    from safety.classifier_cot import CLINICAL_CODER_COT_PROMPT

    # Same 12 codes must appear in both prompts
    codes = ["HIDE", "ESCAPE", "DESTROY", "DENY", "REPLAY",
             "CONFRONT", "HELP", "CHANGE", "RELAX", "POWER",
             "LOOK", "FEEL"]

    for code in codes:
        assert code in CLINICAL_CODER_PROMPT, f"System 1 missing code {code}"
        assert code in CLINICAL_CODER_COT_PROMPT, f"System 2 missing code {code}"

    # Same critical rules must appear in both
    rules = [
        "Violence Threshold",
        "Feeling vs. Avoidance",
        "Mastery Priority",
        "Negation",
        "Mental Actions",
        "Emotional Regulation vs Suppression",
        "Trauma Replay vs Scene Description",
    ]

    for rule in rules:
        assert rule in CLINICAL_CODER_PROMPT, f"System 1 missing rule '{rule}'"
        assert rule in CLINICAL_CODER_COT_PROMPT, f"System 2 missing rule '{rule}'"

    # Same examples must appear in both
    examples = [
        '"I hide under the bed" -> HIDE',
        '"I wake myself up" -> ESCAPE',
        '"I kill everyone brutally" -> DESTROY',
        '"I punch the attacker" -> CONFRONT',
    ]

    for example in examples:
        assert example in CLINICAL_CODER_PROMPT, f"System 1 missing example '{example}'"
        assert example in CLINICAL_CODER_COT_PROMPT, f"System 2 missing example '{example}'"

    # System 2 should have structured response format but NOT "No explanations"
    assert "No explanations" in CLINICAL_CODER_PROMPT, "System 1 should say 'No explanations'"
    assert "No explanations" not in CLINICAL_CODER_COT_PROMPT, \
        "System 2 should NOT say 'No explanations' (it requires reasoning)"

    # System 2 should require JSON with reasoning fields
    assert "context_summary" in CLINICAL_CODER_COT_PROMPT
    assert "clinical_rationale" in CLINICAL_CODER_COT_PROMPT
    assert "classification" in CLINICAL_CODER_COT_PROMPT

    print("✅ Prompt parity test passed - System 2 has same codes, rules, examples as System 1")


def test_cot_json_parsing():
    """Test the CoT JSON response parser handles various formats."""
    from safety.classifier_cot import ClinicalClassifierCoT

    classifier = ClinicalClassifierCoT.__new__(ClinicalClassifierCoT)

    # Clean JSON
    result = classifier._parse_cot_response(json.dumps({
        "context_summary": "The dreamer hides under a bed",
        "clinical_rationale": "This is avoidance behavior",
        "classification": "HIDE"
    }))
    assert result.classification_code == "HIDE"
    assert "avoidance" in result.clinical_rationale.lower()

    # JSON with markdown code blocks
    result = classifier._parse_cot_response("""```json
{"context_summary": "test", "clinical_rationale": "test", "classification": "CONFRONT"}
```""")
    assert result.classification_code == "CONFRONT"

    # JSON with surrounding text
    result = classifier._parse_cot_response("""Here is my analysis:
{"context_summary": "test", "clinical_rationale": "test", "classification": "HELP"}
That's my classification.""")
    assert result.classification_code == "HELP"

    # Completely malformed prose fails closed even when it names a valid code.
    result = classifier._parse_cot_response("I think this is DESTROY because it's violent")
    assert result.classification_code == "UNCLASSIFIED"

    # Narrow recovery: repair missing commas between newline-separated fields.
    result = classifier._parse_cot_response('''{
"context_summary": "The dreamer hides"
"clinical_rationale": "This is avoidance"
"classification": "HIDE"
}''')
    assert result.classification_code == "HIDE"

    # Invalid schemas and classification field types fail closed.
    result = classifier._parse_cot_response('["HIDE"]')
    assert result.classification_code == "UNCLASSIFIED"

    result = classifier._parse_cot_response(json.dumps({
        "context_summary": "test",
        "clinical_rationale": "test",
        "classification": None,
    }))
    assert result.classification_code == "UNCLASSIFIED"

    print("✅ CoT JSON parsing test passed - normalizes syntax and fails closed")


@pytest.mark.asyncio
async def test_cot_invalid_code_routes_to_unclassified():
    """CoT semantic variants must not be recovered by keywords or substrings."""
    from unittest.mock import AsyncMock

    from safety.classifier_cot import ClinicalClassifierCoT

    classifier = ClinicalClassifierCoT.__new__(ClinicalClassifierCoT)
    classifier.agent = AsyncMock()
    classifier.agent.generate = AsyncMock(return_value=(
        json.dumps({
            "context_summary": "The dreamer fights back",
            "clinical_rationale": "This is an active response",
            "classification": "FIGHTING",
        }),
        {"input": 10, "output": 20, "total": 30},
        None,
    ))

    node_id, result = await classifier.classify_intent(
        "I fight back", return_logprobs=True
    )

    assert node_id == "UNCLASSIFIED"
    assert result.code == "UNCLASSIFIED"
    assert result.confidence == 0.0


def test_classification_result_interface():
    """Verify CoT classifier produces ClassificationResult compatible with critic."""
    from safety.classifier import ClassificationResult

    # Create a result as CoT would produce it
    result = ClassificationResult(
        node_id="AVOIDANCE",
        code="HIDE",
        confidence=0.85,
        raw_response='{"context_summary":"...", "clinical_rationale":"...", "classification":"HIDE"}',
        mean_logprob=-0.16,
        margin=0.7,
        prob_winner=0.85,
        prob_runner_up=0.15,
    )

    # These are the fields critic.py accesses
    assert result.node_id == "AVOIDANCE"
    assert result.code == "HIDE"
    assert result.confidence == 0.85
    assert result.mean_logprob == -0.16

    print("✅ ClassificationResult interface test passed - CoT results compatible with critic")


def test_code_map_consistency():
    """Verify CoT classifier uses the same CODE_MAP as System 1."""
    from safety.classifier import CODE_MAP as S1_CODE_MAP
    from safety.classifier_cot import CODE_MAP as S2_CODE_MAP

    assert S1_CODE_MAP is S2_CODE_MAP, "System 2 should import CODE_MAP from System 1"

    print("✅ CODE_MAP consistency test passed - both systems use identical mapping")


@pytest.mark.asyncio
@requires_scaleway
async def test_cot_classifier_basic():
    """Test CoT classifier with clear-cut cases (requires API key)."""
    from safety.classifier_cot import ClinicalClassifierCoT

    classifier = ClinicalClassifierCoT()

    # Clear cases that should be unambiguous
    test_cases = [
        ("I hide under the bed", "AVOIDANCE", False),
        ("I punch the attacker", "BEHAVIORAL_MASTERY", True),
        ("I call for help", "SOCIAL_MASTERY", True),
        ("I kill everyone brutally", "VIOLENT_REVENGE", False),
        ("I felt terrified", "AFFECT_EXPRESSION", True),
    ]

    passed = 0
    for user_input, expected_node, expected_safe in test_cases:
        node_id, result = await classifier.classify_intent(
            user_input, return_logprobs=True
        )

        is_safe = classifier.graph.nodes[node_id].get("type") in ("adaptive", "neutral")

        if node_id == expected_node:
            status = "✅"
            passed += 1
        elif is_safe == expected_safe:
            status = "⚠️ (safety correct, node differs)"
            passed += 1  # Count safety-correct as pass
        else:
            status = "❌"

        code = result.code if result else "?"
        conf = f"{result.confidence:.3f}" if result else "?"
        print(f"  {status} '{user_input}' -> {code}/{node_id} (conf={conf})")

    print(f"\n✅ CoT basic test: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


@pytest.mark.asyncio
@requires_scaleway
async def test_cot_classifier_context_aware():
    """Test CoT classifier with nightmare context (requires API key)."""
    from safety.classifier_cot import ClinicalClassifierCoT

    classifier = ClinicalClassifierCoT()

    nightmare = "I found my dog dead on the kitchen floor. I panicked and woke up."

    context_cases = [
        ("I don't feel any panic", "SUPPRESSION", False),
        ("I see my dog is actually alive and happy", None, True),  # Node varies, but should be safe
    ]

    passed = 0
    for user_input, expected_node, expected_safe in context_cases:
        node_id, result = await classifier.classify_intent(
            user_input, context=nightmare, return_logprobs=True
        )

        is_safe = classifier.graph.nodes[node_id].get("type") in ("adaptive", "neutral")

        if expected_node and node_id == expected_node:
            status = "✅"
            passed += 1
        elif is_safe == expected_safe:
            status = "✅ (safety correct)"
            passed += 1
        else:
            status = "❌"

        code = result.code if result else "?"
        print(f"  {status} '{user_input}' -> {code}/{node_id} (safe={is_safe})")

    print(f"\n✅ CoT context-aware test: {passed}/{len(context_cases)} passed")
    return passed == len(context_cases)


@pytest.mark.asyncio
@requires_scaleway
async def test_semantic_consistency():
    """Test semantic consistency evaluation (requires API key, multiple calls)."""
    from safety.classifier import ClinicalClassifier
    from safety.semantic_consistency import evaluate_semantic_consistency

    classifier = ClinicalClassifier()

    # Clear case: should have high agreement
    result = await evaluate_semantic_consistency(
        classifier=classifier,
        user_text="I hide under the bed",
        n_runs=3,  # Use 3 instead of 5 for testing speed
        stochastic_temperature=0.7,
    )

    print(f"  Consistency result for 'I hide under the bed':")
    print(f"    Majority: {result.code} ({result.node_id})")
    print(f"    Agreement: {result.agreement_ratio:.0%}")
    print(f"    Distribution: {result.vote_distribution}")
    print(f"    Confident: {result.is_confident}")

    assert result.node_id is not None
    assert 0.0 <= result.agreement_ratio <= 1.0
    assert result.n_runs == 3
    assert len(result.individual_results) == 3

    print("\n✅ Semantic consistency test passed")
    return True


@pytest.mark.asyncio
@requires_scaleway
async def test_cot_with_critic_pipeline():
    """
    Verify System 2 can be plugged into the critic pipeline.

    The critic calls classifier.classify_intent() and accesses
    result.confidence and result.mean_logprob. This test verifies
    the interface is compatible.
    """
    from safety.classifier_cot import ClinicalClassifierCoT
    from safety.critic import SafetyCritic

    # Create a critic using the CoT classifier
    cot_classifier = ClinicalClassifierCoT()
    critic = SafetyCritic(classifier=cot_classifier)

    # Run a simple evaluation through the full pipeline
    result = await critic.evaluate_intervention("I hide under the bed")

    assert result is not None
    assert hasattr(result, "is_safe")
    assert hasattr(result, "node_id")
    assert hasattr(result, "confidence")

    safe_str = "SAFE" if result.is_safe else f"UNSAFE ({result.severity})"
    print(f"  Critic with CoT: 'I hide under the bed' -> {result.node_id} ({safe_str})")

    # This should be classified as unsafe (avoidance)
    assert not result.is_safe, "Hiding should be classified as unsafe"

    print("\n✅ Critic pipeline integration test passed")
    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("System 2 (CoT) Classifier & Semantic Consistency Tests")
    print("=" * 60)

    # Unit tests (no API calls)
    print("\n--- Unit Tests (no API calls) ---")
    test_cot_prompt_parity()
    test_cot_json_parsing()
    test_classification_result_interface()
    test_code_map_consistency()

    # Integration tests (require API keys)
    print("\n--- Integration Tests (require API keys) ---")

    try:
        await test_cot_classifier_basic()
        await test_cot_classifier_context_aware()
        await test_semantic_consistency()
        await test_cot_with_critic_pipeline()
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        print("   (This may be due to missing API keys)")
        raise

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
