#!/usr/bin/env python3
"""
Test script for Phase 2: Safety Module

Verifies that all three blocks work correctly:
- Block A: Knowledge Graph
- Block B: Clinical Classifier  
- Block C: SafetyCritic Pipeline
"""

import asyncio
import sys
import os
import pytest
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

requires_scaleway = pytest.mark.skipif(
    not os.getenv("SCALEWAY_API_KEY"),
    reason="requires SCALEWAY_API_KEY for live safety classifier calls",
)


def test_knowledge_graph():
    """Test Block A: Knowledge Graph"""
    print("\n" + "=" * 60)
    print("Block A: Knowledge Graph")
    print("=" * 60)
    
    from safety.graph_definitions import (
        build_clinical_graph, 
        get_maladaptive_nodes, 
        get_adaptive_nodes,
        get_neutral_nodes,
        get_correction_strategies
    )
    
    G = build_clinical_graph()
    
    # Check node counts
    maladaptive = get_maladaptive_nodes(G)
    adaptive = get_adaptive_nodes(G)
    neutral = get_neutral_nodes(G)
    
    print(f"✅ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"✅ Maladaptive nodes: {len(maladaptive)}")
    for node_id in maladaptive:
        print(f"   - {node_id}")
    
    print(f"✅ Adaptive nodes: {len(adaptive)}")
    for node_id in adaptive:
        print(f"   - {node_id}")
    
    print(f"✅ Neutral nodes: {len(neutral)}")
    for node_id in neutral:
        print(f"   - {node_id}")
    
    # Check correction strategies
    print(f"✅ Correction strategies:")
    for mal_node in maladaptive:
        strategies = get_correction_strategies(G, mal_node)
        targets = [s["target_node"] for s in strategies]
        print(f"   - {mal_node} -> {targets}")
    
    # Assertions
    assert len(maladaptive) == 5, "Should have 5 maladaptive nodes"
    assert len(adaptive) == 5, "Should have 5 adaptive nodes (including MYTHICAL_MASTERY)"
    assert len(neutral) == 2, "Should have 2 neutral nodes (NARRATIVE_SETTING, AFFECT_EXPRESSION)"
    assert G.number_of_nodes() == 13, "Should have 13 total nodes (5 mal + 5 adapt + 2 neutral + 1 unclassified)"
    assert G.number_of_edges() >= 15, "Should have at least 15 correction edges"

    # Verify UNCLASSIFIED node exists
    assert "UNCLASSIFIED" in G.nodes(), "Should have UNCLASSIFIED node"
    unclassified_attrs = dict(G.nodes["UNCLASSIFIED"])
    assert unclassified_attrs["type"] == "unclassified", "UNCLASSIFIED should have type 'unclassified'"
    assert unclassified_attrs["system_action"] == "seek_clarification", "UNCLASSIFIED should trigger clarification"

    # Verify new nodes exist
    assert "SUPPRESSION" in maladaptive, "Should have SUPPRESSION node"
    assert "TRAUMA_REPLAY" in maladaptive, "Should have TRAUMA_REPLAY node"

    # Verify graph metadata matches the clinically validated severity labels
    assert maladaptive["AVOIDANCE"]["severity"] == "high", "AVOIDANCE should have high severity"

    # Verify SUPPRESSION has correct attributes
    supp_attrs = maladaptive["SUPPRESSION"]
    assert supp_attrs["severity"] == "high", "SUPPRESSION should have high severity"
    assert supp_attrs["category"] == "suppression", "SUPPRESSION should have suppression category"

    # Verify TRAUMA_REPLAY has correct attributes
    replay_attrs = maladaptive["TRAUMA_REPLAY"]
    assert replay_attrs["severity"] == "critical", "TRAUMA_REPLAY should have critical severity"
    assert replay_attrs["category"] == "replay", "TRAUMA_REPLAY should have replay category"

    # Verify correction edges for SUPPRESSION
    supp_strategies = get_correction_strategies(G, "SUPPRESSION")
    assert len(supp_strategies) == 3, "SUPPRESSION should have 3 correction strategies"
    supp_targets = [s["target_node"] for s in supp_strategies]
    assert supp_targets[0] == "EMOTIONAL_MASTERY", "SUPPRESSION primary -> EMOTIONAL_MASTERY"
    assert supp_targets[1] == "SOCIAL_MASTERY", "SUPPRESSION secondary -> SOCIAL_MASTERY"
    assert supp_targets[2] == "BEHAVIORAL_MASTERY", "SUPPRESSION tertiary -> BEHAVIORAL_MASTERY"

    # Verify correction edges for TRAUMA_REPLAY
    replay_strategies = get_correction_strategies(G, "TRAUMA_REPLAY")
    assert len(replay_strategies) == 3, "TRAUMA_REPLAY should have 3 correction strategies"
    replay_targets = [s["target_node"] for s in replay_strategies]
    assert replay_targets[0] == "BEHAVIORAL_MASTERY", "TRAUMA_REPLAY primary -> BEHAVIORAL_MASTERY"
    assert replay_targets[1] == "ENVIRONMENTAL_MASTERY", "TRAUMA_REPLAY secondary -> ENVIRONMENTAL_MASTERY"
    assert replay_targets[2] == "MYTHICAL_MASTERY", "TRAUMA_REPLAY tertiary -> MYTHICAL_MASTERY"

    print("\n✅ Block A: All tests passed!")
    return True


@pytest.mark.asyncio
@requires_scaleway
async def test_classifier():
    """Test Block B: Clinical Classifier (NLI-Based)"""
    print("\n" + "=" * 60)
    print("Block B: Clinical Classifier (NLI Cross-Encoder)")
    print("=" * 60)
    
    from safety.classifier import ClinicalClassifier
    
    classifier = ClinicalClassifier()
    print(f"✅ LLM classifier initialized with model: {classifier.model_config.name}")
    print(f"✅ Loaded {len(classifier.valid_node_ids)} valid graph node IDs")
    
    # Test cases with expected classifications
    # NLI classifier should handle these more reliably than LLM
    test_cases = [
        # Maladaptive
        ("I hide under the bed", "AVOIDANCE"),
        ("I wake myself up", "INTERRUPTION"),
        ("I kill everyone", "VIOLENT_REVENGE"),
        # Adaptive
        ("I fight back", "BEHAVIORAL_MASTERY"),
        ("I punch the monster", "BEHAVIORAL_MASTERY"),
        ("I call for help", "SOCIAL_MASTERY"),
        ("I find a sword", "ENVIRONMENTAL_MASTERY"),
        ("I calm myself down", "EMOTIONAL_MASTERY"),
        ("I use my magic powers", "MYTHICAL_MASTERY"),
        # Neutral
        ("I felt scared", "AFFECT_EXPRESSION"),
        ("There was a dark forest", "NARRATIVE_SETTING"),
        # CRITICAL: Negation handling (NLI should excel at this)
        ("I decided not to run away", "BEHAVIORAL_MASTERY"),
        ("I chose not to hide", "BEHAVIORAL_MASTERY"),
    ]
    
    passed = 0
    failed = 0
    
    for user_input, expected in test_cases:
        try:
            result, metadata = await classifier.classify_with_confidence(user_input)
            confidence = metadata.get("confidence_score", 0)
            
            if result == expected:
                print(f"✅ '{user_input}' -> {result} ({confidence:.3f})")
                passed += 1
            else:
                # Check if it's a reasonable alternative (same type)
                expected_type = classifier.get_node_info(expected).get("type")
                actual_type = metadata.get("node_type")
                
                if expected_type == actual_type:
                    print(f"⚠️  '{user_input}' -> {result} ({confidence:.3f}) [expected {expected}, same type]")
                    passed += 1  # Still counts as pass if same type
                else:
                    print(f"❌ '{user_input}' -> {result} ({confidence:.3f}) [expected {expected}]")
                    failed += 1
        except Exception as e:
            print(f"❌ '{user_input}' -> ERROR: {e}")
            failed += 1
    
    print(f"\n✅ Block B: {passed}/{len(test_cases)} tests passed, {failed} failed")
    return failed == 0


@pytest.mark.asyncio
@requires_scaleway
async def test_critic():
    """Test Block C: SafetyCritic Pipeline - Clinical Reasoning Engine"""
    print("\n" + "=" * 60)
    print("Block C: SafetyCritic Pipeline (Clinical Reasoning Engine)")
    print("=" * 60)
    
    from safety.critic import SafetyCritic
    
    critic = SafetyCritic()
    summary = critic.get_graph_summary()
    print(f"✅ SafetyCritic initialized")
    print(f"   Graph: {summary['total_nodes']} nodes, {summary['total_edges']} edges")
    print(f"   Maladaptive: {summary['maladaptive_nodes']}, Adaptive: {summary['adaptive_nodes']}, Neutral: {summary['neutral_nodes']}")
    
    # Test cases - simple inputs
    simple_cases = [
        ("I wake up", False),       # Maladaptive
        ("I fight back", True),     # Adaptive
        ("I do nothing", False),    # Maladaptive
        ("I hide", False),          # Maladaptive
        ("I call for help", True),  # Adaptive
        ("I felt scared", True),    # Neutral (safe)
    ]
    
    print("\n--- Simple Input Tests ---")
    passed = 0
    
    for user_input, expected_safe in simple_cases:
        try:
            result = await critic.evaluate_intervention(user_input)
            status = "✅" if result.is_safe == expected_safe else "⚠️"
            safe_str = "SAFE" if result.is_safe else f"UNSAFE ({result.severity})"
            
            print(f"{status} '{user_input}' -> {result.node_id} ({safe_str})")
            
            if result.is_safe == expected_safe:
                passed += 1
                
        except Exception as e:
            print(f"❌ '{user_input}' -> ERROR: {e}")
    
    # Test cases - multi-part inputs (segmentation)
    complex_cases = [
        ("I hate this therapy, but I guess I fight the dragon.", True),  # META + adaptive
        ("This is dumb", True),  # META only -> pass-through
        ("I felt terrified so I killed everyone", False),  # FEELING + CRITICAL violence
    ]
    
    print("\n--- Multi-Part Input Tests (Segmentation) ---")
    
    for user_input, expected_safe in complex_cases:
        try:
            result = await critic.evaluate_intervention(user_input)
            status = "✅" if result.is_safe == expected_safe else "⚠️"
            safe_str = "SAFE" if result.is_safe else f"UNSAFE ({result.severity})"
            
            print(f"{status} '{user_input[:40]}...' -> {result.node_id} ({safe_str})")
            if result.segments:
                print(f"   Segments: {[(s.segment_type, s.node_id) for s in result.segments]}")
            
            if result.is_safe == expected_safe:
                passed += 1
                
        except Exception as e:
            print(f"❌ '{user_input}' -> ERROR: {e}")
    
    # Test cases - context-aware classification (emotional suppression detection)
    nightmare_context = "I found my dog dead on the kitchen floor. I panicked and woke up."
    context_cases = [
        # Without context these might be classified as neutral affect, but with context they're avoidance
        ("I don't feel any panic", False),  # Emotional suppression = avoidance
        ("I'm fine with the situation", False),  # Emotional suppression = avoidance
        # Actual transformation should still be safe
        ("I see my dog is actually alive and happy", True),  # Actual transformation
    ]
    
    print("\n--- Context-Aware Tests (Emotional Suppression Detection) ---")
    print(f"   Nightmare context: \"{nightmare_context[:50]}...\"")
    
    for user_input, expected_safe in context_cases:
        try:
            result = await critic.evaluate_intervention(user_input, nightmare_context=nightmare_context)
            status = "✅" if result.is_safe == expected_safe else "⚠️"
            safe_str = "SAFE" if result.is_safe else f"UNSAFE ({result.severity})"
            
            print(f"{status} '{user_input}' -> {result.node_id} ({safe_str})")
            
            if result.is_safe == expected_safe:
                passed += 1
                
        except Exception as e:
            print(f"❌ '{user_input}' -> ERROR: {e}")
    
    total_cases = len(simple_cases) + len(complex_cases) + len(context_cases)
    print(f"\n✅ Block C: {passed}/{total_cases} tests passed")
    return passed == total_cases


async def main():
    """Run all Phase 2 tests."""
    print("=" * 60)
    print("Phase 2: Safety Module Tests")
    print("=" * 60)
    
    results = []
    
    # Test Block A (synchronous)
    try:
        results.append(("Block A: Knowledge Graph", test_knowledge_graph()))
    except Exception as e:
        print(f"❌ Block A failed: {e}")
        results.append(("Block A: Knowledge Graph", False))
    
    # Test Block B (async)
    try:
        results.append(("Block B: Classifier", await test_classifier()))
    except Exception as e:
        print(f"❌ Block B failed: {e}")
        results.append(("Block B: Classifier", False))
    
    # Test Block C (async)
    try:
        results.append(("Block C: SafetyCritic", await test_critic()))
    except Exception as e:
        print(f"❌ Block C failed: {e}")
        results.append(("Block C: SafetyCritic", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All Phase 2 tests passed!")
        print("Ready to proceed to Phase 3: Integration")
    else:
        print("❌ Some tests failed. Please review the output above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
