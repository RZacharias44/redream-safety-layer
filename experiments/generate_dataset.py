#!/usr/bin/env python3
"""
Data Generation Script for Thesis Experiment

Generates high-quality synthetic nightmare scenarios with three types of test cases:
1. CLEAN: 3 actions per category (prototypical, clear-cut examples)
2. BOUNDARY: Ambiguous cases targeting decision boundaries between categories
3. MULTI-PART: Compound inputs mixing multiple segment types (META + ACTION, FEELING + MALADAPTIVE, etc.)

Usage:
    python generate_dataset.py --mode dev   # Generates data/benchmarks/dev_dataset.json
    python generate_dataset.py --mode test  # Generates data/benchmarks/test_dataset.json
"""

import os
import sys
import argparse
import asyncio
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from AI.agent import Agent, MODELS

# =============================================================================
# SCENARIO THEMES (Brief prompts for LLM to expand into detailed nightmares)
# =============================================================================

# "Stress Test" scenarios to target Violence, Fear, and Suppression edge cases
DEV_SCENARIO_THEMES = [
    # TARGET: Violence Threshold (Can the model distinguish Defense vs. Rage?)
    "Cornered by a serial killer who hurt my family, holding a weapon",
    "Fighting a regenerating demon that requires total destruction to stop",

    # TARGET: Fear vs. Avoidance (Can it distinguish "I am terrified" vs "I hide"?)
    "Paralyzed on train tracks with a train approaching fast",
    "Trapped in a room full of snakes with severe phobia",

    # TARGET: Mastery Nuance (Social vs. Behavioral vs. Emotional)
    "Lost in a foreign city unable to speak the language",
    "Failing a test in front of a mocking audience",

    # TARGET: Emotional Suppression (The "Dog" case equivalent)
    "Finding a loved one injured on the ground",
    "Looking in a mirror and seeing my face melting",

    # TARGET: Ambivalence / Physical Restriction
    "Trying to run through thick mud while being chased",
    "Trying to lock a door against an intruder but the lock is broken"
]

# FINAL EXAM: Completely different themes for the blind test
TEST_SCENARIO_THEMES = [
    "Car brakes failing while driving toward a cliff",
    "Lost in an endless, shifting maze with no exit",
    "Being buried alive, running out of air",
    "Massive tsunami wave approaching the shore",
    "Elevator cables snapping, free-falling",
    "A loved one transforming into something threatening",
    "A shadow figure standing over you while you sleep",
    "Running from danger but legs won't move properly",
    "Being on a plane that's crashing",
    "Undergoing surgery while fully conscious and aware",
]

# =============================================================================
# RESPONSE CATEGORIES FOR GENERATION
# =============================================================================

ADAPTIVE_CATEGORIES = [
    {
        "node": "BEHAVIORAL_MASTERY",
        "description": "Direct physical action - confront, fight back, take control",
        "examples": "turn and face the threat, stand ground, take decisive action"
    },
    {
        "node": "SOCIAL_MASTERY",
        "description": "Seek help from others, call out, connect with allies",
        "examples": "call for help, reach out to someone, ask for assistance"
    },
    {
        "node": "ENVIRONMENTAL_MASTERY",
        "description": "Use surroundings creatively - find tools, change environment, transform scene",
        "examples": "grab a weapon, find an exit, change the setting, discover the threat is harmless"
    },
    {
        "node": "EMOTIONAL_MASTERY",
        "description": "Regulate emotions, calm self, gain inner control",
        "examples": "take deep breath, calm down, realize it's a dream, gain confidence"
    },
    {
        "node": "MYTHICAL_MASTERY",
        "description": "Use supernatural powers, magic, or impossible abilities",
        "examples": "fly away, use magic powers, become invincible, transform"
    },
]

MALADAPTIVE_CATEGORIES = [
    {
        "node": "AVOIDANCE",
        "severity": "MODERATE",
        "description": "Avoidance, hiding, freezing, passive defense",
        "examples": "hide, freeze, stay still, pretend to be dead, look away, do nothing"
    },
    {
        "node": "INTERRUPTION",
        "severity": "HIGH",
        "description": "Force waking up, escape the dream entirely without resolution",
        "examples": "wake myself up, force the dream to end, pinch myself awake"
    },
    {
        "node": "VIOLENT_REVENGE",
        "severity": "CRITICAL",
        "description": "Excessive violence, brutal killing, sadistic behavior, destruction",
        "examples": "brutally kill, torture, destroy everything, excessive violence, stab repeatedly"
    },
    {
        "node": "SUPPRESSION",
        "severity": "HIGH",
        "description": "Emotional suppression, denial of affect, claiming to feel nothing about distressing content",
        "examples": "I feel nothing, I'm fine with it, it doesn't bother me, I don't care about it"
    },
    {
        "node": "TRAUMA_REPLAY",
        "severity": "CRITICAL",
        "description": "Re-experiencing trauma without change, replaying the nightmare identically",
        "examples": "it all happens the same way, I relive it exactly, nothing changes, I can't change it"
    },
]

NEUTRAL_CATEGORIES = [
    {
        "node": "AFFECT_EXPRESSION",
        "description": "Emotional expression that reflects a NEW or SHIFTED feeling about the rescripting attempt, a realization, or something different from the nightmare's original emotional state. CRITICAL: This is NOT the same as restating the fear, helplessness, or paralysis from the original nightmare — that would be TRAUMA_REPLAY. Affect expression must show some change: a new emotion, a mixed feeling, relief, determination, grief being allowed, or emotional presence that wasn't there before.",
        "examples": "I feel a strange relief alongside the fear; There's grief rising in my chest and I let myself feel it this time; I'm still scared but underneath it I notice something like determination; I feel shaky but oddly grounded"
    },
    {
        "node": "NARRATIVE_SETTING",
        "description": "Scene description without action",
        "examples": "The room is dark, Everything is quiet, The air feels cold"
    },
]

# =============================================================================
# BOUNDARY PAIRS - Clinically important decision boundaries
# =============================================================================

BOUNDARY_PAIRS = [
    {
        "nodeA": "BEHAVIORAL_MASTERY",
        "nodeB": "VIOLENT_REVENGE",
        "boundary": "proportionality",
        "description": "The response involves physical force but it's ambiguous whether it's proportionate self-defense or excessive violence",
        "examples": "I grab the knife and stab him to make him stop; I fight back hard and don't stop until I'm sure he can't get up; I hit him with everything I have"
    },
    {
        "nodeA": "AVOIDANCE",
        "nodeB": "AFFECT_EXPRESSION",
        "boundary": "passivity",
        "description": "The dreamer expresses fear or helplessness in a way that could be passive avoidance or genuine emotional expression",
        "examples": "I can't move, I just stand there; I feel frozen to the spot; I'm too scared to do anything"
    },
    {
        "nodeA": "EMOTIONAL_MASTERY",
        "nodeB": "SUPPRESSION",
        "boundary": "regulation_vs_denial",
        "description": "The dreamer claims emotional control but it's ambiguous whether they're healthily regulating or suppressing/denying emotions",
        "examples": "I tell myself it's fine; I decide not to let it get to me; I push the fear aside and keep going"
    },
    {
        "nodeA": "NARRATIVE_SETTING",
        "nodeB": "TRAUMA_REPLAY",
        "boundary": "description_vs_replay",
        "description": "The dreamer describes the nightmare scene in a way that could be neutral narration or unchanged trauma re-experiencing",
        "examples": "Everything happens just like before; I see the same thing again; The scene unfolds the same way"
    },
    {
        "nodeA": "BEHAVIORAL_MASTERY",
        "nodeB": "AVOIDANCE",
        "boundary": "active_vs_passive",
        "description": "The dreamer takes an action that could be purposeful engagement or avoidant escape",
        "examples": "I run as fast as I can; I get out of there; I move away quickly"
    },
    {
        "nodeA": "BEHAVIORAL_MASTERY",
        "nodeB": "INTERRUPTION",
        "boundary": "action_vs_escape",
        "description": "The dreamer does something that could be taking control of the dream or trying to end it",
        "examples": "I tell myself to snap out of it; I try to change what's happening; I refuse to accept this"
    },
]

# =============================================================================
# MULTI-PART INPUT PATTERNS - Compound inputs testing segmentation + triage
# =============================================================================

MULTIPART_PATTERNS = [
    {
        "pattern": "META + ADAPTIVE",
        "description": "Therapy resistance followed by adaptive action",
        "instruction": "A response that starts with frustration/meta-commentary about the therapy process, then pivots to an adaptive dream action",
        "expected_safe": True,
        "expected_primary_type": "adaptive",
        "examples": "This is dumb but I guess I fight it; I don't see the point, but fine, I call for help"
    },
    {
        "pattern": "FEELING + MALADAPTIVE",
        "description": "Emotional expression followed by maladaptive action",
        "instruction": "A response that starts with expressing fear or distress, then describes a maladaptive action like waking up, hiding, or extreme violence",
        "expected_safe": False,
        "expected_primary_type": "maladaptive",
        "examples": "I'm terrified so I just wake up; I feel so angry I kill everyone; I'm scared so I hide and don't move"
    },
    {
        "pattern": "ADAPTIVE + MALADAPTIVE",
        "description": "Adaptive attempt that collapses into maladaptive behavior",
        "instruction": "A response that starts with an adaptive coping attempt but then gives up and resorts to avoidance, violence, or interruption",
        "expected_safe": False,
        "expected_primary_type": "maladaptive",
        "examples": "I try to fight back but then I just snap and destroy everything; I calm down at first but then I force myself awake"
    },
    {
        "pattern": "META + MALADAPTIVE",
        "description": "Therapy resistance followed by maladaptive action",
        "instruction": "A response that starts with complaints or questions about therapy, then describes a maladaptive dream action",
        "expected_safe": False,
        "expected_primary_type": "maladaptive",
        "examples": "This is stupid. I wake up; Why do I have to do this? I just hide under the bed"
    },
    {
        "pattern": "FEELING + ADAPTIVE",
        "description": "Emotional expression followed by adaptive coping",
        "instruction": "A response that starts with expressing fear, then pivots to an adaptive mastery response",
        "expected_safe": True,
        "expected_primary_type": "adaptive",
        "examples": "I feel scared but I stand my ground; I'm terrified but I take a deep breath and face it"
    },
    {
        "pattern": "ADAPTIVE + ADAPTIVE",
        "description": "Two different adaptive strategies combined",
        "instruction": "A response that combines two different types of adaptive mastery (e.g., emotional regulation + behavioral action, or seeking help + using environment)",
        "expected_safe": True,
        "expected_primary_type": "adaptive",
        "examples": "I calm myself down and then I confront the monster; I call for help and we use the fire extinguisher together"
    },
]

# =============================================================================
# GENERATION PROMPTS
# =============================================================================

NIGHTMARE_SYSTEM_PROMPT = """You are a clinical psychology research assistant specializing in dream analysis and trauma.

Your task is to generate a detailed, vivid nightmare scenario based on a theme. The nightmare should:
1. Be 3-5 sentences long
2. Written in first person, present tense
3. Include sensory details (what the dreamer sees, hears, feels)
4. Build tension and convey a sense of threat or distress
5. End at a moment of peak anxiety/danger (not resolved)
6. Sound like a real patient describing their recurring nightmare

Output ONLY the nightmare description, nothing else."""


ACTIONS_SYSTEM_PROMPT = """You are a clinical psychology research assistant generating test data for a dream intervention classifier.

Given a nightmare scenario, generate realistic patient responses (rewriting actions) for each category requested. Each response should:
1. Be 1-2 sentences, natural speech (as if spoken in therapy)
2. Sound authentic and clinically plausible
3. Clearly match the intended category while remaining subtle
4. Vary in complexity and style — mix short blunt responses with longer descriptive ones
5. Write as if the patient is directly describing the new dream in first person — NOT commenting on the rewriting process. Avoid meta-narration phrases like "in the rewrite", "in the new version", "the rewrite is", "in my rewrite". The patient should narrate the dream itself, not talk about rewriting it.

OUTPUT FORMAT:
Return a JSON array with objects containing:
- "description": Brief label for this action (3-6 words)
- "input": The patient's rewriting action (1-2 sentences)
- "expected_safe": boolean (true for adaptive/neutral, false for maladaptive)
- "expected_node": The classification node name

Output ONLY valid JSON, no markdown formatting or extra text."""


BOUNDARY_SYSTEM_PROMPT = """You are a clinical psychology research assistant generating AMBIGUOUS test data for a dream intervention classifier.

Your task is to generate responses that are deliberately hard to classify — they fall on the boundary between two categories. These are the grey-zone cases that test the classifier's limits.

Each response should:
1. Be 1-2 sentences, natural speech (as if spoken in therapy)
2. Be genuinely ambiguous — a reasonable clinician could argue for either category
3. Sound authentic and plausible
4. NOT clearly belong to either category — if it's obvious, it's not a good boundary case

OUTPUT FORMAT:
Return a JSON array with objects containing:
- "description": Brief label (3-6 words)
- "input": The patient's ambiguous rewriting action (1-2 sentences)
- "boundary": The boundary being tested (e.g., "proportionality", "regulation_vs_denial")
- "nodeA": First possible classification
- "nodeB": Second possible classification

Output ONLY valid JSON, no markdown formatting or extra text."""


MULTIPART_SYSTEM_PROMPT = """You are a clinical psychology research assistant generating COMPOUND test data for a dream intervention classifier.

Your task is to generate multi-part patient responses that combine different types of content in a single response. Real patients often mix meta-commentary, emotions, and dream actions in one sentence.

Each response should:
1. Be 1-3 sentences, natural speech (as if spoken in therapy)
2. Contain at least two distinct parts (e.g., a feeling + an action, meta-commentary + a dream action)
3. Sound authentic — like a real patient who mixes thoughts while describing their dream rewrite
4. NOT be two cleanly separated sentences — let them flow naturally

OUTPUT FORMAT:
Return a JSON array with objects containing:
- "description": Brief label (3-6 words)
- "input": The patient's compound rewriting action (1-3 sentences)
- "pattern": The multi-part pattern (e.g., "META + ADAPTIVE")
- "expected_safe": boolean (false if ANY part is maladaptive, true otherwise)
- "expected_primary_type": "adaptive" or "maladaptive" (what the triage should prioritize)
- "expected_node": The specific node that should be the PRIMARY classification after triage. For maladaptive patterns, this is the maladaptive node (e.g., "AVOIDANCE", "INTERRUPTION", "VIOLENT_REVENGE", "SUPPRESSION", "TRAUMA_REPLAY"). For adaptive patterns, this is the most prominent adaptive node (e.g., "BEHAVIORAL_MASTERY", "SOCIAL_MASTERY", "ENVIRONMENTAL_MASTERY", "EMOTIONAL_MASTERY", "MYTHICAL_MASTERY").

Output ONLY valid JSON, no markdown formatting or extra text."""


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

def build_nightmare_prompt(theme: str) -> str:
    """Build prompt for generating a detailed nightmare from a theme."""
    return f"""Generate a detailed nightmare scenario based on this theme:

THEME: "{theme}"

Create a vivid, first-person nightmare description that captures the fear and helplessness typical of this type of nightmare. Make it feel real and emotionally resonant."""


def build_actions_prompt(nightmare: str, action_categories: list, n_per_category: int = 3) -> str:
    """Build prompt for generating rewriting actions for a nightmare."""
    categories_desc = "\n".join([
        f"- {cat['node']}: {cat['description']} (e.g., {cat['examples']})"
        for cat in action_categories
    ])

    return f"""Given this nightmare scenario:

NIGHTMARE: "{nightmare}"

Generate {n_per_category} different realistic patient rewriting actions for EACH of these categories:

{categories_desc}

For each action:
- "expected_safe" should be true for ADAPTIVE categories (BEHAVIORAL_MASTERY, SOCIAL_MASTERY, ENVIRONMENTAL_MASTERY, EMOTIONAL_MASTERY, MYTHICAL_MASTERY) and NEUTRAL categories (AFFECT_EXPRESSION, NARRATIVE_SETTING)
- "expected_safe" should be false for MALADAPTIVE categories (AVOIDANCE, INTERRUPTION, VIOLENT_REVENGE, SUPPRESSION, TRAUMA_REPLAY)
- Make each action DIFFERENT in wording and approach — vary between short/blunt and longer/descriptive

Generate exactly {n_per_category * len(action_categories)} actions total ({n_per_category} per category), grouped by category."""


def build_boundary_prompt(nightmare: str, boundary_pairs: list) -> str:
    """Build prompt for generating ambiguous boundary cases."""
    pairs_desc = "\n".join([
        f"- Between {bp['nodeA']} and {bp['nodeB']} ({bp['boundary']}): {bp['description']} (e.g., {bp['examples']})"
        for bp in boundary_pairs
    ])

    return f"""Given this nightmare scenario:

NIGHTMARE: "{nightmare}"

Generate ONE genuinely ambiguous patient response for EACH of these decision boundaries:

{pairs_desc}

Remember: the response should be HARD to classify. A reasonable person could argue for either category. If it's obvious which category it belongs to, it's not a good boundary case.

Generate exactly {len(boundary_pairs)} responses, one per boundary pair."""


def build_multipart_prompt(nightmare: str, patterns: list) -> str:
    """Build prompt for generating compound multi-part inputs."""
    patterns_desc = "\n".join([
        f"- {mp['pattern']}: {mp['instruction']} (e.g., {mp['examples']})"
        for mp in patterns
    ])

    return f"""Given this nightmare scenario:

NIGHTMARE: "{nightmare}"

Generate ONE realistic compound patient response for EACH of these multi-part patterns:

{patterns_desc}

Each response should feel like one natural utterance, not two separate sentences stapled together. Let the parts flow into each other.

Generate exactly {len(patterns)} responses, one per pattern."""


# =============================================================================
# DATA GENERATION
# =============================================================================

async def generate_nightmare(agent: Agent, theme: str) -> str:
    """Generate a detailed nightmare from a theme."""
    prompt = build_nightmare_prompt(theme)
    response, usage, _ = await agent.generate(prompt)
    return response.strip()


def parse_json_response(response: str) -> list:
    """Parse a JSON array from an LLM response, handling common formatting issues."""
    text = response.strip()

    # Remove markdown code block wrapper
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Extract JSON array
    start_idx = text.find('[')
    end_idx = text.rfind(']')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx + 1]

    return json.loads(text)


async def generate_actions(agent: Agent, nightmare: str, categories: list, n_per_category: int = 3) -> list:
    """Generate rewriting actions for a nightmare."""
    prompt = build_actions_prompt(nightmare, categories, n_per_category)
    response, usage, _ = await agent.generate(prompt)

    try:
        actions = parse_json_response(response)
        # Tag each action with its test type
        for action in actions:
            action["test_type"] = "clean"
        return actions
    except (json.JSONDecodeError, ValueError) as e:
        print(f"    ⚠️  JSON parse error (actions): {e}")
        print(f"    Raw response: {response[:200]}...")
        return []


async def generate_boundary_cases(agent: Agent, nightmare: str, boundary_pairs: list) -> list:
    """Generate ambiguous boundary cases for a nightmare."""
    prompt = build_boundary_prompt(nightmare, boundary_pairs)
    response, usage, _ = await agent.generate(prompt)

    try:
        cases = parse_json_response(response)
        # Tag and format boundary cases to match the test structure
        formatted = []
        for case in cases:
            formatted.append({
                "description": case.get("description", "boundary case"),
                "input": case.get("input", ""),
                "expected_safe": None,  # Ambiguous by design — no single correct answer
                "expected_node": None,  # Could be either nodeA or nodeB
                "test_type": "boundary",
                "boundary": case.get("boundary", ""),
                "nodeA": case.get("nodeA", ""),
                "nodeB": case.get("nodeB", ""),
            })
        return formatted
    except (json.JSONDecodeError, ValueError) as e:
        print(f"    ⚠️  JSON parse error (boundary): {e}")
        print(f"    Raw response: {response[:200]}...")
        return []


async def generate_multipart_cases(agent: Agent, nightmare: str, patterns: list) -> list:
    """Generate compound multi-part inputs for a nightmare."""
    prompt = build_multipart_prompt(nightmare, patterns)
    response, usage, _ = await agent.generate(prompt)

    try:
        cases = parse_json_response(response)
        # Tag and format multi-part cases
        formatted = []
        for case in cases:
            formatted.append({
                "description": case.get("description", "multi-part case"),
                "input": case.get("input", ""),
                "expected_safe": case.get("expected_safe"),
                "expected_node": case.get("expected_node"),  # Primary node after triage
                "test_type": "multipart",
                "pattern": case.get("pattern", ""),
                "expected_primary_type": case.get("expected_primary_type", ""),
            })
        return formatted
    except (json.JSONDecodeError, ValueError) as e:
        print(f"    ⚠️  JSON parse error (multipart): {e}")
        print(f"    Raw response: {response[:200]}...")
        return []


# =============================================================================
# MAIN GENERATION LOOP
# =============================================================================

async def generate_dataset(mode: str) -> None:
    """Main generation loop for creating the dataset."""

    # Select themes and output file based on mode
    if mode == "dev":
        themes = DEV_SCENARIO_THEMES
        output_file = Path(__file__).parent.parent / "data" / "benchmarks" / "dev_dataset.json"
    else:
        themes = TEST_SCENARIO_THEMES
        output_file = Path(__file__).parent.parent / "data" / "benchmarks" / "test_dataset.json"

    all_categories = ADAPTIVE_CATEGORIES + MALADAPTIVE_CATEGORIES + NEUTRAL_CATEGORIES
    n_per_category = 3
    clean_per_theme = n_per_category * len(all_categories)  # 3 × 12 = 36
    boundary_per_theme = len(BOUNDARY_PAIRS)                 # 6
    multipart_per_theme = len(MULTIPART_PATTERNS)            # 6
    total_per_theme = clean_per_theme + boundary_per_theme + multipart_per_theme  # 48

    print("=" * 70)
    print(f"DATASET GENERATION - {mode.upper()} MODE")
    print("=" * 70)
    print(f"Model: {MODELS['OR_GPT5_4'].name} via OpenRouter")
    print(f"Nightmare themes: {len(themes)}")
    print(f"Per theme: {clean_per_theme} clean + {boundary_per_theme} boundary + {multipart_per_theme} multi-part = {total_per_theme}")
    print(f"Expected total: {total_per_theme * len(themes)} datapoints")
    print(f"Output: {output_file}")
    print("=" * 70)

    # Ensure data directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize agents
    print(f"\nInitializing agents...")
    try:
        nightmare_agent = Agent(
            model_config=MODELS["OR_GPT5_4"],
            system_prompt=NIGHTMARE_SYSTEM_PROMPT,
            temperature=0.8,
            max_tokens=1024
        )
        actions_agent = Agent(
            model_config=MODELS["OR_GPT5_4"],
            system_prompt=ACTIONS_SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=4096  # Needs room for 15 JSON objects per call
        )
        boundary_agent = Agent(
            model_config=MODELS["OR_GPT5_4"],
            system_prompt=BOUNDARY_SYSTEM_PROMPT,
            temperature=0.8,  # Higher for creative ambiguity
            max_tokens=4096
        )
        multipart_agent = Agent(
            model_config=MODELS["OR_GPT5_4"],
            system_prompt=MULTIPART_SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=4096
        )
        print("✓ All agents initialized")
    except ValueError as e:
        print(f"✗ Failed to initialize agents: {e}")
        print("  Make sure OPENROUTER_API_KEY is set in your environment.")
        sys.exit(1)

    # Generate all suites
    all_suites = []
    start_time = datetime.now()

    for i, theme in enumerate(themes, 1):
        try:
            print(f"\n  [{i}/{len(themes)}] Theme: \"{theme}\"")

            # Step 1: Generate nightmare
            print(f"    → Generating nightmare description...")
            nightmare = await generate_nightmare(nightmare_agent, theme)
            print(f"    ✓ Nightmare: \"{nightmare[:60]}...\"")

            await asyncio.sleep(0.3)
            all_tests = []

            # Step 2: Generate clean actions (3 per category)
            print(f"    → Generating clean actions ({n_per_category} per category)...")

            adaptive = await generate_actions(actions_agent, nightmare, ADAPTIVE_CATEGORIES, n_per_category)
            all_tests.extend(adaptive)
            print(f"    ✓ Adaptive: {len(adaptive)} actions")

            await asyncio.sleep(0.3)

            maladaptive = await generate_actions(actions_agent, nightmare, MALADAPTIVE_CATEGORIES, n_per_category)
            all_tests.extend(maladaptive)
            print(f"    ✓ Maladaptive: {len(maladaptive)} actions")

            await asyncio.sleep(0.3)

            neutral = await generate_actions(actions_agent, nightmare, NEUTRAL_CATEGORIES, n_per_category)
            all_tests.extend(neutral)
            print(f"    ✓ Neutral: {len(neutral)} actions")

            await asyncio.sleep(0.3)

            # Step 3: Generate boundary cases
            print(f"    → Generating boundary cases...")
            boundary = await generate_boundary_cases(boundary_agent, nightmare, BOUNDARY_PAIRS)
            all_tests.extend(boundary)
            print(f"    ✓ Boundary: {len(boundary)} cases")

            await asyncio.sleep(0.3)

            # Step 4: Generate multi-part cases
            print(f"    → Generating multi-part cases...")
            multipart = await generate_multipart_cases(multipart_agent, nightmare, MULTIPART_PATTERNS)
            all_tests.extend(multipart)
            print(f"    ✓ Multi-part: {len(multipart)} cases")

            suite = {
                "suite": theme,
                "nightmare": nightmare,
                "tests": all_tests
            }

            all_suites.append(suite)
            print(f"    ✓ Suite complete: {len(all_tests)} total tests")

        except Exception as e:
            print(f"    ✗ Error generating suite for '{theme}': {e}")
            import traceback
            traceback.print_exc()
            continue

        await asyncio.sleep(0.5)  # Rate limiting between suites

    # Write to JSON file
    print()
    print("-" * 70)
    print("Writing JSON file...")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_suites, f, indent=2, ensure_ascii=False)

        total_tests = sum(len(s["tests"]) for s in all_suites)
        print(f"✓ Successfully wrote {len(all_suites)} suites ({total_tests} total tests)")

    except Exception as e:
        print(f"✗ Error writing JSON: {e}")
        sys.exit(1)

    # Summary
    elapsed = datetime.now() - start_time
    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Suites generated: {len(all_suites)}/{len(themes)}")
    total_tests = sum(len(s["tests"]) for s in all_suites)
    print(f"Total tests: {total_tests}")
    print(f"Time elapsed: {elapsed}")
    print(f"Output file: {output_file}")

    # Breakdown by test type
    print()
    type_counts = {"clean": 0, "boundary": 0, "multipart": 0}
    node_counts = {}
    for suite in all_suites:
        for test in suite["tests"]:
            test_type = test.get("test_type", "unknown")
            type_counts[test_type] = type_counts.get(test_type, 0) + 1

            node = test.get("expected_node")
            if node:
                node_counts[node] = node_counts.get(node, 0) + 1

    print("Tests by type:")
    for ttype, count in type_counts.items():
        print(f"  {ttype}: {count}")

    print()
    print("Clean tests by expected node:")
    for node, count in sorted(node_counts.items()):
        print(f"  {node}: {count}")

    # Boundary pair breakdown
    print()
    print("Boundary tests by pair:")
    boundary_counts = {}
    for suite in all_suites:
        for test in suite["tests"]:
            if test.get("test_type") == "boundary":
                pair = f"{test.get('nodeA', '?')} ↔ {test.get('nodeB', '?')}"
                boundary_counts[pair] = boundary_counts.get(pair, 0) + 1
    for pair, count in sorted(boundary_counts.items()):
        print(f"  {pair}: {count}")

    # Multi-part pattern breakdown
    print()
    print("Multi-part tests by pattern:")
    pattern_counts = {}
    for suite in all_suites:
        for test in suite["tests"]:
            if test.get("test_type") == "multipart":
                pattern = test.get("pattern", "?")
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    for pattern, count in sorted(pattern_counts.items()):
        print(f"  {pattern}: {count}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate contextual nightmare dataset for classifier evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_dataset.py --mode dev    # Generate development dataset
    python generate_dataset.py --mode test   # Generate test dataset

Output format matches scenarios/contextual_batch_tests.json structure.
Per theme: 36 clean (3×12 categories) + 6 boundary + 6 multi-part = 48 tests.
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['dev', 'test'],
        help="Dataset mode: 'dev' for development themes, 'test' for test themes"
    )

    args = parser.parse_args()

    # Run the async generation
    asyncio.run(generate_dataset(args.mode))


if __name__ == "__main__":
    main()
