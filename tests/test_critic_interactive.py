#!/usr/bin/env python3
"""
Tier 1: Interactive Critic Testing Script

Standalone test script for the SafetyCritic "Clinical Reasoning Engine".

Modes:
1. Live mode (--live): Chat through actual IRT stages, then test critic at rewriting
2. Context mode (--context): Load pre-made context and test critic interactively
3. Scenario mode (--scenario): Load nightmare + test cases, run all tests against that nightmare
4. Batch mode (--batch): Legacy batch testing without nightmare context

Usage:
    python test_critic_interactive.py --live                   # Chat through IRT
    python test_critic_interactive.py --context FILE           # Load context, test interactively
    python test_critic_interactive.py --scenario FILE          # Run scenario test suite
    python test_critic_interactive.py --batch FILE             # Legacy batch tests
    python test_critic_interactive.py --scenario FILE -v       # Verbose: show constraints too

See Documents/Testing_Guide.md for full documentation.
"""

import asyncio
import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Optional, List, Dict
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import safety module
from safety.critic import SafetyCritic, SafetyResult

# AI.irt_app is imported lazily in run_live_mode() to avoid triggering
# Langfuse initialization when only testing the critic (scenario/batch modes).
from AI.models import Conversation, ChatInput, Stage


# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


def print_header(text: str):
    """Print a styled header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'━' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'━' * 70}{Colors.RESET}")


def print_subheader(text: str):
    """Print a styled subheader."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{'─' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{'─' * 70}{Colors.RESET}")


def print_section(title: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}▶ {title}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")


def print_stage_banner(stage: str):
    """Print current stage prominently."""
    stage_colors = {
        "recording": Colors.BLUE,
        "rewriting": Colors.YELLOW,
        "summary": Colors.CYAN,
        "rehearsal": Colors.GREEN,
        "final": Colors.GREEN
    }
    color = stage_colors.get(stage, Colors.RESET)
    print(f"\n{Colors.BOLD}{color}┌{'─' * 68}┐{Colors.RESET}")
    print(f"{Colors.BOLD}{color}│  CURRENT STAGE: {stage.upper():^48} │{Colors.RESET}")
    print(f"{Colors.BOLD}{color}└{'─' * 68}┘{Colors.RESET}")


def print_nightmare_box(nightmare: str):
    """Print nightmare in a prominent box."""
    print(f"\n{Colors.BOLD}{Colors.RED}╔{'═' * 68}╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.RED}║  NIGHTMARE CONTEXT{' ' * 49}║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.RED}╠{'═' * 68}╣{Colors.RESET}")
    
    # Word wrap the nightmare text
    words = nightmare.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= 66:
            current_line += (" " + word if current_line else word)
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    
    for line in lines:
        print(f"{Colors.BOLD}{Colors.RED}║{Colors.RESET}  {line:<66}{Colors.BOLD}{Colors.RED}║{Colors.RESET}")
    
    print(f"{Colors.BOLD}{Colors.RED}╚{'═' * 68}╝{Colors.RESET}")


def print_safe():
    """Print safe verdict."""
    print(f"{Colors.BOLD}{Colors.GREEN}✅ SAFE{Colors.RESET}")


def print_unsafe(severity: str):
    """Print unsafe verdict with severity."""
    color = Colors.RED if severity == "CRITICAL" else Colors.YELLOW
    print(f"{Colors.BOLD}{color}⚠️  UNSAFE ({severity}){Colors.RESET}")


def format_segments(result: SafetyResult) -> str:
    """Format segments for compact display."""
    if not result.segments:
        return "(no segments)"
    
    parts = []
    for seg in result.segments:
        type_abbrev = {
            "ACTION": "ACT",
            "THOUGHT": "THT", 
            "FEELING": "FEEL",
            "META": "META"
        }.get(seg.segment_type, seg.segment_type)
        
        node_abbrev = ""
        if seg.node_id:
            # Shorten node names
            node_abbrev = seg.node_id.replace("_MASTERY", "").replace("_", " ")
        
        text_preview = seg.text[:25] + "..." if len(seg.text) > 25 else seg.text
        parts.append(f"[{type_abbrev}] \"{text_preview}\"" + (f" → {node_abbrev}" if node_abbrev else ""))
    
    return " | ".join(parts)


def display_result_compact(result: SafetyResult, show_segments: bool = True):
    """Display result in compact format for batch/scenario testing."""
    # Verdict line
    if result.is_safe:
        verdict = f"{Colors.GREEN}✅ SAFE{Colors.RESET}"
    else:
        severity_color = Colors.RED if result.severity == "CRITICAL" else Colors.YELLOW
        verdict = f"{severity_color}⚠️  UNSAFE ({result.severity}){Colors.RESET}"
    
    print(f"  {verdict} → {result.node_id}")
    
    # Segmentation
    if show_segments and result.segments:
        print(f"  {Colors.DIM}Segments:{Colors.RESET}")
        for seg in result.segments:
            type_color = {
                "ACTION": Colors.GREEN,
                "THOUGHT": Colors.BLUE,
                "FEELING": Colors.CYAN,
                "META": Colors.DIM
            }.get(seg.segment_type, Colors.RESET)
            
            node_info = ""
            if seg.node_id:
                node_color = {
                    "adaptive": Colors.GREEN,
                    "maladaptive": Colors.RED,
                    "neutral": Colors.YELLOW
                }.get(seg.node_type, Colors.RESET)
                node_info = f" → {node_color}{seg.node_id}{Colors.RESET}"
            
            print(f"    {type_color}[{seg.segment_type}]{Colors.RESET} \"{seg.text}\"{node_info}")


def display_result_full(result: SafetyResult, verbose: bool = True):
    """Display the full critic pipeline result."""
    
    # Segmentation
    print_section("SEGMENTATION")
    if result.segments:
        for i, seg in enumerate(result.segments, 1):
            type_color = {
                "ACTION": Colors.GREEN,
                "THOUGHT": Colors.BLUE,
                "FEELING": Colors.CYAN,
                "META": Colors.DIM
            }.get(seg.segment_type, Colors.RESET)
            
            node_info = f" → {seg.node_id}" if seg.node_id else ""
            print(f"  [{i}] {type_color}{seg.segment_type}{Colors.RESET}: \"{seg.text}\"{node_info}")
    else:
        print(f"  {Colors.DIM}(Single segment, no parsing needed){Colors.RESET}")
    
    # Classification
    print_section("CLASSIFICATION")
    classified = [s for s in result.segments if s.node_id]
    if classified:
        for seg in classified:
            type_color = {
                "adaptive": Colors.GREEN,
                "maladaptive": Colors.RED,
                "neutral": Colors.YELLOW,
                "meta": Colors.DIM
            }.get(seg.node_type, Colors.RESET)
            
            print(f"  • \"{seg.text[:50]}{'...' if len(seg.text) > 50 else ''}\"")
            print(f"    {type_color}→ {seg.node_id} ({seg.node_type}){Colors.RESET}")
    else:
        print(f"  {Colors.DIM}No segments classified (possibly all META){Colors.RESET}")
    
    # Triage (if unsafe)
    if not result.is_safe:
        print_section("TRIAGE DECISION")
        print(f"  {Colors.RED}Primary risk selected:{Colors.RESET} {result.node_id}")
        print(f"  {Colors.RED}Severity:{Colors.RESET} {result.severity}")
        print(f"  {Colors.DIM}Category:{Colors.RESET} {result.category}")
        
        if result.adaptive_segments:
            print(f"\n  {Colors.GREEN}Adaptive parts to validate:{Colors.RESET}")
            for seg in result.adaptive_segments:
                print(f"    • {seg.node_id}: \"{seg.text[:50]}{'...' if len(seg.text) > 50 else ''}\"")
    
    # Verdict
    print_section("VERDICT")
    if result.is_safe:
        print_safe()
    else:
        print_unsafe(result.severity or "UNKNOWN")
    
    # Constraint message
    if verbose and result.constraint:
        print_section("CONSTRAINT MESSAGE (would be injected)")
        lines = result.constraint.split('\n')
        for line in lines:
            if line.strip():
                print(f"  {Colors.DIM}{line}{Colors.RESET}")
    
    # Correction strategies (if unsafe)
    if not result.is_safe and result.correction_strategies:
        print_section("CORRECTION STRATEGIES")
        for i, strategy in enumerate(result.correction_strategies[:3], 1):
            print(f"  [{i}] {Colors.GREEN}{strategy['target_node']}{Colors.RESET}")
            direction = strategy.get('therapeutic_direction', 'N/A')
            print(f"      {Colors.DIM}{direction[:80]}{'...' if len(direction) > 80 else ''}{Colors.RESET}")


def load_context(filepath: str) -> Optional[Dict]:
    """Load conversation context from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{Colors.RED}Error: Context file not found: {filepath}{Colors.RESET}")
        return None
    except json.JSONDecodeError as e:
        print(f"{Colors.RED}Error: Invalid JSON in context file: {e}{Colors.RESET}")
        return None


def display_context(context: Dict):
    """Display loaded context information - FULL TEXT, no truncation."""
    stage = context.get("stage", "rewriting")
    print_stage_banner(stage)
    
    # Show FULL nightmare summary in prominent box
    nightmare = context.get("nightmare_summary", context.get("nightmare", "No summary provided"))
    print_nightmare_box(nightmare)
    
    if "messages" in context:
        print(f"\n{Colors.DIM}Messages in history: {len(context['messages'])}{Colors.RESET}")
    
    if "notes" in context:
        print(f"{Colors.DIM}Testing notes: {context['notes']}{Colors.RESET}")


def display_conversation_history(conversation: Conversation):
    """Display full conversation history."""
    print_section("CONVERSATION HISTORY")
    
    for msg in conversation.messages:
        role_color = Colors.CYAN if msg.role == "user" else Colors.GREEN
        role_label = "You" if msg.role == "user" else "Therapist"
        stage_info = f" [{msg.stage}]" if msg.stage else ""
        
        print(f"{role_color}{Colors.BOLD}{role_label}{stage_info}:{Colors.RESET}")
        print(f"  {msg.content}\n")


# =============================================================================
# SCENARIO-BASED TESTING (NEW)
# =============================================================================

async def run_scenario(filepath: str, verbose: bool = False):
    """
    Run scenario-based test suite.
    
    A scenario file contains:
    - nightmare: The full nightmare text
    - test_cases: List of rewriting attempts to test against that nightmare
    
    Each test is evaluated in context of the nightmare.
    """
    print_header("SCENARIO TEST MODE")
    
    # Load scenario file
    try:
        with open(filepath, 'r') as f:
            scenario = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{Colors.RED}Error loading scenario file: {e}{Colors.RESET}")
        return
    
    # Extract scenario data
    nightmare = scenario.get("nightmare", scenario.get("nightmare_summary", ""))
    test_cases = scenario.get("test_cases", [])
    scenario_name = scenario.get("name", scenario.get("description", Path(filepath).stem))
    notes = scenario.get("notes", "")
    
    if not nightmare:
        print(f"{Colors.RED}Error: No nightmare found in scenario file{Colors.RESET}")
        return
    
    if not test_cases:
        print(f"{Colors.RED}Error: No test_cases found in scenario file{Colors.RESET}")
        return
    
    # Initialize critic
    try:
        critic = SafetyCritic()
        summary = critic.get_graph_summary()
        print(f"{Colors.GREEN}✓ Critic initialized{Colors.RESET}")
        print(f"  Graph: {summary['total_nodes']} nodes")
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to initialize critic: {e}{Colors.RESET}")
        return
    
    # Display scenario info
    print(f"\n{Colors.BOLD}Scenario:{Colors.RESET} {scenario_name}")
    if notes:
        print(f"{Colors.DIM}{notes}{Colors.RESET}")
    
    # Show the nightmare prominently
    print_nightmare_box(nightmare)
    
    print(f"\n{Colors.BOLD}Running {len(test_cases)} test cases against this nightmare...{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 70}{Colors.RESET}")
    
    # Run test cases
    results = []
    passed = 0
    failed = 0
    
    for i, case in enumerate(test_cases, 1):
        input_text = case.get("input", "")
        expected_safe = case.get("expected_safe")
        description = case.get("description", f"Test {i}")
        
        print(f"\n{Colors.BOLD}[{i}/{len(test_cases)}] {description}{Colors.RESET}")
        print(f"  {Colors.CYAN}Input:{Colors.RESET} \"{input_text}\"")
        
        try:
            # Pass nightmare context for accurate classification (e.g., emotional suppression detection)
            result = await critic.evaluate_intervention(input_text, nightmare_context=nightmare)
            
            # Show segmentation and result
            display_result_compact(result, show_segments=True)
            
            # Show full constraint if verbose and unsafe
            if verbose and not result.is_safe and result.constraint:
                print(f"\n  {Colors.BOLD}Constraint Message:{Colors.RESET}")
                print(f"  {Colors.DIM}{'─' * 60}{Colors.RESET}")
                for line in result.constraint.split('\n'):
                    if line.strip():
                        print(f"  {Colors.DIM}{line}{Colors.RESET}")
                print(f"  {Colors.DIM}{'─' * 60}{Colors.RESET}")
            
            # Check against expected
            if expected_safe is not None:
                if result.is_safe == expected_safe:
                    print(f"  {Colors.GREEN}✓ PASS (expected {'safe' if expected_safe else 'unsafe'}){Colors.RESET}")
                    passed += 1
                else:
                    print(f"  {Colors.RED}✗ FAIL (expected {'safe' if expected_safe else 'unsafe'}, got {'safe' if result.is_safe else 'unsafe'}){Colors.RESET}")
                    failed += 1
            else:
                print(f"  {Colors.BLUE}? No expectation defined{Colors.RESET}")
            
            results.append({
                "description": description,
                "input": input_text,
                "expected_safe": expected_safe,
                "actual_safe": result.is_safe,
                "node_id": result.node_id,
                "severity": result.severity,
                "segments": [(s.segment_type, s.text, s.node_id) for s in result.segments],
                "passed": expected_safe is None or result.is_safe == expected_safe
            })
            
        except Exception as e:
            print(f"  {Colors.RED}✗ Error: {e}{Colors.RESET}")
            failed += 1
            results.append({
                "description": description,
                "input": input_text,
                "error": str(e),
                "passed": False
            })
    
    # Summary
    print_header("SCENARIO TEST RESULTS")
    print(f"\n  {Colors.BOLD}Scenario:{Colors.RESET} {scenario_name}")
    print(f"  {Colors.BOLD}Nightmare:{Colors.RESET} {nightmare[:60]}...")
    print(f"\n  {Colors.GREEN}Passed:{Colors.RESET} {passed}")
    print(f"  {Colors.RED}Failed:{Colors.RESET} {failed}")
    print(f"  {Colors.DIM}Total:{Colors.RESET} {len(test_cases)}")
    
    if failed > 0:
        print(f"\n  {Colors.RED}Failed cases:{Colors.RESET}")
        for r in results:
            if not r.get("passed", False):
                expected = "safe" if r.get("expected_safe") else "unsafe"
                actual = "safe" if r.get("actual_safe") else "unsafe"
                if "error" in r:
                    print(f"    • {r['description']}: {r['error']}")
                else:
                    print(f"    • {r['description']}: expected {expected}, got {actual} ({r.get('node_id', 'N/A')})")
    
    # Pass rate
    total_with_expectations = sum(1 for r in results if r.get("expected_safe") is not None)
    if total_with_expectations > 0:
        pass_rate = (passed / total_with_expectations) * 100
        color = Colors.GREEN if pass_rate >= 80 else Colors.YELLOW if pass_rate >= 60 else Colors.RED
        print(f"\n  {Colors.BOLD}Pass rate:{Colors.RESET} {color}{pass_rate:.1f}%{Colors.RESET}")


# =============================================================================
# LEGACY BATCH TESTING (Updated with segmentation)
# =============================================================================

async def run_batch(filepath: str, verbose: bool = False):
    """Run batch test cases from file (legacy format without nightmare context)."""
    print_header("BATCH TEST MODE (Legacy)")
    print(f"{Colors.DIM}Note: For better testing, use --scenario with nightmare context{Colors.RESET}")
    
    try:
        with open(filepath, 'r') as f:
            test_cases = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{Colors.RED}Error loading batch file: {e}{Colors.RESET}")
        return
    
    # Initialize critic
    try:
        critic = SafetyCritic()
        print(f"{Colors.GREEN}✓ Critic initialized{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to initialize critic: {e}{Colors.RESET}")
        return
    
    print(f"\n{Colors.BOLD}Running {len(test_cases)} test cases...{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 70}{Colors.RESET}")
    
    # Run test cases
    results = []
    passed = 0
    failed = 0
    
    for i, case in enumerate(test_cases, 1):
        input_text = case.get("input", "")
        expected_safe = case.get("expected_safe")
        description = case.get("description", f"Test case {i}")
        
        print(f"\n{Colors.BOLD}[{i}/{len(test_cases)}] {description}{Colors.RESET}")
        print(f"  {Colors.CYAN}Input:{Colors.RESET} \"{input_text}\"")
        
        try:
            result = await critic.evaluate_intervention(input_text)
            
            # Show segmentation and result
            display_result_compact(result, show_segments=True)
            
            # Check against expected
            if expected_safe is not None:
                if result.is_safe == expected_safe:
                    print(f"  {Colors.GREEN}✓ PASS{Colors.RESET}")
                    passed += 1
                else:
                    print(f"  {Colors.RED}✗ FAIL (expected {'safe' if expected_safe else 'unsafe'}){Colors.RESET}")
                    failed += 1
            
            results.append({
                "description": description,
                "input": input_text,
                "expected_safe": expected_safe,
                "actual_safe": result.is_safe,
                "node_id": result.node_id,
                "passed": expected_safe is None or result.is_safe == expected_safe
            })
            
        except Exception as e:
            print(f"  {Colors.RED}Error: {e}{Colors.RESET}")
            failed += 1
            results.append({
                "description": description,
                "input": input_text,
                "error": str(e),
                "passed": False
            })
    
    # Summary
    print_header("BATCH RESULTS SUMMARY")
    print(f"  {Colors.GREEN}Passed:{Colors.RESET} {passed}")
    print(f"  {Colors.RED}Failed:{Colors.RESET} {failed}")
    
    if failed > 0:
        print(f"\n  {Colors.RED}Failed cases:{Colors.RESET}")
        for r in results:
            if not r.get("passed", False):
                print(f"    • {r['description']}: {r.get('error', 'Mismatch')}")


# =============================================================================
# LIVE MODE
# =============================================================================

async def run_live_mode(language: str = "en"):
    """
    Run through actual IRT chatbot flow until reaching rewriting stage,
    then test critic on user inputs.
    """
    # Lazy import to avoid triggering Langfuse initialization in non-live modes
    from AI.irt_app import process_chat_message

    print_header("LIVE IRT SESSION + CRITIC TESTING")
    print(f"{Colors.DIM}Chat through the IRT session naturally.{Colors.RESET}")
    print(f"{Colors.DIM}When you reach REWRITING stage, critic testing begins.{Colors.RESET}")
    
    # Initialize conversation
    session_id = str(uuid.uuid4())
    user_id = "test_user"
    conversation = Conversation(session_id=session_id, user_id=user_id, language=language)
    
    # Initialize critic (pass through if fails)
    try:
        critic = SafetyCritic()
        print(f"\n{Colors.GREEN}✓ Critic initialized{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to initialize critic: {e}{Colors.RESET}")
        critic = None
    
    current_stage = "recording"
    in_critic_mode = False
    nightmare_summary = None
    
    # Show initial prompt
    print_stage_banner(current_stage)
    print(f"\n{Colors.GREEN}{Colors.BOLD}Therapist:{Colors.RESET}")
    print(f"  I'm here to help you work through your nightmares.")
    print(f"  Take your time to describe your nightmare in as much detail as you can.\n")
    
    print(f"{Colors.BOLD}Commands:{Colors.RESET}")
    print(f"  {Colors.DIM}• Type naturally to chat with the IRT therapist{Colors.RESET}")
    print(f"  {Colors.DIM}• 'history' to see full conversation{Colors.RESET}")
    print(f"  {Colors.DIM}• 'skip' to jump to rewriting stage (loads example context){Colors.RESET}")
    print(f"  {Colors.DIM}• 'q' or 'quit' to exit{Colors.RESET}")
    
    while True:
        print(f"\n{Colors.BOLD}{'─' * 60}{Colors.RESET}")
        
        prompt_label = "Test rewriting" if in_critic_mode else "You"
        try:
            user_input = input(f"{Colors.BOLD}[{current_stage.upper()}] {prompt_label} > {Colors.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Colors.DIM}Exiting...{Colors.RESET}")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ('q', 'quit', 'exit'):
            print(f"{Colors.DIM}Goodbye!{Colors.RESET}")
            break
        
        if user_input.lower() == 'history':
            display_conversation_history(conversation)
            continue
        
        if user_input.lower() == 'skip':
            context = load_context("data/scenarios/chase_nightmare.json")
            if context:
                nightmare_summary = context.get("nightmare_summary", context.get("nightmare", ""))
                current_stage = "rewriting"
                in_critic_mode = True
                
                display_context(context)
                print(f"\n{Colors.YELLOW}Skipped to rewriting stage. Now testing critic.{Colors.RESET}")
                print(f"{Colors.DIM}Enter dream rewriting attempts to test:{Colors.RESET}")
            continue
        
        if user_input.lower() == 'examples' and in_critic_mode:
            print_example_cases()
            continue
        
        if user_input.lower() == 'graph' and in_critic_mode:
            if critic:
                print_graph_summary(critic)
            continue
        
        # If in critic testing mode (rewriting stage), run through critic
        if in_critic_mode and critic:
            try:
                # Pass nightmare context for accurate classification (e.g., emotional suppression detection)
                result = await critic.evaluate_intervention(user_input, nightmare_context=nightmare_summary)
                display_result_full(result)
            except Exception as e:
                print(f"{Colors.RED}✗ Critic error (pass-through): {e}{Colors.RESET}")
            continue
        
        # Otherwise, process through actual IRT chatbot
        try:
            chat_input = ChatInput(
                session_id=session_id,
                message=user_input,
                user_id=user_id,
                language_override=language
            )
            
            response = await process_chat_message(chat_input, conversation)
            current_stage = response.stage
            
            if current_stage == "rewriting" and not in_critic_mode:
                in_critic_mode = True
                
                # Accumulate all user messages from recording stage as nightmare context
                user_messages = []
                for msg in conversation.messages:
                    if msg.role == "user" and msg.stage == "recording":
                        user_messages.append(msg.content)
                    elif msg.role == "user" and len(msg.content) > 50:
                        # Fallback: include substantial user messages even if stage not tagged
                        user_messages.append(msg.content)
                
                if user_messages:
                    nightmare_summary = " ".join(user_messages)
                else:
                    # Last resort: grab the longest user message
                    for msg in conversation.messages:
                        if msg.role == "user" and len(msg.content) > 100:
                            nightmare_summary = msg.content
                            break
                
                print_stage_banner(current_stage)
                
                if nightmare_summary:
                    print_nightmare_box(nightmare_summary)
                
                print(f"\n{Colors.GREEN}{Colors.BOLD}Therapist:{Colors.RESET}")
                print(f"  {response.response}\n")
                
                print(f"\n{Colors.YELLOW}{'━' * 60}{Colors.RESET}")
                print(f"{Colors.YELLOW}{Colors.BOLD}  CRITIC TESTING MODE ACTIVATED{Colors.RESET}")
                print(f"{Colors.YELLOW}{'━' * 60}{Colors.RESET}")
                print(f"{Colors.DIM}Your rewriting attempts will now be analyzed by the critic.{Colors.RESET}")
            else:
                print_stage_banner(current_stage)
                print(f"\n{Colors.GREEN}{Colors.BOLD}Therapist:{Colors.RESET}")
                print(f"  {response.response}")
                
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

async def run_interactive(context: Optional[Dict] = None):
    """Run interactive testing loop with static context."""
    print_header("CRITIC INTERACTIVE TEST")
    print(f"{Colors.DIM}Testing the SafetyCritic 'Clinical Reasoning Engine'{Colors.RESET}")
    
    # Initialize critic
    try:
        critic = SafetyCritic()
        summary = critic.get_graph_summary()
        print(f"\n{Colors.GREEN}✓ Critic initialized{Colors.RESET}")
        print(f"  Graph: {summary['total_nodes']} nodes ({summary['maladaptive_nodes']} mal / {summary['adaptive_nodes']} adapt / {summary['neutral_nodes']} neut)")
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to initialize critic: {e}{Colors.RESET}")
        critic = None
    
    if context:
        display_context(context)
    
    print(f"\n{Colors.BOLD}Commands:{Colors.RESET}")
    print(f"  {Colors.DIM}• Type a dream rewriting attempt to test{Colors.RESET}")
    print(f"  {Colors.DIM}• 'examples' to see test cases{Colors.RESET}")
    print(f"  {Colors.DIM}• 'graph' to show graph summary{Colors.RESET}")
    print(f"  {Colors.DIM}• 'context' to re-show nightmare{Colors.RESET}")
    print(f"  {Colors.DIM}• 'q' or 'quit' to exit{Colors.RESET}")
    
    while True:
        print(f"\n{Colors.BOLD}{'─' * 60}{Colors.RESET}")
        stage = context.get("stage", "rewriting") if context else "rewriting"
        try:
            user_input = input(f"{Colors.BOLD}[{stage.upper()}] Enter rewriting attempt > {Colors.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Colors.DIM}Exiting...{Colors.RESET}")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ('q', 'quit', 'exit'):
            print(f"{Colors.DIM}Goodbye!{Colors.RESET}")
            break
        
        if user_input.lower() == 'examples':
            print_example_cases()
            continue
        
        if user_input.lower() == 'graph':
            if critic:
                print_graph_summary(critic)
            continue
        
        if user_input.lower() == 'context' and context:
            display_context(context)
            continue
        
        if critic:
            try:
                # Extract nightmare context if available
                nightmare_ctx = None
                if context:
                    nightmare_ctx = context.get("nightmare_summary", context.get("nightmare", None))
                
                # Pass nightmare context for accurate classification (e.g., emotional suppression detection)
                result = await critic.evaluate_intervention(user_input, nightmare_context=nightmare_ctx)
                display_result_full(result)
            except Exception as e:
                print(f"{Colors.RED}✗ Critic error (pass-through): {e}{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}  [Pass-through mode] No safety check performed{Colors.RESET}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_example_cases():
    """Print example test cases."""
    print_section("EXAMPLE TEST CASES")
    examples = [
        ("Simple adaptive", "I fight back against the monster"),
        ("Simple maladaptive (escape)", "I wake myself up"),
        ("Simple maladaptive (violence)", "I kill everyone in the room"),
        ("Simple maladaptive (avoidance)", "I hide under the bed"),
        ("Neutral (feeling)", "I felt really scared"),
        ("Neutral (narrative)", "There was a dark forest"),
        ("Negation (should be adaptive)", "I decided not to run away"),
        ("Mixed (META + action)", "This is stupid, but I guess I punch the monster"),
        ("Mixed (feeling + action)", "I feel terrified but I stand my ground"),
        ("Mixed (feeling + violence)", "I was so angry I killed everyone"),
    ]
    
    for name, example in examples:
        print(f"  {Colors.DIM}{name}:{Colors.RESET}")
        print(f"    \"{example}\"")


def print_graph_summary(critic: SafetyCritic):
    """Print knowledge graph summary."""
    print_section("KNOWLEDGE GRAPH SUMMARY")
    summary = critic.get_graph_summary()
    
    print(f"  {Colors.RED}Maladaptive:{Colors.RESET} {summary['maladaptive_nodes']}")
    print(f"  {Colors.GREEN}Adaptive:{Colors.RESET} {summary['adaptive_nodes']}")
    print(f"  {Colors.YELLOW}Neutral:{Colors.RESET} {summary['neutral_nodes']}")
    print(f"  {Colors.DIM}Edges:{Colors.RESET} {summary['total_edges']}")
    print(f"\n  {Colors.DIM}All nodes:{Colors.RESET}")
    for node_id in summary['node_ids']:
        print(f"    • {node_id}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Interactive Critic Testing Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --live                        Chat through IRT naturally
  %(prog)s --context data/scenarios/chase.json  Test with loaded nightmare
  %(prog)s --scenario data/scenarios/chase_tests.json  Run scenario test suite
  %(prog)s --batch data/scenarios/batch_tests.json     Legacy batch testing
  %(prog)s --scenario FILE -v            Verbose: show constraints

See docs/testing/Testing_Guide.md for full documentation.
        """
    )
    parser.add_argument("--live", "-l", action="store_true", 
                        help="Live mode: Chat through actual IRT stages")
    parser.add_argument("--context", "-c", type=str, 
                        help="Path to context JSON file for interactive testing")
    parser.add_argument("--scenario", "-s", type=str,
                        help="Path to scenario test file (nightmare + test cases)")
    parser.add_argument("--batch", "-b", type=str, 
                        help="Path to batch test cases JSON file (legacy)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full constraint messages in scenario/batch mode")
    parser.add_argument("--language", type=str, default="en",
                        help="Language for live mode (en/de)")
    args = parser.parse_args()
    
    if args.scenario:
        asyncio.run(run_scenario(args.scenario, verbose=args.verbose))
    elif args.batch:
        asyncio.run(run_batch(args.batch, verbose=args.verbose))
    elif args.live:
        asyncio.run(run_live_mode(args.language))
    else:
        context = load_context(args.context) if args.context else None
        asyncio.run(run_interactive(context))


if __name__ == "__main__":
    main()
