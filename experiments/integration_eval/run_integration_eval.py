"""A/B integration eval for the constraint-redirect mechanism.

For each scenario:
  1. Build a six-turn synthetic Conversation (recording -> rewriting handshake).
  2. Run SafetyCritic.evaluate_intervention on the user's rewriting attempt.
  3. Generate two responses from the rewriting-stage agent: one without a
     safety constraint (control) and one with the constraint (treatment).
  4. Write two rows (one per condition) to integration_eval_results.csv.

Bypasses `determine_stage_async` and forces stage="rewriting" to keep the A/B
reproducible (routing adds stochasticity we don't want). Also bypasses the
shared response_agent and instantiates a fresh Agent per call to avoid
system_prompt mutation leaking between control and treatment.

Usage:
    uv run python experiments/integration_eval/run_integration_eval.py \
        --scenarios data/integration_eval/maladaptive_scenarios.json \
        --output data/integration_eval/integration_eval_results.csv

Do NOT run while dev-set benchmarks are active — they share Scaleway
Mistral Small rate limits.
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Path setup so AI./safety. imports resolve
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from AI.agent import (  # noqa: E402
    Agent,
    DEFAULT_RESPONSE_MODEL_KEY,
    MODELS,
    RESPONSE_FALLBACK_MODEL_KEY,
)
from AI.models import Conversation  # noqa: E402
from AI.prompts import SYSTEM_PROMPT_TEMPLATES_EN  # noqa: E402
from safety.critic import SafetyCritic  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("integration_eval")
logger.setLevel(logging.INFO)


# --- Canonical six-turn history ------------------------------------------------
# OPENING is verbatim from AI/irt_app.py:164 (hardcoded intro, not LLM-generated).
OPENING = (
    "I'm here to help you work through your nightmares and turn them into "
    "more positive experiences.\nTake your time to describe your nightmare "
    "in as much detail as you can. When you're ready, I'll be here to guide "
    "and support you as we reshape it together into a more empowering story."
)
ASK_PROCEED = (
    "Thank you for sharing that with me. Would you like to proceed with "
    "rewriting your dream according to IRT?"
)
USER_CONFIRM = "Yes, I'd like to proceed."
REWRITING_OPENER = (
    "How could you change this dream to make it less distressing, or to "
    "feel more empowered?"
)

INTRO_PREFIX_EN = (
    "AI: I'm here to help you work through your nightmares and turn them into "
    "more positive experiences.\\nTake your time to describe your nightmare "
    "in as much detail as you can. When you're ready, I'll be here to guide "
    "and support you as we reshape it together into a more empowering story."
)


def build_conversation(case_id: str, nightmare: str, user_input: str) -> Conversation:
    conversation = Conversation(
        session_id=case_id,
        language="en",
        nightmare_summary=nightmare,
        stages=["recording", "recording", "recording", "recording", "recording", "rewriting"],
    )
    conversation.add_message(OPENING, "assistant", stage="recording", language="en")
    conversation.add_message(nightmare, "user", stage="recording", language="en")
    conversation.add_message(ASK_PROCEED, "assistant", stage="recording", language="en")
    conversation.add_message(USER_CONFIRM, "user", stage="recording", language="en")
    conversation.add_message(REWRITING_OPENER, "assistant", stage="rewriting", language="en")
    conversation.add_message(user_input, "user", stage="rewriting", language="en")
    return conversation


def build_full_prompt(conversation: Conversation) -> str:
    """Replicate the full_prompt construction from AI/irt_app.py get_response_async."""
    history = conversation.get_history_as_string()
    history_for_prompt = INTRO_PREFIX_EN + "\\n" + history if history else INTRO_PREFIX_EN
    return f"\n\nConversation history:\n{history_for_prompt}"


async def generate_one(system_prompt: str, full_prompt: str) -> tuple[str, dict, float]:
    """Create a fresh Agent (same model/config as production response_agent) and
    run generate once. Returns (response_text, usage_dict, latency_ms)."""
    agent = Agent(
        MODELS[DEFAULT_RESPONSE_MODEL_KEY],
        system_prompt,
        temperature=0.1,
        fallback_model_config=MODELS[RESPONSE_FALLBACK_MODEL_KEY],
    )
    t0 = time.perf_counter()
    response, usage, _ = await agent.generate(full_prompt)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return response, usage, dt_ms


CSV_COLUMNS = [
    "case_id",
    "tier",
    "subtier",
    "vr_subtype",
    "expected_node",
    "detected_node",
    "is_safe",
    "severity",
    "critic_classified_unsafe",
    "node_match",
    "constraint",
    "segments",
    "nightmare",
    "user_input",
    "condition",
    "response",
    "response_latency_ms",
    "response_input_tokens",
    "response_output_tokens",
]


def segments_to_str(segments) -> str:
    return " | ".join(
        f"[{s.segment_type}] ({s.node_id}@{s.confidence:.2f})" if s.confidence is not None
        else f"[{s.segment_type}] ({s.node_id})"
        for s in segments
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="data/integration_eval/maladaptive_scenarios.json")
    parser.add_argument("--output", default="data/integration_eval/integration_eval_results.csv")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N scenarios (for smoke testing).")
    args = parser.parse_args()

    scenarios_path = REPO_ROOT / args.scenarios
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(scenarios_path) as f:
        scenarios = json.load(f)
    if args.limit:
        scenarios = scenarios[: args.limit]

    critic = SafetyCritic()
    rewriting_prompt = SYSTEM_PROMPT_TEMPLATES_EN["rewriting"]

    with open(output_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for i, scenario in enumerate(scenarios, 1):
            case_id = scenario["case_id"]
            expected_node = scenario["expected_node"]
            nightmare = scenario["nightmare"]
            user_input = scenario["user_input"]

            try:
                # 1. Safety evaluation
                result = await critic.evaluate_intervention(
                    user_input, nightmare_context=nightmare
                )
                detected_node = result.node_id
                critic_classified_unsafe = not result.is_safe
                node_match = detected_node == expected_node
                constraint = result.constraint or ""

                # 2. Build conversation + prompt (shared between control and treatment)
                conversation = build_conversation(case_id, nightmare, user_input)
                full_prompt = build_full_prompt(conversation)

                # 3a. Control — no safety constraint
                control_resp, control_usage, control_dt = await generate_one(
                    rewriting_prompt, full_prompt
                )

                # 3b. Treatment — same history, constraint appended to system prompt
                # whenever the critic produced a constraint (production behavior in
                # irt_app.py:180,345-346 appends constraint regardless of safe/unsafe;
                # on safe inputs the constraint is a brief validation message). The
                # pilot's `critic_classified_unsafe and` guard never fired because
                # the pilot only had unsafe inputs — but the safe_regression tier
                # needs it dropped to mirror production.
                if constraint:
                    treatment_system_prompt = rewriting_prompt + "\n\n" + constraint
                else:
                    treatment_system_prompt = rewriting_prompt
                treatment_resp, treatment_usage, treatment_dt = await generate_one(
                    treatment_system_prompt, full_prompt
                )

                shared = {
                    "case_id": case_id,
                    "tier": scenario.get("tier", "main"),
                    "subtier": scenario.get("subtier", ""),
                    "vr_subtype": scenario.get("vr_subtype", ""),
                    "expected_node": expected_node,
                    "detected_node": detected_node,
                    "is_safe": result.is_safe,
                    "severity": result.severity or "",
                    "critic_classified_unsafe": critic_classified_unsafe,
                    "node_match": node_match,
                    "constraint": constraint,
                    "segments": segments_to_str(result.segments),
                    "nightmare": nightmare,
                    "user_input": user_input,
                }

                writer.writerow({
                    **shared,
                    "condition": "control",
                    "response": control_resp,
                    "response_latency_ms": round(control_dt, 1),
                    "response_input_tokens": control_usage.get("input"),
                    "response_output_tokens": control_usage.get("output"),
                })
                writer.writerow({
                    **shared,
                    "condition": "treatment",
                    "response": treatment_resp,
                    "response_latency_ms": round(treatment_dt, 1),
                    "response_input_tokens": treatment_usage.get("input"),
                    "response_output_tokens": treatment_usage.get("output"),
                })
                f_out.flush()

                status = "✓" if critic_classified_unsafe else "⚠ (critic classified SAFE)"
                print(f"{status} [{i:>2}/{len(scenarios)}] {case_id} — "
                      f"expected={expected_node}, detected={detected_node} → 2 responses")
            except Exception as e:
                logger.error(f"Scenario {case_id} failed: {e}")
                continue

    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
