"""
Block B: The Entity Linker (Clinical Classifier)

Uses an LLM with "Label Engineering" to map user text to clinical graph node IDs.
The classifier acts as a "Clinical Coder" that identifies whether
user input represents adaptive or maladaptive dream behaviors.

Uses semantically clear single-token codes for reliable logprobs-based
confidence scoring via length-normalized joint probability.
First-token Margin of Victory is retained as a secondary diagnostic metric.
"""

import os
import sys
import math
import time
from typing import Optional, Tuple
import logging
from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AI.agent import Agent, ModelConfig, MODELS, LogprobsResult
from .graph_definitions import build_clinical_graph, get_all_node_ids, get_maladaptive_nodes, get_adaptive_nodes

logger = logging.getLogger(__name__)


# ============================================================================
# LABEL ENGINEERING: Semantic single-token codes for reliable classification
# ============================================================================
CODE_MAP = {
    # Maladaptive (Risk) codes
    "HIDE": "AVOIDANCE",                       # Avoidance, freezing, hiding
    "ESCAPE": "INTERRUPTION",                  # Waking up, dream ending
    "DESTROY": "VIOLENT_REVENGE",              # Extreme violence, rage
    "DENY": "SUPPRESSION",                     # Emotional denial, numbing
    "REPLAY": "TRAUMA_REPLAY",                 # Re-experiencing without change
    # Adaptive (Goal) codes
    "CONFRONT": "BEHAVIORAL_MASTERY",          # Fighting back, active response
    "HELP": "SOCIAL_MASTERY",                  # Seeking/receiving help
    "CHANGE": "ENVIRONMENTAL_MASTERY",         # Using tools, changing setting
    "RELAX": "EMOTIONAL_MASTERY",              # Calming, self-regulation
    "POWER": "MYTHICAL_MASTERY",               # Magic, supernatural abilities
    # Neutral codes
    "LOOK": "NARRATIVE_SETTING",               # Scene description
    "FEEL": "AFFECT_EXPRESSION",               # Emotion expression
}

# Reverse map for debugging
NODE_TO_CODE = {v: k for k, v in CODE_MAP.items()}


def normalize_classification_code(raw_code: object) -> Optional[str]:
    """Normalize superficial formatting and require an exact classifier code.

    The classifier prompt defines a closed set of output codes. This helper only
    removes surrounding whitespace and common presentation punctuation; it does
    not infer a code from semantic keywords, substrings, or misspellings.

    Args:
        raw_code: Value returned by the direct classifier or parsed from CoT JSON.

    Returns:
        A valid code from ``CODE_MAP``, or ``None`` when the value is invalid.
    """
    if not isinstance(raw_code, str):
        return None

    normalized = raw_code.upper().strip()
    normalized = normalized.strip("`*_\"'.,;:!? \t\r\n")
    return normalized if normalized in CODE_MAP else None


@dataclass
class ClassificationResult:
    """Result of classification with length-normalized joint probability confidence."""
    node_id: str
    code: str
    confidence: float  # Length-normalized joint probability: exp(mean_logprob)
    raw_response: str
    mean_logprob: Optional[float] = None
    margin: Optional[float] = None  # Margin of Victory (diagnostic): prob_winner - prob_runner_up
    prob_winner: Optional[float] = None  # First-token winner probability (diagnostic)
    prob_runner_up: Optional[float] = None  # First-token runner-up probability (diagnostic)
    # Logprob provenance (System 2 diagnostic): which token span produced `confidence`.
    # "key_scan"   — code token(s) located after the "classification" key (intended path)
    # "tail_scan"  — code token matched in the final third of the response (fallback scan)
    # "full_seq"   — code token span not found; confidence is over the whole response
    logprob_source: Optional[str] = None
    full_seq_confidence: Optional[float] = None  # exp(mean logprob) over all response tokens
    code_token_count: Optional[int] = None  # Number of tokens in the located code span
    response_token_count: Optional[int] = None  # Total tokens in the response
    # Computational cost metrics
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None  # Wall-clock time for the API call in milliseconds


# System prompt for the Clinical Coder with Label Engineering
CLINICAL_CODER_PROMPT = """You are a Clinical Coder for Imagery Rehearsal Therapy (IRT). Classify dream actions into one of 12 codes.

## CODES (respond with ONE word only)

### MALADAPTIVE (Risk):
- **HIDE** - Avoidance: hiding, freezing by choice, looking away, passive observation, choosing to do nothing
- **ESCAPE** - Dream interruption: forcing awake, waking up, dream ending abruptly, refusing to continue
- **DESTROY** - Extreme violence: brutal excessive violence, torture, mutilation, "killing everyone", uncontrolled rage
- **DENY** - Emotional suppression: claiming to feel nothing, denying distress in a distressing context, emotional numbing, "I'm fine with it" when clearly not
- **REPLAY** - Trauma replay: re-experiencing trauma without change, rehearsing the nightmare identically, refusing to alter the narrative, reliving events exactly as they happened

### ADAPTIVE (Goal):
- **CONFRONT** - Active response: fighting back proportionately, running purposefully, taking control, direct action
- **HELP** - Social support: calling for help, allies appearing, working with others, asking for assistance
- **CHANGE** - Environment/Tools: finding weapons/tools, using objects, environment changing favorably, manipulating setting
- **RELAX** - Self-regulation: calming down, deep breathing, accepting peacefully, transforming fear, staying present
- **POWER** - Supernatural: magical powers, divine intervention, superpowers, impossible abilities

### NEUTRAL:
- **LOOK** - Scene description: describing environment, characters, or setting without action
- **FEEL** - Emotion expression: stating feelings ("I felt scared", "I was terrified") without behavioral strategy

## CRITICAL RULES

1. **Violence Threshold**
   - Controlled defense (punch, push, shoot to stop) -> CONFRONT
   - Excessive/rage violence (stab repeatedly, torture, destroy everything) -> DESTROY

2. **Feeling vs. Avoidance**
   - Expressing emotion ("I feel terrified", "I can't move") -> FEEL
   - Actively disengaging ("I try to feel nothing", "I make my mind blank") -> HIDE

3. **Mastery Priority**
   - Seeking help/communication -> HELP
   - Using objects/changing setting -> CHANGE
   - Direct physical action -> CONFRONT

4. **Negation**: "I decided NOT to run" -> CONFRONT (active choice to stay)

5. **Mental Actions**
   - "I imagine a shield", "I visualize protection" -> POWER or CHANGE
   - "I focus on breathing", "I tell myself I'm safe" -> RELAX
   - "I try to not feel", "I go blank" -> DENY

6. **Emotional Regulation vs Suppression**
   - Actively managing emotions while engaged: "I stay calm and face it" -> RELAX
   - Denying emotions in distressing context: "I feel nothing about my dead dog" -> DENY
   - Key: RELAX involves engagement; DENY involves emotional shutdown

7. **Trauma Replay vs Scene Description**
   - Describing the setting: "There was a dark forest" -> LOOK
   - Reliving trauma without change: "It all happens exactly the same" -> REPLAY
   - Key: REPLAY means re-experiencing the trauma identically; LOOK is passive observation

## EXAMPLES
- "I hide under the bed" -> HIDE
- "I wake myself up" -> ESCAPE
- "I kill everyone brutally" -> DESTROY
- "I don't feel anything about it" -> DENY
- "It all happens exactly the same way again" -> REPLAY
- "I punch the attacker" -> CONFRONT
- "I call for help" -> HELP
- "I find a sword" -> CHANGE
- "I take a deep breath" -> RELAX
- "I use magic to fly" -> POWER
- "There was a dark forest" -> LOOK
- "I felt terrified" -> FEEL

## RESPONSE FORMAT
Respond with ONLY ONE CODE from: HIDE, ESCAPE, DESTROY, DENY, REPLAY, CONFRONT, HELP, CHANGE, RELAX, POWER, LOOK, FEEL

No explanations. Just the code.
"""

# ============================================================================
# CONTEXT-AWARE EXAMPLES: Appended when use_context_examples=True
# Teaches the model that nightmare context changes classification.
# Clinically grounded — not derived from any benchmark data.
# ============================================================================
CONTEXT_AWARE_EXAMPLES = """

## CONTEXT-AWARE EXAMPLES
When nightmare context is provided, use it to disambiguate. The same surface action can classify differently depending on context.

- [NIGHTMARE CONTEXT] A shadowy figure was chasing me through endless hallways
  [USER'S REWRITING ATTEMPT] I just keep running and running
  -> HIDE (fleeing without agency or destination is avoidance, not purposeful action)

- [NIGHTMARE CONTEXT] A shadowy figure was chasing me through endless hallways
  [USER'S REWRITING ATTEMPT] I sprint toward the exit at the end of the hall
  -> CONFRONT (running with purpose and direction is active behavioral mastery)

- [NIGHTMARE CONTEXT] I watched my childhood home burn down with my family inside
  [USER'S REWRITING ATTEMPT] I'm fine, it doesn't bother me at all
  -> DENY (claiming no distress about deeply distressing content is emotional suppression)
"""


class ClinicalClassifier:
    """
    Classifies user dream interventions into clinical categories using an LLM.
    
    Uses the Agent class to send classification requests and validates
    that returned codes exist in the clinical knowledge graph.
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None, use_context_examples: bool = False):
        """
        Initialize the Clinical Classifier.

        Args:
            model_config: Optional model configuration. Defaults to MISTRAL_NEMO for speed.
            use_context_examples: If True, append context-aware few-shot examples to the prompt.
                                  Default False preserves the original prompt for backward compatibility.
        """
        # Use Scaleway Mistral Small for classification (logprobs required, not supported on OpenRouter)
        self.model_config = model_config or MODELS["MISTRAL_SMALL"]
        self.use_context_examples = use_context_examples

        # Build prompt: base + optional context-aware examples
        prompt = CLINICAL_CODER_PROMPT
        if use_context_examples:
            prompt = CLINICAL_CODER_PROMPT + CONTEXT_AWARE_EXAMPLES
            logger.info("Using context-aware examples (enhanced prompt)")

        # Initialize the agent with clinical coder prompt
        self.agent = Agent(
            model_config=self.model_config,
            system_prompt=prompt,
            temperature=0  # Deterministic baseline; consistency mode overrides to 0.7
        )
        
        # Load the graph to validate node IDs
        self.graph = build_clinical_graph()
        self.valid_node_ids = set(get_all_node_ids(self.graph))
        
        logger.info(f"ClinicalClassifier initialized with model: {self.model_config.name}")
        logger.info(f"Valid node IDs: {self.valid_node_ids}")
    
    async def classify_intent(
        self, 
        user_text: str, 
        context: Optional[str] = None,
        return_logprobs: bool = False
    ) -> Tuple[str, Optional[ClassificationResult]]:
        """
        Classify user dream intervention text into a clinical category.

        Uses Label Engineering with single-token codes and calculates
        confidence via length-normalized joint probability across all tokens.
        First-token Margin of Victory is retained as a diagnostic metric.
        
        Args:
            user_text: The user's proposed dream action/intervention
            context: Optional context about the original nightmare for disambiguation
            return_logprobs: Whether to request and return log probabilities
            
        Returns:
            Tuple of (node_id, ClassificationResult or None)
            
        Invalid outputs are routed to UNCLASSIFIED rather than inferred.
        """
        try:
            # Build context-aware message if nightmare context is provided
            if context:
                classification_input = f"""[NIGHTMARE CONTEXT]
{context}

[USER'S REWRITING ATTEMPT]
{user_text}

Classify the rewriting attempt. If user claims to feel nothing about distressing content, use DENY. If user replays the trauma without any change, use REPLAY."""
            else:
                classification_input = user_text
            
            # Always request logprobs for margin calculation
            t0 = time.perf_counter()
            response, usage, logprobs = await self.agent.generate(
                classification_input,
                logprobs=True,
                top_logprobs=5
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            
            # Parse the code from response
            raw_response = response.strip()
            code = normalize_classification_code(raw_response)
            
            # Primary confidence: length-normalized joint probability from LogprobsResult
            confidence = 0.0
            mean_logprob_value = None
            # Diagnostic: first-token Margin of Victory
            margin = None
            prob_winner = None
            prob_runner_up = None

            if logprobs and logprobs.tokens:
                # Primary: use LogprobsResult.confidence (exp of mean logprob across all tokens)
                confidence = logprobs.confidence
                mean_logprob_value = logprobs.mean_logprob

                # Diagnostic: calculate first-token Margin of Victory
                first_token = logprobs.tokens[0]
                if first_token.top_logprobs and len(first_token.top_logprobs) >= 1:
                    prob_winner = math.exp(first_token.top_logprobs[0]["logprob"])
                    if len(first_token.top_logprobs) >= 2:
                        prob_runner_up = math.exp(first_token.top_logprobs[1]["logprob"])
                    else:
                        prob_runner_up = 0.0
                    margin = prob_winner - prob_runner_up
            
            # Map code to node ID
            node_id = CODE_MAP.get(code) if code is not None else None

            if node_id is None:
                logger.warning(
                    f"Invalid classifier response '{raw_response}'. Routing to UNCLASSIFIED."
                )
                node_id = "UNCLASSIFIED"
                code = "UNCLASSIFIED"
                confidence = 0.0
            
            logger.info(f"Classification: '{user_text[:50]}...' -> {code} ({node_id}) [confidence: {confidence:.3f}, margin: {margin}]")

            # Build result with all fields
            result = ClassificationResult(
                node_id=node_id,
                code=code,
                confidence=confidence,
                raw_response=raw_response,
                mean_logprob=mean_logprob_value,
                margin=margin,
                prob_winner=prob_winner,
                prob_runner_up=prob_runner_up,
                input_tokens=usage.get("input"),
                output_tokens=usage.get("output"),
                total_tokens=usage.get("total"),
                latency_ms=latency_ms,
            ) if return_logprobs else None
            
            return node_id, result
            
        except Exception as e:
            logger.error(f"Error classifying intent: {str(e)}")
            raise
    
    async def classify_with_confidence(
        self, 
        user_text: str, 
        context: Optional[str] = None,
        include_logprobs: bool = True
    ) -> Tuple[str, dict]:
        """
        Classify user text and return additional metadata including confidence.
        
        Args:
            user_text: The user's proposed dream action/intervention
            context: Optional context about the original nightmare for disambiguation
            include_logprobs: Whether to include logprobs-based confidence (default True)
            
        Returns:
            Tuple of (node_id, metadata_dict with confidence info)
        """
        node_id, result = await self.classify_intent(
            user_text, 
            context=context,
            return_logprobs=include_logprobs
        )
        
        # Get node attributes from graph
        node_attrs = dict(self.graph.nodes[node_id])
        
        metadata = {
            "node_id": node_id,
            "node_type": node_attrs.get("type"),
            "category": node_attrs.get("category"),
            "description": node_attrs.get("description"),
            "is_maladaptive": node_attrs.get("type") == "maladaptive",
            "input_text": user_text
        }
        
        # Add confidence metrics from classification result if available
        if result:
            metadata["code"] = result.code
            metadata["confidence"] = result.confidence  # Length-normalized joint probability
            metadata["mean_logprob"] = result.mean_logprob
            metadata["margin"] = result.margin  # First-token Margin of Victory (diagnostic)
            metadata["prob_winner"] = result.prob_winner
            metadata["prob_runner_up"] = result.prob_runner_up
        
        return node_id, metadata
    
    def get_node_info(self, node_id: str) -> dict:
        """
        Get information about a specific node.
        
        Args:
            node_id: The node ID to look up
            
        Returns:
            Dictionary of node attributes
        """
        if node_id not in self.graph:
            raise ValueError(f"Node '{node_id}' not found in graph")
        
        return dict(self.graph.nodes[node_id])


# For testing
if __name__ == "__main__":
    import asyncio
    
    async def test_classifier():
        classifier = ClinicalClassifier()
        
        test_cases = [
            # Maladaptive
            "I hide under the bed",
            "I wake myself up to escape the nightmare",
            "I kill everyone in the room brutally",
            "I do nothing and just watch",
            # Adaptive
            "I punch the attacker in the face",
            "I call for my friend to help me",
            "I find a magical sword",
            "I take a deep breath and stay calm",
            "I run away as fast as I can",
            # Neutral - Narrative
            "There was a dark forest",
            "I saw a monster in the corner",
            # Neutral - Affect
            "I felt terrified",
            "I was scared",
            # Negation handling
            "I decided not to run away",
            "I chose to stay instead of hiding",
        ]
        
        print("=" * 70)
        print("Clinical Classifier Test - Label Engineering with Joint Probability Confidence")
        print("=" * 70)
        
        for test in test_cases:
            try:
                node_id, metadata = await classifier.classify_with_confidence(test)
                node_type = metadata.get("node_type", "unknown")
                code = metadata.get("code", "?")
                confidence = metadata.get("confidence", 0)
                margin = metadata.get("margin")
                mean_lp = metadata.get("mean_logprob")

                if node_type == "maladaptive":
                    status = "⚠️ MALADAPTIVE"
                elif node_type == "adaptive":
                    status = "✅ ADAPTIVE"
                else:
                    status = "🟡 NEUTRAL"

                print(f"\n{status} [{code}]")
                print(f"  Input: '{test}'")
                print(f"  Node: {node_id}")
                print(f"  Confidence: {confidence:.3f} (mean_logprob={mean_lp})")
                if margin is not None:
                    print(f"  Margin (diagnostic): {margin:.3f}")
            except Exception as e:
                print(f"\n❌ ERROR for '{test}': {e}")
    
    asyncio.run(test_classifier())
