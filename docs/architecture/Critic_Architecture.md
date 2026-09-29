# SafetyCritic Architecture: Technical Deep Dive

This document provides a detailed technical explanation of the SafetyCritic "Clinical Reasoning Engine" - a neuro-symbolic safety layer for Imagery Rehearsal Therapy (IRT) chatbots.

## Table of Contents
1. [Overview](#overview)
2. [Three-Block Architecture](#three-block-architecture)
3. [Pipeline Flow](#pipeline-flow)
4. [Block A: Knowledge Graph](#block-a-knowledge-graph)
5. [Block B: Clinical Classifier](#block-b-clinical-classifier)
6. [Block C: SafetyCritic Pipeline](#block-c-safetycritic-pipeline)
7. [Data Structures](#data-structures)
8. [Constraint Generation](#constraint-generation)
9. [Integration Points](#integration-points)

---

## Overview

The SafetyCritic is a **neuro-symbolic** system that combines:
- **Symbolic reasoning**: A clinical knowledge graph encoding therapeutic guidelines
- **Neural processing**: LLMs for text understanding, segmentation, and classification

### Purpose
When a user proposes how to change their nightmare during IRT, the SafetyCritic evaluates whether the proposal is therapeutically safe (adaptive) or potentially harmful (maladaptive), and generates appropriate guidance for the chatbot.

### Key Innovation
Unlike simple keyword filters, the SafetyCritic:
1. Understands complex, multi-part inputs ("I hate this, but I'll fight back")
2. Distinguishes feelings from actions ("I felt scared" vs "I hid")
3. Handles negation ("I decided NOT to run" = adaptive)
4. Prioritizes the most critical risk when multiple issues exist

---

## Three-Block Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SafetyCritic                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────┐   ┌─────────────────┐   ┌──────────────────┐  │
│   │    Block A      │   │    Block B      │   │    Block C       │  │
│   │  Knowledge      │   │  Clinical       │   │  Critic          │  │
│   │  Graph          │   │  Classifier     │   │  Pipeline        │  │
│   │                 │   │                 │   │                  │  │
│   │  NetworkX       │   │  LLM-based      │   │  Orchestration   │  │
│   │  DiGraph        │   │  Entity Linker  │   │  + Constraint    │  │
│   │                 │   │                 │   │  Synthesis       │  │
│   └────────┬────────┘   └────────┬────────┘   └────────┬─────────┘  │
│            │                     │                      │            │
│            └─────────────────────┴──────────────────────┘            │
│                                  │                                   │
└──────────────────────────────────┼───────────────────────────────────┘
                                   ▼
                           SafetyResult
```

---

## Pipeline Flow

The SafetyCritic processes user input through 5 sequential stages:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    Stage 1   │    │    Stage 2   │    │    Stage 3   │    │    Stage 4   │    │    Stage 5   │
│  Segmenter   │───▶│   Filter     │───▶│  Classifier  │───▶│   Triage     │───▶│  Synthesis   │
│              │    │   META       │    │              │    │              │    │              │
│  LLM parses  │    │  segments    │    │  Map to      │    │  Select      │    │  Generate    │
│  multi-part  │    │  discarded   │    │  graph       │    │  primary     │    │  constraint  │
│  inputs      │    │              │    │  nodes       │    │  risk        │    │  message     │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Detailed Flow

```python
async def evaluate_intervention(user_input: str) -> SafetyResult:
    # Stage 1: Segment
    segments = await self.segment_input(user_input)
    # Example: "I hate this, but I fight" → [META, ACTION]
    
    # Stage 2: Filter
    dream_segments = [s for s in segments if s.segment_type != "META"]
    # If all META → return SAFE with "META_ONLY"
    
    # Stage 3: Classify
    for segment in dream_segments:
        node_id = await self.classifier.classify_intent(segment.text)
        # Look up node in graph → get type (adaptive/maladaptive/neutral)
    
    # Stage 4: Triage
    if maladaptive_segments:
        sorted_by_severity = self._sort_by_severity(maladaptive_segments)
        primary_risk = sorted_by_severity[0]  # Most critical
    else:
        return SAFE
    
    # Stage 5: Synthesize
    constraint = self._build_constraint_message(primary_risk, adaptive_segments)
    return SafetyResult(is_safe=False, constraint=constraint, ...)
```

---

## Block A: Knowledge Graph

**File**: `safety/graph_definitions.py`

### Structure

The knowledge graph is a **directed graph** (NetworkX DiGraph) with:
- **Nodes**: Clinical categories (10 total)
- **Edges**: Correction strategies (9 total)

### Node Types

| Type | Count | Purpose |
|------|-------|---------|
| **Maladaptive** | 3 | Behaviors that need redirection |
| **Adaptive** | 5 | Healthy coping goals |
| **Neutral** | 2 | Context without action |

### Maladaptive Nodes (Risks)

From IRT clinical guidelines (Krakow & Zadra, 2006), DSM-5, and Germain et al. (2004):

```
┌─────────────────────────────────┬──────────┬─────────────────────────────────┐
│ Node ID                         │ Severity │ Description                     │
├─────────────────────────────────┼──────────┼─────────────────────────────────┤
│ VIOLENT_REVENGE                 │ CRITICAL │ Extreme violence, killing,      │
│                                 │ (1)      │ destroying everything           │
├─────────────────────────────────┼──────────┼─────────────────────────────────┤
│ INTERRUPTION                    │ HIGH     │ Waking up, ending the dream,    │
│                                 │ (2)      │ escaping the narrative          │
├─────────────────────────────────┼──────────┼─────────────────────────────────┤
│ AVOIDANCE                       │ MODERATE │ Hiding, freezing, avoidance,    │
│                                 │ (3)      │ doing nothing                   │
└─────────────────────────────────┴──────────┴─────────────────────────────────┘
```

### Adaptive Nodes (Goals)

From MMS (Mastery Measurement Scale):

```
┌─────────────────────────────────┬─────────────────────────────────────────────┐
│ Node ID                         │ Description                                 │
├─────────────────────────────────┼─────────────────────────────────────────────┤
│ BEHAVIORAL_MASTERY              │ Active response: fighting, confronting      │
│ SOCIAL_MASTERY                  │ Seeking help: calling others, allies        │
│ ENVIRONMENTAL_MASTERY           │ Using environment: finding tools, exits     │
│ EMOTIONAL_MASTERY               │ Self-regulation: calming, staying present   │
│ MYTHICAL_MASTERY                │ Supernatural: magic powers, divine help     │
└─────────────────────────────────┴─────────────────────────────────────────────┘
```

### Neutral Nodes (Context)

```
┌─────────────────────────────────┬─────────────────────────────────────────────┐
│ Node ID                         │ Description                                 │
├─────────────────────────────────┼─────────────────────────────────────────────┤
│ NARRATIVE_SETTING                │ Scene description without action            │
│ AFFECT_EXPRESSION                │ Feeling statement without action            │
└─────────────────────────────────┴─────────────────────────────────────────────┘
```

### Correction Edges

Each maladaptive node has edges to multiple adaptive nodes, prioritized:

```
AVOIDANCE
    ├── [PRIMARY]   → BEHAVIORAL_MASTERY    "What action could you take?"
    ├── [SECONDARY] → SOCIAL_MASTERY        "Who could help you?"
    └── [TERTIARY]  → MYTHICAL_MASTERY      "What power would help?"

INTERRUPTION (Escape)
    ├── [PRIMARY]   → EMOTIONAL_MASTERY     "How could you calm yourself?"
    ├── [SECONDARY] → ENVIRONMENTAL_MASTERY "What would make you feel safe?"
    └── [TERTIARY]  → MYTHICAL_MASTERY      "What protective presence?"

VIOLENT_REVENGE (Violence)
    ├── [PRIMARY]   → BEHAVIORAL_MASTERY    "What focused action instead?"
    ├── [SECONDARY] → SOCIAL_MASTERY        "Who could help handle this?"
    └── [TERTIARY]  → EMOTIONAL_MASTERY     "What feeling drives this?"
```

### Graph Attributes

Each node contains rich metadata:

```python
{
    "type": "maladaptive",           # or "adaptive" or "neutral"
    "category": "avoidance",         # semantic category
    "severity": "high",              # for maladaptive nodes
    "description": "...",            # human-readable explanation
    "clinical_guideline": "...",     # why this is problematic
    "behavioral_markers": [...],     # example behaviors
    "reference": "...",              # academic citation
}
```

Each edge contains:

```python
{
    "relation": "correction_strategy",
    "priority": "primary",           # or "secondary", "tertiary"
    "rationale": "...",              # why this correction works
    "therapeutic_direction": "...",   # what to say to patient
    "clinical_reference": "...",      # citation
}
```

---

## Block B: Clinical Classifier

**File**: `safety/classifier.py`

### Purpose
Maps free-text user input to a single graph node ID using LLM-based classification.

### Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                      ClinicalClassifier                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐      │
│   │   System     │     │    Agent     │     │  Validator   │      │
│   │   Prompt     │────▶│   (LLM)      │────▶│  + Fuzzy     │      │
│   │              │     │              │     │  Matching    │      │
│   │  "Clinical   │     │  Groq 8B     │     │              │      │
│   │   Coder"     │     │  temp=0.1    │     │  Ensures     │      │
│   │              │     │              │     │  valid node  │      │
│   └──────────────┘     └──────────────┘     └──────────────┘      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### System Prompt (Condensed)

```
You are a Clinical Coder for IRT. Classify dream actions into:

MALADAPTIVE:
1. AVOIDANCE - avoidance, hiding, doing nothing
2. INTERRUPTION - waking up, ending dream
3. VIOLENT_REVENGE - extreme violence

ADAPTIVE:
4. BEHAVIORAL_MASTERY - active response, fighting
5. SOCIAL_MASTERY - calling for help
6. ENVIRONMENTAL_MASTERY - finding tools
7. EMOTIONAL_MASTERY - calming down
8. MYTHICAL_MASTERY - supernatural powers

NEUTRAL:
9. NARRATIVE_SETTING - scene description
10. AFFECT_EXPRESSION - feeling statement

NEGATION HANDLING:
- "I decided not to run" → BEHAVIORAL_MASTERY (the decision, not the avoided action)

Respond with ONLY the category code.
```

### Exact Code Validation

After superficial formatting normalization, classifier output must match one
of the defined codes exactly. Invalid or partial outputs fail closed:

```python
code = normalize_classification_code(raw_response)
node_id = CODE_MAP.get(code) if code is not None else "UNCLASSIFIED"
```

---

## Block C: SafetyCritic Pipeline

**File**: `safety/critic.py`

### Class: SafetyCritic

The main orchestrator that combines Blocks A and B.

### Initialization

```python
class SafetyCritic:
    def __init__(self):
        # Block A: Load knowledge graph
        self.graph = build_clinical_graph()
        
        # Block B: Initialize classifier
        self.classifier = ClinicalClassifier()
        
        # Segmenter: Separate LLM for parsing multi-part inputs
        self.segmenter = Agent(
            model_config=MODELS["MISTRAL_NEMO"],
            system_prompt=SEGMENTER_PROMPT,
            temperature=0
        )
```

### Stage 1: Segmentation

**Purpose**: Parse complex inputs into discrete segments with type labels.

**LLM Prompt** (SEGMENTER_PROMPT):
```
Segment Types:
- ACTION: doing something ("I fight")
- THOUGHT: decision/realization ("I decide to stay")  
- FEELING: emotion ("I feel scared")
- META: not dream content ("I hate this therapy")

Return JSON array: [{"text": "...", "type": "..."}, ...]
```

**Example**:
```
Input:  "I hate this therapy, but I guess I fight the dragon."
Output: [
    {"text": "I hate this therapy", "type": "META"},
    {"text": "but I guess I fight the dragon", "type": "ACTION"}
]
```

### Stage 2: META Filtering

```python
dream_segments = [s for s in segments if s.segment_type != "META"]

if not dream_segments:
    # All META → safe pass-through
    return SafetyResult(is_safe=True, node_id="META_ONLY", ...)
```

**Why filter META?**
- Resistance/complaints are normal in therapy
- Not actionable for safety evaluation
- Allows chatbot to address resistance naturally

### Stage 3: Classification

Each non-META segment is classified independently:

```python
for segment in dream_segments:
    node_id = await self.classifier.classify_intent(segment.text)
    node_attrs = self.graph.nodes[node_id]
    segment.node_id = node_id
    segment.node_type = node_attrs["type"]  # adaptive/maladaptive/neutral
```

**Result**: Segments grouped into:
- `maladaptive_segments`
- `adaptive_segments`
- `neutral_segments`

### Stage 4: Triage

When multiple maladaptive segments exist, select the most critical:

```python
SEVERITY_PRIORITY = {
    "VIOLENT_REVENGE": 1,               # CRITICAL
    "INTERRUPTION": 2,                  # HIGH
    "AVOIDANCE": 3,                     # MODERATE
}

def _sort_by_severity(segments):
    return sorted(segments, key=lambda s: SEVERITY_PRIORITY.get(s.node_id, 99))

primary_risk = sorted_maladaptive[0]
```

**Example**:
```
Input: "I try calling for help but when no one comes I just wake myself up"

Segments:
- "I try calling for help" → SOCIAL_MASTERY (adaptive)
- "I just wake myself up" → INTERRUPTION (maladaptive, HIGH)

Result: UNSAFE, primary_risk = INTERRUPTION
```

### Stage 5: Constraint Synthesis

Generate the message to inject into the chatbot's prompt:

```python
def _build_constraint_message(primary_risk, adaptive_segments):
    # VALIDATE section (if any adaptive parts)
    if adaptive_segments:
        validate = f"VALIDATE: User showed positive elements: {[s.text for s in adaptive_segments]}"
    
    # CORRECT section
    correct = f"""
    CORRECT: User demonstrated "{primary_risk.node_id}" - {severity} risk.
    Risk Description: {node_attrs['description']}
    Clinical Guideline: {node_attrs['clinical_guideline']}
    """
    
    # INSTRUCTION section
    strategies = get_correction_strategies(graph, primary_risk.node_id)
    instruction = f"INSTRUCTION: Gently redirect using: {strategies[0]['therapeutic_direction']}"
    
    return f"[SAFETY SYSTEM]\n{validate}\n{correct}\n{instruction}"
```

---

## Data Structures

### SegmentResult

```python
@dataclass
class SegmentResult:
    text: str                    # The segment text
    segment_type: str            # ACTION, THOUGHT, FEELING, META
    node_id: Optional[str]       # Graph node (after classification)
    node_type: Optional[str]     # adaptive, maladaptive, neutral
    category: Optional[str]      # Semantic category from graph
```

### SafetyResult

```python
@dataclass
class SafetyResult:
    is_safe: bool                           # Final verdict
    node_id: str                            # Primary node (most critical if unsafe)
    node_type: str                          # adaptive, maladaptive, neutral, meta
    category: str                           # Semantic category
    constraint: Optional[str]               # Message for chatbot
    correction_strategies: Optional[List]   # From graph edges
    input_text: str                         # Original user input
    clinical_guideline: str                 # Why problematic (if unsafe)
    segments: List[SegmentResult]           # All parsed segments
    adaptive_segments: List[SegmentResult]  # Positive elements to validate
    severity: Optional[str]                 # CRITICAL, HIGH, MODERATE
```

---

## Constraint Generation

### Safe Input Constraint

```
VALIDATE: User demonstrated adaptive responses: 
'I turn around' (BEHAVIORAL_MASTERY), 'I felt scared' (AFFECT_EXPRESSION). 
Proceed with encouragement.
```

### Unsafe Input Constraint

```
[SAFETY SYSTEM - INTERVENTION REQUIRED]

VALIDATE: The user showed positive elements: 'I tried calling for help'. 
Acknowledge these first (e.g., "Good job noticing..." or "I appreciate that you...").

CORRECT: User demonstrated "INTERRUPTION" - this is a HIGH risk.
Risk Description: Breaking the dream narrative completely to stop the affect...
Clinical Guideline: Interrupting the dream prevents emotional processing...

INSTRUCTION: Gently redirect using this approach:
How could you calm yourself to stay in the dream? What would help you feel safe enough to continue?

Alternative therapeutic directions:
- EMOTIONAL_MASTERY: How could you calm yourself to stay in the dream?
- ENVIRONMENTAL_MASTERY: What in the environment could make you feel safe?

IMPORTANT: 
- Do NOT validate or encourage the maladaptive response ("I just wake myself up")
- Focus correction on THIS specific issue only - do not list other potential problems
- Use empathetic, supportive language while redirecting
- Do not shame or criticize the user's suggestion
```

---

## Integration Points

### Where the Critic Fits in IRT Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  RECORDING  │────▶│  REWRITING  │────▶│   SUMMARY   │
│   Stage     │     │   Stage     │     │   Stage     │
└─────────────┘     └──────┬──────┘     └─────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ SafetyCritic │
                   │  evaluates   │
                   │  user input  │
                   └──────┬───────┘
                          │
           ┌──────────────┴──────────────┐
           ▼                              ▼
     ┌──────────┐                  ┌───────────┐
     │   SAFE   │                  │  UNSAFE   │
     │          │                  │           │
     │ Continue │                  │ Inject    │
     │ normally │                  │ constraint│
     └──────────┘                  └───────────┘
```

### Integration Code (irt_app.py)

```python
async def get_response_async(stage, user_input, conversation):
    # Only check safety in rewriting stage
    if stage == "rewriting":
        critic = SafetyCritic()
        result = await critic.evaluate_intervention(user_input)
        
        if not result.is_safe:
            # Inject constraint into system prompt
            full_prompt += f"\n\n{result.constraint}"
    
    # Continue with normal response generation
    response = await response_agent.generate(full_prompt)
```

---

## LLM Usage Summary

The SafetyCritic makes **2 LLM calls** per evaluation:

| Call | Model | Purpose | Temperature |
|------|-------|---------|-------------|
| 1. Segmenter | Mistral NeMo Instruct 2407 (Scaleway) | Parse multi-part input | 0 |
| 2. Classifier | Mistral Small 3.2 24B Instruct 2506 (Scaleway) | Map segment to node | 0 (0.7 in classification-consistency mode) |

**Note**: If input has multiple non-META segments, classification is called once per segment.

---

## Error Handling

### Pass-Through Philosophy
If the critic fails for any reason, the system passes through (allows the input):

```python
try:
    result = await critic.evaluate_intervention(user_input)
except Exception as e:
    logger.error(f"Critic error: {e}")
    # Continue without safety check - don't block the user
```

### Segmentation Fallback
If JSON parsing fails, treat entire input as single ACTION:

```python
except json.JSONDecodeError:
    return [SegmentResult(text=user_input, segment_type="ACTION")]
```

### Classification Fallback
If the LLM returns an invalid code, route it to clarification without semantic
keyword recovery:

```python
if code not in CODE_MAP:
    node_id = "UNCLASSIFIED"
```

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| LLM calls per evaluation | 2+ (1 segment + N classify) |
| Latency | ~500ms-1500ms depending on input complexity |
| Graph operations | O(1) lookups, O(E) edge traversal |
| Memory | ~10KB graph, minimal state |

---

## Files Reference

```
safety/
├── __init__.py              # Module exports
├── graph_definitions.py     # Block A: Knowledge Graph
├── classifier.py            # Block B: Clinical Classifier (System 1, direct code)
├── classifier_cot.py        # Block B variant: structured-rationale classifier (System 2)
├── critic.py                # Block C: SafetyCritic Pipeline
├── semantic_consistency.py  # Classification-consistency procedure (uncertainty)
└── stage_merge_critic.py    # Ablation A: segmentation and classification in one call

docs/
├── architecture/Critic_Architecture.md              # This file
├── ablation_spec.md                                 # Ablation conditions
└── clinical/Clinical_Ontology_Scientific_Rationale.md  # Academic basis
```
