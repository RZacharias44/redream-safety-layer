"""
Ablation A — Stage-Merge critic (docs/ablation_spec.md §"Ablation A").

A single LLM call performs BOTH segmentation and classification on the
full input (the "stage merge"). Triage (severity priority) and the
is_safe / primary_node verdict are derived post-hoc in code, identically
to the structured pipeline — the LLM does NOT do constraint synthesis or
correction generation (those are not scored on the dev benchmark).

Parity (spec §A.3 + Addenda 2, 3 & 4): the system prompt is the
CLINICAL_CODER_PROMPT clinical body VERBATIM (intro + 12 definitions +
7 rules + examples), then a segmentation + context-extraction section
ported from SEGMENTER_PROMPT (verbatim guidance + 5 verbatim Inputs from
its examples, adapted to the merged schema). Both knowledge sources are
the merged LLM's structurally-equivalent counterparts.

Schema (Addendum 3): {"segments": [{"text", "context", "code"}, ...]} —
three fields, each matching something a structured component already
emits (text + context from segmenter, code from classifier). `type` is
DROPPED (the structured classifier never emits it; node_type/category
are derived deterministically from code via the SAME CODE_MAP the
structured pipeline uses). No CoT, no severity priorities, no correction
edges in the prompt (§A.3 / §"Don'ts").

Fail-closed JSON policy (spec §A.4, adapted): two attempts at
temperature=0; validation = JSON object with non-empty segments array,
each segment has text (non-empty str) + code (str) + context (str|null,
optional defaulting to null). Unknown per-segment codes route to
UNCLASSIFIED, matching the structured classifier's post-prompt fallback.
On double JSON/schema failure record is_safe=False, node_id="PARSE_FAILURE",
parse_failure=True, segments=[].
"""

import json
import logging
from typing import Optional, List, Tuple

from .classifier import CLINICAL_CODER_PROMPT, CODE_MAP
from .critic import SegmentResult, SafetyResult, SEVERITY_PRIORITY
from .graph_definitions import build_clinical_graph
from AI.agent import Agent, MODELS

logger = logging.getLogger(__name__)

# Spec §A.3: reuse the CLINICAL_CODER_PROMPT *body* verbatim; replace only
# the response-format section. Splitting on the heading keeps the intro +
# CODES + CRITICAL RULES + EXAMPLES byte-identical (no fragile renaming).
_CLINICAL_BODY = CLINICAL_CODER_PROMPT.split("## RESPONSE FORMAT")[0].rstrip()

# Addendum 3 (2026-05-21): segmentation + context-extraction guidance,
# verbatim adaptations from SEGMENTER_PROMPT (safety/critic.py:49-95).
# Examples 1, 2, 4, 5, 6 — verbatim Inputs from the segmenter examples,
# adapted to the merged schema {text, context, code}. Example 3 dropped
# (its classification depends on CONTEXT_AWARE_EXAMPLES, not in the base
# parity prompt). Codes are directly derivable from CLINICAL_CODER_PROMPT
# above: "I feel scared" → FEEL (line 144), "punch / fight" → CONFRONT
# (line 138), "This is dumb" / "I hate" → META (rules for meta content).
_STAGE_MERGE_RESPONSE_FORMAT = """

## SEGMENTATION

Split the user's Input into consecutive segments. Split based on
**meaning**, not just punctuation — one segment per discrete action,
feeling, or remark. If the entire Input is a single action/feeling,
return a single-element segments array. Preserve the original text
verbatim in each segment.

Conversational / meta content (e.g. "this is dumb", "I hate this
therapy", questions about the therapy itself — not part of the dream
narrative) is labelled, never dropped, with `code: "META"`.

## CONTEXT EXTRACTION (per segment)

For each segment, identify the **Immediate Threat** or **Situation** the
user is responding to.
- Use the provided Nightmare Context (if any) and the User Input.
- Example: if a segment is "I punch it", context might be "facing the monster".
- Example: if a segment is "I hide", context might be "being chased".
- If no clear context applies (e.g. for META or pure feeling expressions),
  use null.

## CLASSIFICATION (per segment)

Assign `code` using the CODES and CRITICAL RULES above. One of:
HIDE, ESCAPE, DESTROY, DENY, REPLAY, CONFRONT, HELP, CHANGE, RELAX,
POWER, LOOK, FEEL  — or META for conversational/meta content.

## EXAMPLES (segment + classify with context)

Context: "Being chased by a dragon in a cave."
Input: "I hate this therapy, but I guess I turn and fight it."
Output: {"segments": [
  {"text": "I hate this therapy", "context": null, "code": "META"},
  {"text": "but I guess I turn and fight it", "context": "being chased by dragon", "code": "CONFRONT"}
]}

Input: "I feel scared but I stand my ground"
Output: {"segments": [
  {"text": "I feel scared", "context": null, "code": "FEEL"},
  {"text": "but I stand my ground", "context": "facing the threat", "code": "CONFRONT"}
]}

Input: "This is dumb"
Output: {"segments": [
  {"text": "This is dumb", "context": null, "code": "META"}
]}

Input: "I'm terrified"
Output: {"segments": [
  {"text": "I'm terrified", "context": null, "code": "FEEL"}
]}

Input: "Why do I have to do this? Fine, I'll punch the monster I guess"
Output: {"segments": [
  {"text": "Why do I have to do this?", "context": null, "code": "META"},
  {"text": "Fine, I'll punch the monster I guess", "context": "confronting the monster", "code": "CONFRONT"}
]}

## RESPONSE FORMAT

Respond with ONLY a JSON object, no markdown, no explanation:

{"segments": [{"text": "...", "context": "..."|null, "code": "..."}, ...]}

The segments array must be non-empty (every input has at least one
segment, even if it is pure meta). No other keys. Just the JSON.
"""

STAGE_MERGE_PROMPT = _CLINICAL_BODY + _STAGE_MERGE_RESPONSE_FORMAT

_RETRY_REMINDER = (
    "\n\nReturn ONLY a valid JSON object matching the specified schema. "
    "No markdown, no explanation."
)


class StageMergeCritic:
    """Drop-in for SafetyCritic.evaluate_intervention (Ablation A)."""

    def __init__(self):
        self.graph = build_clinical_graph()
        # Same backend/decoding as the production S1 classifier
        # (MISTRAL_SMALL via Scaleway-direct, temperature=0) — spec
        # §"Configuration": backend identical across all conditions.
        self.agent = Agent(
            model_config=MODELS["MISTRAL_SMALL"],
            system_prompt=STAGE_MERGE_PROMPT,
            temperature=0,
        )

    # ---- parsing / validation (spec §A.4) ----------------------------

    def _extract_json(self, raw: str) -> str:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]
        return cleaned

    def _parse_and_validate(self, raw: str) -> Optional[List[SegmentResult]]:
        """Return validated SegmentResult list, or None if invalid.

        Schema (Addendum 3): each segment has `text` (non-empty str),
        `code` (str), and optionally `context` (str | null; defaults to
        None if missing). `node_type` and `category` are derived
        deterministically from `code` via CODE_MAP + graph lookup — the
        SAME path the structured classifier uses post-hoc. Unknown
        per-segment codes route to UNCLASSIFIED as a deterministic
        backstop (Addendum 4). No `type` field check (Addendum 3).
        """
        try:
            obj = json.loads(self._extract_json(raw))
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(obj, dict):
            return None
        seg_data = obj.get("segments")
        if not isinstance(seg_data, list) or not seg_data:
            return None  # missing / not array / empty -> fail (spec §A.2)

        out: List[SegmentResult] = []
        for s in seg_data:
            if not isinstance(s, dict):
                return None
            text, code = s.get("text"), s.get("code")
            if not isinstance(text, str) or not text.strip():
                return None
            if not isinstance(code, str):
                return None
            code = code.strip().upper()
            # context: optional; if present must be str or null (any
            # other type -> invalid). Empty string normalised to None.
            context = s.get("context", None)
            if context is not None and not isinstance(context, str):
                return None
            if context == "":
                context = None

            if code == "META":
                out.append(SegmentResult(text=text, segment_type="META",
                                         context=context, node_id="META",
                                         node_type="meta", category="meta"))
                continue
            if code == "UNCLASSIFIED":
                # Addendum 4 (2026-05-21): parity with structured pipeline's
                # per-segment UNCLASSIFIED routing (safety/critic.py:480-490).
                out.append(SegmentResult(
                    text=text, segment_type="ACTION", context=context,
                    node_id="UNCLASSIFIED", node_type="unclassified",
                    category="unknown",
                ))
                continue
            # Clinical content: code maps post-hoc, mirroring the structured
            # classifier's CODE_MAP lookup. Unknown per-segment codes are not
            # whole-response parse failures; they route to UNCLASSIFIED, the
            # structured classifier's deterministic backstop after parse/fuzzy
            # matching failure.
            node_id = CODE_MAP.get(code)
            if node_id is None or node_id not in self.graph:
                out.append(SegmentResult(
                    text=text, segment_type="ACTION", context=context,
                    node_id="UNCLASSIFIED", node_type="unclassified",
                    category="unknown",
                ))
                continue
            out.append(SegmentResult(
                text=text, segment_type="ACTION", context=context,
                node_id=node_id,
                node_type=self.graph.nodes[node_id].get("type"),
                category=self.graph.nodes[node_id].get("category", "unknown"),
            ))
        return out

    # ---- post-hoc derivation (spec §A.5) -----------------------------

    def _derive(self, segments: List[SegmentResult], user_input: str) -> SafetyResult:
        # Step 1: mirror the structured META filter (critic.py:296)
        dream = [s for s in segments if s.node_type != "meta"]
        # Step 2: three-bucket triage IDENTICAL to structured pipeline
        # (safety/critic.py:494-531; Addendum 4):
        #   maladaptive > UNCLASSIFIED (fail-closed) > adaptive/neutral.
        maladaptive = [s for s in dream if s.node_type == "maladaptive"]
        unclassified = [s for s in dream if s.node_type == "unclassified"]
        adaptive_segments = [s for s in dream
                             if s.node_type in ("adaptive", "neutral")]

        if maladaptive:
            primary = min(maladaptive,
                          key=lambda s: SEVERITY_PRIORITY.get(s.node_id, 99))
            node_id, node_type = primary.node_id, "maladaptive"
            category = primary.category
            is_safe = False
        elif unclassified:
            # Fail-closed: no clear risk but uncertain segments present.
            # Mirrors safety/critic.py:520-531 exactly (is_safe=False,
            # primary="UNCLASSIFIED" — clarification-seeking verdict).
            node_id, node_type, category = "UNCLASSIFIED", "unclassified", "unknown"
            is_safe = False
        elif dream:
            p = dream[0]
            node_id, node_type, category = p.node_id, p.node_type, p.category
            is_safe = True
        else:
            node_id, node_type, category = "META_ONLY", "neutral", "meta"
            is_safe = True

        severity = None
        if not is_safe and node_id != "UNCLASSIFIED":
            sev_rank = SEVERITY_PRIORITY.get(node_id)
            severity = {1: "CRITICAL", 2: "HIGH", 3: "HIGH",
                        4: "HIGH"}.get(sev_rank, "HIGH")
        elif node_id == "UNCLASSIFIED":
            severity = "UNCERTAIN"

        # NB SafetyResult has no `unclassified_segments` field — the
        # structured pipeline tracks unclassified locally and lets them
        # live in `segments[]` (each carrying node_id="UNCLASSIFIED"). We
        # do the same: unclassified info is preserved per-segment.
        return SafetyResult(
            is_safe=is_safe, node_id=node_id, node_type=node_type,
            category=category, input_text=user_input,
            segments=segments,
            adaptive_segments=adaptive_segments,
            severity=severity,
        )

    # ---- public interface (matches SafetyCritic) ---------------------

    async def evaluate_intervention(self, user_input: str,
                                    nightmare_context: Optional[str] = None
                                    ) -> SafetyResult:
        if nightmare_context:
            llm_input = f'Context: "{nightmare_context}"\nInput: "{user_input}"'
        else:
            llm_input = f'Input: "{user_input}"'

        # Attempt 1
        try:
            resp, _, _ = await self.agent.generate(llm_input)
            segs = self._parse_and_validate(resp)
        except Exception as e:
            logger.warning(f"Stage-merge attempt 1 errored: {e}")
            segs = None

        # Attempt 2 (single retry with reminder, spec §A.4)
        if segs is None:
            try:
                resp, _, _ = await self.agent.generate(llm_input + _RETRY_REMINDER)
                segs = self._parse_and_validate(resp)
            except Exception as e:
                logger.warning(f"Stage-merge attempt 2 errored: {e}")
                segs = None

        if segs is None:
            logger.error("Stage-merge double parse failure -> PARSE_FAILURE")
            return SafetyResult(
                is_safe=False, node_id="PARSE_FAILURE",
                node_type="unclassified", category="unknown",
                input_text=user_input, segments=[],
                severity="CRITICAL", parse_failure=True,
            )

        return self._derive(segs, user_input)
