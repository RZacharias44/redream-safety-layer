"""
Ablation A — StageMergeCritic unit tests.

Spec: docs/ablation_spec.md §"Ablation A" + Addenda 2, 3 & 4.

Schema under test (Addendum 3): per-segment {text, context, code} — no
`type` field (derived deterministically from code post-hoc, mirroring
the structured classifier's CODE_MAP lookup).

Covers: valid parse + post-hoc derivation (§A.5), META filtering,
severity-priority primary selection, all-meta -> META_ONLY safe,
single-retry-then-success, fail-closed PARSE_FAILURE (§A.4), schema
validation (text/code required, invalid code, optional context,
markdown-fenced JSON recovery), and prompt-parity guards (§A.3 +
Addendum 3/4: clinical body verbatim, segmenter guidance + examples,
UNCLASSIFIED parity, no severity/correction/CoT leakage).

Hermetic: safety.stage_merge_critic.Agent patched (no API key/network);
critic.agent.generate stubbed with canned LLM responses.
"""

import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from safety.stage_merge_critic import StageMergeCritic, STAGE_MERGE_PROMPT


def _critic(responses):
    """StageMergeCritic whose LLM returns the given list of strings in order."""
    with patch("safety.stage_merge_critic.Agent", MagicMock()):
        c = StageMergeCritic()
    c.agent.generate = AsyncMock(side_effect=[(r, {}, None) for r in responses])
    return c


def _run(c, text, ctx="ctx"):
    return asyncio.run(c.evaluate_intervention(text, nightmare_context=ctx))


# ---- valid parse + post-hoc derivation (§A.5) ------------------------

def test_valid_mixed_segments_derives_unsafe_and_filters_meta():
    payload = json.dumps({"segments": [
        {"text": "this exercise is dumb", "context": None, "code": "META"},
        {"text": "I hide under the bed", "context": "being chased", "code": "HIDE"},
        {"text": "then I call for help", "context": "facing the threat", "code": "HELP"},
    ]})
    r = _run(_critic([payload]), "x")
    assert r.is_safe is False
    assert r.node_id == "AVOIDANCE"             # HIDE -> AVOIDANCE
    assert r.node_type == "maladaptive"         # derived deterministically
    assert r.parse_failure is False
    # META labelled, not dropped from the segment list
    assert any(s.segment_type == "META" for s in r.segments)
    assert len(r.segments) == 3
    # context preserved
    assert r.segments[1].context == "being chased"


def test_severity_priority_picks_most_critical_maladaptive():
    payload = json.dumps({"segments": [
        {"text": "I hide", "context": "fear", "code": "HIDE"},          # AVOIDANCE prio 4
        {"text": "I butcher him", "context": "rage", "code": "DESTROY"}, # VIOLENT_REVENGE prio 1
    ]})
    r = _run(_critic([payload]), "x")
    assert r.is_safe is False
    assert r.node_id == "VIOLENT_REVENGE"       # highest severity wins
    assert r.severity == "CRITICAL"


def test_all_meta_is_safe_meta_only():
    payload = json.dumps({"segments": [
        {"text": "I hate this therapy", "context": None, "code": "META"},
    ]})
    r = _run(_critic([payload]), "x")
    assert r.is_safe is True
    assert r.node_id == "META_ONLY"


def test_all_adaptive_is_safe():
    payload = json.dumps({"segments": [
        {"text": "I fight back", "context": "facing the threat", "code": "CONFRONT"},
    ]})
    r = _run(_critic([payload]), "x")
    assert r.is_safe is True
    assert r.node_id == "BEHAVIORAL_MASTERY"


# ---- retry / fail-closed (§A.4) --------------------------------------

def test_single_retry_then_success():
    bad = "sorry, here is the analysis: the user hides."
    good = json.dumps({"segments": [
        {"text": "I hide", "context": "fear", "code": "HIDE"}
    ]})
    c = _critic([bad, good])
    r = _run(c, "x")
    assert c.agent.generate.await_count == 2
    assert r.parse_failure is False and r.node_id == "AVOIDANCE"


def test_double_failure_is_parse_failure():
    c = _critic(["not json", "still not json"])
    r = _run(c, "x")
    assert c.agent.generate.await_count == 2
    assert r.parse_failure is True
    assert r.node_id == "PARSE_FAILURE" and r.is_safe is False
    assert r.segments == []


def test_unknown_code_routes_to_unclassified_without_retry():
    bad = json.dumps({"segments": [{"text": "I hide", "context": None, "code": "BOGUS"}]})
    c = _critic([bad])
    r = _run(c, "x")
    assert c.agent.generate.await_count == 1
    assert r.parse_failure is False
    assert r.is_safe is False
    assert r.node_id == "UNCLASSIFIED"
    assert r.severity == "UNCERTAIN"


def test_empty_segments_array_is_invalid():
    c = _critic([json.dumps({"segments": []}), json.dumps({"segments": []})])
    assert _run(c, "x").parse_failure is True


def test_markdown_fenced_json_is_recovered():
    payload = "```json\n" + json.dumps(
        {"segments": [{"text": "I wake myself up", "context": "panic", "code": "ESCAPE"}]}
    ) + "\n```"
    r = _run(_critic([payload]), "x")
    assert r.parse_failure is False and r.node_id == "INTERRUPTION"


# ---- UNCLASSIFIED parity (Addendum 4) --------------------------------

def test_unclassified_segment_fails_closed_to_unsafe():
    """Mirrors safety/critic.py:520-531: no maladaptive but UNCLASSIFIED
    present -> is_safe=False, primary_node='UNCLASSIFIED'."""
    payload = json.dumps({"segments": [
        {"text": "I do something weird", "context": None, "code": "UNCLASSIFIED"},
        {"text": "then I fight back", "context": "facing the threat", "code": "CONFRONT"},
    ]})
    r = _run(_critic([payload]), "x")
    assert r.is_safe is False                    # fail-closed
    assert r.node_id == "UNCLASSIFIED"
    assert r.severity == "UNCERTAIN"
    # The CONFRONT segment is still tracked as adaptive (for clarification messages)
    assert any(s.node_id == "BEHAVIORAL_MASTERY" for s in r.adaptive_segments)


def test_unclassified_does_not_override_maladaptive():
    """Maladaptive present -> still the primary, even with UNCLASSIFIED in mix."""
    payload = json.dumps({"segments": [
        {"text": "ambiguous bit", "context": None, "code": "UNCLASSIFIED"},
        {"text": "I butcher him", "context": "rage", "code": "DESTROY"},
    ]})
    r = _run(_critic([payload]), "x")
    assert r.is_safe is False
    assert r.node_id == "VIOLENT_REVENGE"        # maladaptive wins
    assert r.severity == "CRITICAL"


def test_unclassified_does_not_count_as_maladaptive():
    """All-UNCLASSIFIED (no maladaptive) -> unsafe via fail-closed, NOT via
    treating UNCLASSIFIED as maladaptive."""
    payload = json.dumps({"segments": [
        {"text": "ambiguous", "context": None, "code": "UNCLASSIFIED"},
    ]})
    r = _run(_critic([payload]), "x")
    assert r.is_safe is False
    assert r.node_id == "UNCLASSIFIED"
    assert r.node_type == "unclassified"          # not "maladaptive"


# ---- context field handling (Addendum 3) ------------------------------

def test_context_optional_when_missing_defaults_to_none():
    payload = json.dumps({"segments": [
        {"text": "I fight back", "code": "CONFRONT"}  # no context key
    ]})
    r = _run(_critic([payload]), "x")
    assert r.parse_failure is False
    assert r.segments[0].context is None


def test_context_must_be_string_or_null():
    bad = json.dumps({"segments": [
        {"text": "I fight back", "context": 42, "code": "CONFRONT"}  # int — invalid
    ]})
    c = _critic([bad, bad])
    assert _run(c, "x").parse_failure is True


# ---- prompt parity guards (§A.3 / §"Don'ts" + Addendum 3) ------------

def test_prompt_has_clinical_body_segmentation_and_schema():
    # CLINICAL_CODER_PROMPT body verbatim
    assert "Clinical Coder for Imagery Rehearsal Therapy" in STAGE_MERGE_PROMPT
    assert "Violence Threshold" in STAGE_MERGE_PROMPT
    # Addendum 3: segmentation + context guidance from SEGMENTER_PROMPT
    assert "## SEGMENTATION" in STAGE_MERGE_PROMPT
    assert "## CONTEXT EXTRACTION" in STAGE_MERGE_PROMPT
    # Schema (Addendum 3): text + context + code, no `type` field
    assert '"context"' in STAGE_MERGE_PROMPT
    assert '"code"' in STAGE_MERGE_PROMPT
    assert '"type"' not in STAGE_MERGE_PROMPT  # dropped (Addendum 3)
    # Addendum 4: UNCLASSIFIED is a code-level fallback, not a prompted option.
    assert "UNCLASSIFIED" not in STAGE_MERGE_PROMPT


def test_prompt_excludes_severity_correction_and_cot():
    low = STAGE_MERGE_PROMPT.lower()
    assert "severity_priority" not in low
    assert "correction" not in low and "redirect" not in low
    assert "step by step" not in low and "think through" not in low  # no CoT


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
