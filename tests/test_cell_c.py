"""
Cell C (naive sentence-splitter + no META) — unit tests.

Spec: docs/ablation_spec.md §"Cell C".
Contracts under test:
  1. naive_segment splits on sentence boundaries only ([.!?]+ws), labels
     every part ACTION, preserves context, and does NOT clause-split on
     conjunctions (but/and/so).
  2. SafetyCritic(naive_segmentation=True) routes Step 1 through
     naive_segment, never the LLM segmenter.
  3. naive_segmentation and skip_segmentation are mutually exclusive.

Hermetic: safety.critic.Agent patched (no API key / network).
"""

import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from safety.critic import SafetyCritic, naive_segment


def _fake_classifier():
    clf = MagicMock()
    clf.classify_intent = AsyncMock(return_value=("BEHAVIORAL_MASTERY", None))
    return clf


# --- 1. naive_segment behaviour ---------------------------------------

def test_naive_splits_on_sentence_boundaries():
    segs = naive_segment("I turn around. I fight him! Do I win?", "ctx")
    assert [s.text for s in segs] == ["I turn around.", "I fight him!", "Do I win?"]
    assert all(s.segment_type == "ACTION" for s in segs)
    assert all(s.context == "ctx" for s in segs)


def test_naive_does_not_clause_split_on_conjunctions():
    # spec §"Don'ts": sentence-level ONLY — one sentence stays one segment
    segs = naive_segment("I hide for a moment but then I turn and fight", None)
    assert len(segs) == 1
    assert segs[0].text == "I hide for a moment but then I turn and fight"


def test_naive_handles_empty_and_no_terminator():
    assert len(naive_segment("just one clause no period", None)) == 1
    only = naive_segment("   ", "c")
    assert len(only) == 1  # never returns empty


# --- 2. flag routing ---------------------------------------------------

def test_naive_segmentation_defaults_false():
    with patch("safety.critic.Agent", MagicMock()):
        c = SafetyCritic(classifier=_fake_classifier())
    assert c.naive_segmentation is False


def test_naive_segmentation_bypasses_llm_segmenter():
    with patch("safety.critic.Agent", MagicMock()):
        critic = SafetyCritic(classifier=_fake_classifier(), naive_segmentation=True)
    critic.segment_input = AsyncMock(
        side_effect=AssertionError("LLM segment_input called despite naive_segmentation")
    )

    result = asyncio.run(
        critic.evaluate_intervention("I freeze up. Then I run away.", nightmare_context="ctx")
    )

    critic.segment_input.assert_not_called()
    assert [s.text for s in result.segments] == ["I freeze up.", "Then I run away."]
    assert all(s.segment_type == "ACTION" for s in result.segments)


# --- 3. mutual exclusivity --------------------------------------------

def test_naive_and_skip_segmentation_mutually_exclusive():
    with patch("safety.critic.Agent", MagicMock()):
        with pytest.raises(ValueError, match="mutually exclusive"):
            SafetyCritic(classifier=_fake_classifier(),
                         naive_segmentation=True, skip_segmentation=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
