"""
Ablation B (no-segmentation) — unit tests.

Spec: docs/ablation_spec.md §B.1.
Contract under test: when SafetyCritic is constructed with
skip_segmentation=True, the LLM segmenter is bypassed and the whole
user input is forced through as a single ACTION segment, carrying the
nightmare context. The default (skip_segmentation=False) still calls
the segmenter.

Hermetic: safety.critic.Agent is patched so no API key / network is
needed, and a fake classifier is injected.
"""

import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from safety.critic import SafetyCritic


def _fake_classifier():
    """Classifier stub: every segment classifies as a benign adaptive node."""
    clf = MagicMock()
    # evaluate_intervention does: node_id, logprobs = await classify_intent(...)
    clf.classify_intent = AsyncMock(return_value=("BEHAVIORAL_MASTERY", None))
    return clf


def test_skip_segmentation_defaults_false():
    with patch("safety.critic.Agent", MagicMock()):
        critic = SafetyCritic(classifier=_fake_classifier())
    assert critic.skip_segmentation is False


def test_skip_segmentation_bypasses_segmenter():
    """skip_segmentation=True -> segment_input never called; one ACTION segment."""
    user_input = "I hide under the bed. Then I turn and fight him off."
    ctx = "He is chasing me through the house."

    with patch("safety.critic.Agent", MagicMock()):
        critic = SafetyCritic(classifier=_fake_classifier(), skip_segmentation=True)

    # If the bypass works, segment_input must never be awaited.
    critic.segment_input = AsyncMock(
        side_effect=AssertionError("segment_input called despite skip_segmentation=True")
    )

    result = asyncio.run(
        critic.evaluate_intervention(user_input, nightmare_context=ctx)
    )

    critic.segment_input.assert_not_called()
    assert len(result.segments) == 1
    seg = result.segments[0]
    assert seg.text == user_input          # whole input, not split
    assert seg.segment_type == "ACTION"
    assert seg.context == ctx              # nightmare context preserved


def test_default_path_uses_segmenter():
    """Default (skip_segmentation=False) still routes through segment_input."""
    from safety.critic import SegmentResult

    with patch("safety.critic.Agent", MagicMock()):
        critic = SafetyCritic(classifier=_fake_classifier())

    seg_in = AsyncMock(
        return_value=[SegmentResult(text="I fight back", segment_type="ACTION", context="ctx")]
    )
    critic.segment_input = seg_in

    asyncio.run(critic.evaluate_intervention("I fight back", nightmare_context="ctx"))

    seg_in.assert_awaited_once()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
