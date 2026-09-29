"""
Cell B (LLM segmenter + META filter OFF) — unit tests.

Spec: docs/ablation_spec.md §"Cell B".
Contract under test: when SafetyCritic is constructed with
skip_meta_filter=True, META-typed segments are NOT dropped at Step 2 —
they are classified/bucketed alongside dream content. The default
(skip_meta_filter=False) still filters META out.

Hermetic: safety.critic.Agent is patched (no API key / network); a fake
classifier records which segment texts it is asked to classify.
"""

import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from safety.critic import SafetyCritic, SegmentResult


def _recording_classifier():
    """Classifier stub: records classified texts; everything benign-adaptive."""
    clf = MagicMock()
    clf.classify_intent = AsyncMock(return_value=("BEHAVIORAL_MASTERY", None))
    return clf


def _mixed_segments():
    # one META segment + one real dream ACTION segment
    return [
        SegmentResult(text="this therapy is dumb", segment_type="META", context=None),
        SegmentResult(text="I fight him off", segment_type="ACTION", context="ctx"),
    ]


def test_skip_meta_filter_defaults_false():
    with patch("safety.critic.Agent", MagicMock()):
        critic = SafetyCritic(classifier=_recording_classifier())
    assert critic.skip_meta_filter is False


def test_meta_filtered_by_default():
    """Default: META segment dropped -> only the ACTION segment is classified."""
    with patch("safety.critic.Agent", MagicMock()):
        critic = SafetyCritic(classifier=_recording_classifier())
    critic.segment_input = AsyncMock(return_value=_mixed_segments())

    asyncio.run(critic.evaluate_intervention("x", nightmare_context="ctx"))

    classified = [c.args[0] for c in critic.classifier.classify_intent.call_args_list]
    assert classified == ["I fight him off"]          # META excluded
    assert "this therapy is dumb" not in classified


def test_meta_kept_when_filter_off():
    """skip_meta_filter=True: BOTH segments (incl. META) are classified."""
    with patch("safety.critic.Agent", MagicMock()):
        critic = SafetyCritic(classifier=_recording_classifier(), skip_meta_filter=True)
    critic.segment_input = AsyncMock(return_value=_mixed_segments())

    asyncio.run(critic.evaluate_intervention("x", nightmare_context="ctx"))

    classified = [c.args[0] for c in critic.classifier.classify_intent.call_args_list]
    assert "this therapy is dumb" in classified        # META NOT dropped
    assert "I fight him off" in classified
    assert len(classified) == 2


def test_flags_are_independent():
    with patch("safety.critic.Agent", MagicMock()):
        c = SafetyCritic(classifier=_recording_classifier(),
                         skip_segmentation=True, skip_meta_filter=True)
    assert c.skip_segmentation is True and c.skip_meta_filter is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
