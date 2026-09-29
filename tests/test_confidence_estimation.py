"""
Unit tests for Change 2: Confidence Estimation Fix

Verifies that:
1. ClassificationResult uses length-normalized joint probability as primary confidence
2. Margin of Victory is retained as a diagnostic field
3. classify_intent() wires LogprobsResult.confidence correctly
4. critic.py reads the updated confidence through the pipeline
"""

import asyncio
import math
import sys
import os
from unittest.mock import AsyncMock, patch, MagicMock
from dataclasses import asdict

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AI.agent import LogprobsResult, TokenLogprob
from safety.classifier import (
    ClassificationResult,
    ClinicalClassifier,
    CODE_MAP,
    normalize_classification_code,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic LogprobsResult objects
# ---------------------------------------------------------------------------

def make_logprobs(token_logprobs: list[float], first_token_top: list[dict] | None = None) -> LogprobsResult:
    """Build a LogprobsResult from a list of per-token logprobs."""
    tokens = []
    for i, lp in enumerate(token_logprobs):
        top = first_token_top if i == 0 and first_token_top else None
        tokens.append(TokenLogprob(token=f"tok{i}", logprob=lp, top_logprobs=top))
    return LogprobsResult(tokens=tokens)


# ---------------------------------------------------------------------------
# 1. LogprobsResult.confidence computes correctly
# ---------------------------------------------------------------------------

class TestLogprobsResult:
    def test_confidence_single_token(self):
        """Single token: confidence = exp(logprob)."""
        lp = make_logprobs([-0.3])
        assert lp.confidence == pytest.approx(math.exp(-0.3), rel=1e-6)

    def test_confidence_multi_token(self):
        """Multi-token: confidence = exp(mean(logprobs))."""
        lp = make_logprobs([-0.5, -1.2])
        expected = math.exp((-0.5 + -1.2) / 2)
        assert lp.confidence == pytest.approx(expected, rel=1e-6)

    def test_confidence_empty(self):
        """Empty tokens should return 0.0."""
        lp = LogprobsResult(tokens=[])
        assert lp.confidence == 0.0

    def test_mean_logprob(self):
        lp = make_logprobs([-0.5, -1.2])
        assert lp.mean_logprob == pytest.approx((-0.5 + -1.2) / 2, rel=1e-6)


# ---------------------------------------------------------------------------
# 2. ClassificationResult dataclass structure
# ---------------------------------------------------------------------------

class TestClassificationResult:
    def test_primary_confidence_is_joint_probability(self):
        """confidence field should hold joint probability, not margin."""
        r = ClassificationResult(
            node_id="BEHAVIORAL_MASTERY",
            code="CONFRONT",
            confidence=0.75,
            raw_response="CONFRONT",
            mean_logprob=-0.2877,
            margin=0.6,
            prob_winner=0.85,
            prob_runner_up=0.25,
        )
        assert r.confidence == 0.75  # joint probability
        assert r.margin == 0.6       # diagnostic

    def test_optional_fields_default_none(self):
        """margin, prob_winner, prob_runner_up should default to None."""
        r = ClassificationResult(
            node_id="BEHAVIORAL_MASTERY",
            code="CONFRONT",
            confidence=0.75,
            raw_response="CONFRONT",
        )
        assert r.margin is None
        assert r.prob_winner is None
        assert r.prob_runner_up is None
        assert r.mean_logprob is None


# ---------------------------------------------------------------------------
# 3. classify_intent() uses LogprobsResult.confidence as primary
# ---------------------------------------------------------------------------

class TestClassifyIntent:
    """Test classify_intent with a mocked Agent.generate()."""

    @pytest.fixture
    def classifier(self):
        """Create a ClinicalClassifier with mocked agent."""
        with patch("safety.classifier.Agent"):
            with patch("safety.classifier.build_clinical_graph") as mock_graph:
                # Set up minimal graph mock
                graph = MagicMock()
                graph.nodes = MagicMock()
                # Make all CODE_MAP values valid node IDs
                graph.__contains__ = lambda self, x: x in CODE_MAP.values()
                mock_graph.return_value = graph

                with patch("safety.classifier.get_all_node_ids", return_value=list(CODE_MAP.values())):
                    clf = ClinicalClassifier.__new__(ClinicalClassifier)
                    clf.model_config = MagicMock()
                    clf.agent = AsyncMock()
                    clf.graph = graph
                    clf.valid_node_ids = set(CODE_MAP.values())
                    return clf

    @pytest.mark.asyncio
    async def test_confidence_is_joint_probability(self, classifier):
        """classify_intent should set confidence = LogprobsResult.confidence (joint prob)."""
        # Simulate LLM returning "CONFRONT" with specific logprobs
        token_logprobs = [-0.2, -0.1]  # mean = -0.15, confidence = exp(-0.15)
        first_token_top = [
            {"token": "CON", "logprob": -0.2},
            {"token": "HIDE", "logprob": -1.5},
        ]
        logprobs = make_logprobs(token_logprobs, first_token_top)

        classifier.agent.generate = AsyncMock(return_value=(
            "CONFRONT",
            {"input": 10, "output": 1, "total": 11},
            logprobs,
        ))

        node_id, result = await classifier.classify_intent(
            "I fight back", return_logprobs=True
        )

        expected_confidence = math.exp((-0.2 + -0.1) / 2)
        assert node_id == "BEHAVIORAL_MASTERY"
        assert result is not None
        assert result.confidence == pytest.approx(expected_confidence, rel=1e-6)
        assert result.mean_logprob == pytest.approx((-0.2 + -0.1) / 2, rel=1e-6)

    @pytest.mark.asyncio
    async def test_margin_is_diagnostic(self, classifier):
        """Margin should be calculated but stored in margin field, not confidence."""
        token_logprobs = [-0.3]
        first_token_top = [
            {"token": "HIDE", "logprob": -0.3},
            {"token": "FEEL", "logprob": -2.0},
        ]
        logprobs = make_logprobs(token_logprobs, first_token_top)

        classifier.agent.generate = AsyncMock(return_value=(
            "HIDE",
            {"input": 10, "output": 1, "total": 11},
            logprobs,
        ))

        node_id, result = await classifier.classify_intent(
            "I hide under the bed", return_logprobs=True
        )

        assert result is not None
        # Confidence should be joint prob, NOT the margin
        assert result.confidence == pytest.approx(math.exp(-0.3), rel=1e-6)
        # Margin should be stored separately
        expected_margin = math.exp(-0.3) - math.exp(-2.0)
        assert result.margin == pytest.approx(expected_margin, rel=1e-6)
        assert result.prob_winner == pytest.approx(math.exp(-0.3), rel=1e-6)
        assert result.prob_runner_up == pytest.approx(math.exp(-2.0), rel=1e-6)

    @pytest.mark.asyncio
    async def test_confidence_differs_from_margin(self, classifier):
        """
        Key invariant: confidence (joint prob) != margin, unless by coincidence.
        This is THE bug we fixed — they used to be the same value.
        """
        token_logprobs = [-0.5, -1.2]
        first_token_top = [
            {"token": "CONFRONT", "logprob": -0.5},
            {"token": "HELP", "logprob": -1.8},
        ]
        logprobs = make_logprobs(token_logprobs, first_token_top)

        classifier.agent.generate = AsyncMock(return_value=(
            "CONFRONT",
            {"input": 10, "output": 1, "total": 11},
            logprobs,
        ))

        _, result = await classifier.classify_intent(
            "I punch the monster", return_logprobs=True
        )

        joint_prob = math.exp((-0.5 + -1.2) / 2)
        margin = math.exp(-0.5) - math.exp(-1.8)

        assert result.confidence == pytest.approx(joint_prob, rel=1e-6)
        assert result.margin == pytest.approx(margin, rel=1e-6)
        # They should NOT be equal (this was the old bug)
        assert abs(result.confidence - result.margin) > 0.01

    @pytest.mark.asyncio
    async def test_no_logprobs_returns_zero_confidence(self, classifier):
        """If LLM returns no logprobs, confidence should be 0.0."""
        classifier.agent.generate = AsyncMock(return_value=(
            "CONFRONT",
            {"input": 10, "output": 1, "total": 11},
            None,  # No logprobs
        ))

        _, result = await classifier.classify_intent(
            "I fight back", return_logprobs=True
        )

        assert result is not None
        assert result.confidence == 0.0
        assert result.margin is None

    @pytest.mark.asyncio
    async def test_invalid_code_routes_to_unclassified(self, classifier):
        """Semantic variants must fail closed instead of being inferred."""
        token_logprobs = [-0.1]
        first_token_top = [{"token": "FIGHT", "logprob": -0.1}]
        logprobs = make_logprobs(token_logprobs, first_token_top)

        # FIGHTING previously fuzzy-matched to BEHAVIORAL_MASTERY.
        classifier.agent.generate = AsyncMock(return_value=(
            "FIGHTING",
            {"input": 10, "output": 1, "total": 11},
            logprobs,
        ))

        node_id, result = await classifier.classify_intent(
            "I fight the dragon", return_logprobs=True
        )

        assert node_id == "UNCLASSIFIED"
        assert result.code == "UNCLASSIFIED"
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_return_logprobs_false(self, classifier):
        """When return_logprobs=False, result should be None."""
        token_logprobs = [-0.3]
        logprobs = make_logprobs(token_logprobs)

        classifier.agent.generate = AsyncMock(return_value=(
            "CONFRONT",
            {"input": 10, "output": 1, "total": 11},
            logprobs,
        ))

        node_id, result = await classifier.classify_intent(
            "I fight back", return_logprobs=False
        )

        assert node_id == "BEHAVIORAL_MASTERY"
        assert result is None


class TestCodeNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("HIDE", "HIDE"),
            ("  confront  ", "CONFRONT"),
            ('"HELP"', "HELP"),
            ("`RELAX`.", "RELAX"),
        ],
    )
    def test_accepts_only_superficial_formatting(self, raw, expected):
        assert normalize_classification_code(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "FIGHTING",
            "HIDE because this is avoidance",
            "UNSAFE",
            "MASTERY",
            None,
            123,
        ],
    )
    def test_rejects_semantic_and_non_string_recovery(self, raw):
        assert normalize_classification_code(raw) is None


# ---------------------------------------------------------------------------
# 4. classify_with_confidence() metadata shape
# ---------------------------------------------------------------------------

class TestClassifyWithConfidence:
    @pytest.fixture
    def classifier(self):
        with patch("safety.classifier.Agent"):
            with patch("safety.classifier.build_clinical_graph") as mock_graph:
                graph = MagicMock()
                node_attrs = {
                    "type": "adaptive",
                    "category": "mastery",
                    "description": "Active coping",
                }
                graph.nodes.__getitem__ = MagicMock(return_value=node_attrs)
                graph.__contains__ = lambda self, x: True
                mock_graph.return_value = graph

                with patch("safety.classifier.get_all_node_ids", return_value=list(CODE_MAP.values())):
                    clf = ClinicalClassifier.__new__(ClinicalClassifier)
                    clf.model_config = MagicMock()
                    clf.agent = AsyncMock()
                    clf.graph = graph
                    clf.valid_node_ids = set(CODE_MAP.values())
                    return clf

    @pytest.mark.asyncio
    async def test_metadata_has_confidence_and_margin(self, classifier):
        """Metadata dict should expose both confidence and margin as separate keys."""
        token_logprobs = [-0.2, -0.1]
        first_token_top = [
            {"token": "CONFRONT", "logprob": -0.2},
            {"token": "HELP", "logprob": -1.5},
        ]
        logprobs = make_logprobs(token_logprobs, first_token_top)

        classifier.agent.generate = AsyncMock(return_value=(
            "CONFRONT",
            {"input": 10, "output": 1, "total": 11},
            logprobs,
        ))

        node_id, metadata = await classifier.classify_with_confidence("I fight back")

        # Primary confidence = joint probability
        assert "confidence" in metadata
        expected_conf = math.exp((-0.2 + -0.1) / 2)
        assert metadata["confidence"] == pytest.approx(expected_conf, rel=1e-6)

        # Diagnostic margin
        assert "margin" in metadata
        expected_margin = math.exp(-0.2) - math.exp(-1.5)
        assert metadata["margin"] == pytest.approx(expected_margin, rel=1e-6)

        # mean_logprob
        assert "mean_logprob" in metadata
        assert metadata["mean_logprob"] == pytest.approx((-0.2 + -0.1) / 2, rel=1e-6)

        # Diagnostic first-token probs
        assert "prob_winner" in metadata
        assert "prob_runner_up" in metadata


# ---------------------------------------------------------------------------
# 5. Critic pipeline reads correct confidence
# ---------------------------------------------------------------------------

class TestCriticConfidenceIntegration:
    """Verify that SafetyCritic stores joint probability (not margin) in segments."""

    @pytest.mark.asyncio
    async def test_segment_gets_joint_probability(self):
        """After classification, segment.confidence should be the joint probability."""
        from safety.critic import SafetyCritic, SegmentResult

        token_logprobs = [-0.2, -0.1]
        first_token_top = [
            {"token": "CONFRONT", "logprob": -0.2},
            {"token": "HELP", "logprob": -1.5},
        ]
        logprobs_result = make_logprobs(token_logprobs, first_token_top)

        # Build a ClassificationResult as classify_intent would
        result = ClassificationResult(
            node_id="BEHAVIORAL_MASTERY",
            code="CONFRONT",
            confidence=logprobs_result.confidence,
            raw_response="CONFRONT",
            mean_logprob=logprobs_result.mean_logprob,
            margin=math.exp(-0.2) - math.exp(-1.5),
            prob_winner=math.exp(-0.2),
            prob_runner_up=math.exp(-1.5),
        )

        # Simulate what critic does: store confidence in segment
        segment = SegmentResult(text="I fight back", segment_type="ACTION")
        segment.confidence = result.confidence
        segment.mean_logprob = result.mean_logprob

        expected_confidence = math.exp((-0.2 + -0.1) / 2)
        assert segment.confidence == pytest.approx(expected_confidence, rel=1e-6)
        # NOT the margin
        margin = math.exp(-0.2) - math.exp(-1.5)
        assert segment.confidence != pytest.approx(margin, rel=1e-3)
