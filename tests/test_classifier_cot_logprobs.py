#!/usr/bin/env python3
"""
Tests for System 2 classification-token log-probability extraction.

Background: System 2 derives its confidence from the log probabilities of the
classification code tokens. Before the span-scan fix, the extractor looked for a
whole `classification` token and a whole code token. Mistral Small emits neither
(`class` + `ification`, `CON` + `FR` + `ONT`), so extraction missed on 87.1% of
development calls and confidence silently fell back to the mean log probability
of the entire response. See docs/planning/S2_logprob_diagnostic_findings.md.

These tests use synthetic token sequences copied from real Mistral Small
responses, so they run without API access.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from AI.agent import LogprobsResult, TokenLogprob
from safety.classifier_cot import ClinicalClassifierCoT


def make_logprobs(token_strings, logprob_by_token=None):
    """Build a LogprobsResult from token strings, defaulting to logprob -0.2."""
    logprob_by_token = logprob_by_token or {}
    return LogprobsResult(tokens=[
        TokenLogprob(token=t, logprob=logprob_by_token.get(i, -0.2))
        for i, t in enumerate(token_strings)
    ])


# Verbatim tail of a real Mistral Small System 2 response (CONFRONT case).
REAL_TAIL_CONFRONT = [
    ' or', ' descriptive', ',', ' so', ' LO', 'OK', ' and', ' FE', 'EL', ' are',
    ' not', ' relevant', '.",\n', ' ', ' "', 'class', 'ification', '":', ' "',
    'CON', 'FR', 'ONT', '"\n', '}', '</s>',
]

# Real response where the rationale contains the bare word "confront" before the
# JSON value — the pre-fix tail scan matched the prose word, not the label.
REAL_TAIL_PROSE_COLLISION = [
    ' neg', 'ate', ' the', ' initial', ' adaptive', ' action', ' of', ' confront',
    'ing', ' the', ' attacker', '.",\n', ' ', ' "', 'class', 'ification', '":',
    ' "', 'CON', 'FR', 'ONT', '"\n', '}\n', '```', '</s>',
]


class TestSpanScan:
    """The character-offset scan is the primary extraction path."""

    def test_finds_split_key_and_split_code(self):
        classifier = ClinicalClassifierCoT.__new__(ClinicalClassifierCoT)
        logprobs = make_logprobs(REAL_TAIL_CONFRONT)

        tokens, strategy = classifier._locate_classification_tokens(logprobs, "CONFRONT")

        assert strategy == "span_scan"
        assert [t.token for t in tokens] == ['CON', 'FR', 'ONT']

    def test_prefers_json_value_over_prose_occurrence(self):
        """A bare code word in the rationale must not be mistaken for the label."""
        classifier = ClinicalClassifierCoT.__new__(ClinicalClassifierCoT)
        # Give the prose token a distinctive logprob so a wrong match is visible.
        logprobs = make_logprobs(REAL_TAIL_PROSE_COLLISION, {7: -3.5})

        tokens, strategy = classifier._locate_classification_tokens(logprobs, "CONFRONT")

        assert strategy == "span_scan"
        assert [t.token for t in tokens] == ['CON', 'FR', 'ONT']
        assert all(t.logprob == -0.2 for t in tokens)

    def test_single_token_code(self):
        classifier = ClinicalClassifierCoT.__new__(ClinicalClassifierCoT)
        logprobs = make_logprobs([
            '{"', 'context', '_summary', '":', ' "', 'He', ' hides', '.",',
            ' "', 'class', 'ification', '":', ' "', 'HIDE', '"}',
        ])

        tokens, strategy = classifier._locate_classification_tokens(logprobs, "HIDE")

        assert strategy == "span_scan"
        assert [t.token for t in tokens] == ['HIDE']

    def test_takes_last_classification_key(self):
        """The word appears in the rationale; only the final JSON field counts."""
        classifier = ClinicalClassifierCoT.__new__(ClinicalClassifierCoT)
        logprobs = make_logprobs([
            '{"clinical_rationale": "The "classification" here is borderline.",',
            ' "class', 'ification', '": "', 'DENY', '"}',
        ])

        tokens, strategy = classifier._locate_classification_tokens(logprobs, "DENY")

        assert strategy == "span_scan"
        assert [t.token for t in tokens] == ['DENY']

    def test_mismatched_code_does_not_span_match(self):
        """If the parsed code and the JSON value disagree, do not claim a hit."""
        classifier = ClinicalClassifierCoT.__new__(ClinicalClassifierCoT)
        logprobs = make_logprobs(['{"class', 'ification', '": "', 'HIDE', '"}'])

        tokens, strategy = classifier._locate_classification_tokens(logprobs, "ESCAPE")

        assert strategy == "miss"
        assert tokens is None

    def test_empty_logprobs(self):
        classifier = ClinicalClassifierCoT.__new__(ClinicalClassifierCoT)

        tokens, strategy = classifier._locate_classification_tokens(
            LogprobsResult(tokens=[]), "HIDE"
        )

        assert strategy == "miss"
        assert tokens is None


class TestExtractionWrapper:
    """The legacy helper still returns just the token span."""

    def test_wrapper_returns_tokens(self):
        classifier = ClinicalClassifierCoT.__new__(ClinicalClassifierCoT)
        logprobs = make_logprobs(REAL_TAIL_CONFRONT)

        tokens = classifier._extract_classification_logprobs(logprobs, "CONFRONT")

        assert [t.token for t in tokens] == ['CON', 'FR', 'ONT']


class TestPreFixBehaviourIsGone:
    """Regression guard for the bug the diagnostic found."""

    def test_split_key_no_longer_falls_back(self):
        """
        Pre-fix, neither scan could see through `class` + `ification`, so this
        sequence produced a miss and confidence over the whole response.
        """
        classifier = ClinicalClassifierCoT.__new__(ClinicalClassifierCoT)
        logprobs = make_logprobs(REAL_TAIL_CONFRONT)

        _tokens, strategy = classifier._locate_classification_tokens(logprobs, "CONFRONT")

        assert strategy != "miss"
