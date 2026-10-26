# tests/test_evaluator.py
"""Unit tests for the FaithfulnessEvaluator (NLI-based citation verifier)."""

from unittest.mock import patch

import numpy as np
import pytest

from rag_hybrid.src.eval.faithfulness import (
    CitationVerificationResult,
    FaithfulnessEvaluator,
)

# ---------------------------------------------------------------------------
# We mock the CrossEncoder so tests don't require the full NLI model in CI.
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_evaluator():
    """FaithfulnessEvaluator with a mocked CrossEncoder."""
    with patch("rag_hybrid.src.eval.faithfulness.CrossEncoder") as MockCE:
        instance = MockCE.return_value
        # Default: always return entailment prob = 0.9 → verified
        instance.predict.return_value = np.array([[0.02, 0.08, 0.90]])
        evaluator = FaithfulnessEvaluator.__new__(FaithfulnessEvaluator)
        evaluator.nli = instance
        evaluator.entail_idx = 2
        yield evaluator, instance


@pytest.fixture()
def sample_map():
    return {
        "c1": "Python is a programming language.",
        "c2": "FastAPI is a web framework.",
    }


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_claims_returns_perfect_score(self, mock_evaluator):
        evaluator, _ = mock_evaluator
        result = evaluator.evaluate_generation([], {})
        assert result.faithfulness_score == 1.0
        assert result.total_claims == 0
        assert result.verified_claims == 0
        assert result.violations == []

    def test_empty_claims_returns_correct_type(self, mock_evaluator):
        evaluator, _ = mock_evaluator
        result = evaluator.evaluate_generation([], {})
        assert isinstance(result, CitationVerificationResult)


# ---------------------------------------------------------------------------
# Verified claims (entailment prob above threshold)
# ---------------------------------------------------------------------------

class TestVerifiedClaims:
    def test_single_verified_claim(self, mock_evaluator, sample_map):
        evaluator, nli = mock_evaluator
        nli.predict.return_value = np.array([[0.02, 0.08, 0.90]])
        claims = [{"claim": "Python is a programming language.", "citation": "c1"}]
        result = evaluator.evaluate_generation(claims, sample_map)
        assert result.verified_claims == 1
        assert result.total_claims == 1
        assert result.faithfulness_score == 1.0
        assert result.violations == []

    def test_all_claims_verified(self, mock_evaluator, sample_map):
        evaluator, nli = mock_evaluator
        nli.predict.return_value = np.array([[0.0, 0.1, 0.90]])
        claims = [
            {"claim": "Python is a programming language.", "citation": "c1"},
            {"claim": "FastAPI is a web framework.", "citation": "c2"},
        ]
        result = evaluator.evaluate_generation(claims, sample_map)
        assert result.faithfulness_score == 1.0
        assert len(result.violations) == 0


# ---------------------------------------------------------------------------
# Hallucinated claims (entailment prob below threshold)
# ---------------------------------------------------------------------------

class TestHallucinatedClaims:
    def test_hallucination_creates_violation(self, mock_evaluator, sample_map):
        evaluator, nli = mock_evaluator
        # Low entailment → hallucination
        nli.predict.return_value = np.array([[0.70, 0.20, 0.10]])
        claims = [{"claim": "Python was invented in 1950.", "citation": "c1"}]
        result = evaluator.evaluate_generation(claims, sample_map)
        assert result.verified_claims == 0
        assert result.faithfulness_score == 0.0
        assert len(result.violations) == 1
        assert "Hallucination" in result.violations[0]

    def test_mixed_verified_and_hallucinated(self, mock_evaluator, sample_map):
        evaluator, nli = mock_evaluator
        # First call: entails; second call: contradicts
        nli.predict.side_effect = [
            np.array([[0.02, 0.08, 0.90]]),
            np.array([[0.70, 0.20, 0.10]]),
        ]
        claims = [
            {"claim": "Python is a programming language.", "citation": "c1"},
            {"claim": "Python was invented in 1950.", "citation": "c1"},
        ]
        result = evaluator.evaluate_generation(claims, sample_map)
        assert result.verified_claims == 1
        assert result.total_claims == 2
        assert round(result.faithfulness_score, 4) == 0.5
        assert len(result.violations) == 1


# ---------------------------------------------------------------------------
# Invalid citations
# ---------------------------------------------------------------------------

class TestInvalidCitations:
    def test_missing_citation_key_is_violation(self, mock_evaluator, sample_map):
        evaluator, _ = mock_evaluator
        claims = [{"claim": "Some claim.", "citation": "nonexistent_key"}]
        result = evaluator.evaluate_generation(claims, sample_map)
        assert result.verified_claims == 0
        assert len(result.violations) == 1
        assert "Invalid citation" in result.violations[0]

    def test_empty_citation_key_is_violation(self, mock_evaluator, sample_map):
        evaluator, _ = mock_evaluator
        claims = [{"claim": "Some claim.", "citation": ""}]
        result = evaluator.evaluate_generation(claims, sample_map)
        assert result.verified_claims == 0

    def test_all_invalid_citations_zero_score(self, mock_evaluator, sample_map):
        evaluator, _ = mock_evaluator
        claims = [
            {"claim": "Claim A.", "citation": "bad1"},
            {"claim": "Claim B.", "citation": "bad2"},
        ]
        result = evaluator.evaluate_generation(claims, sample_map)
        assert result.faithfulness_score == 0.0


# ---------------------------------------------------------------------------
# Score bounds
# ---------------------------------------------------------------------------

class TestScoreBounds:
    def test_score_is_float(self, mock_evaluator, sample_map):
        evaluator, _ = mock_evaluator
        claims = [{"claim": "Python is a programming language.", "citation": "c1"}]
        result = evaluator.evaluate_generation(claims, sample_map)
        assert isinstance(result.faithfulness_score, float)

    def test_score_between_0_and_1(self, mock_evaluator, sample_map):
        evaluator, nli = mock_evaluator
        nli.predict.return_value = np.array([[0.5, 0.3, 0.20]])
        claims = [{"claim": "Some claim.", "citation": "c1"}]
        result = evaluator.evaluate_generation(claims, sample_map)
        assert 0.0 <= result.faithfulness_score <= 1.0
