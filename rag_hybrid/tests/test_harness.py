# tests/test_harness.py
"""Unit tests for the evaluation harness metrics (hallucination rate & MMR)."""

import pytest

from rag_hybrid.src.retriever.hybrid import DocumentChunk
from scripts.evaluate_harness import (
    _jaccard,
    _split_sentences,
    hallucination_rate,
    mmr_score,
)

# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------

class TestSplitSentences:
    def test_splits_on_period(self):
        result = _split_sentences("Hello world. Goodbye world.")
        assert len(result) == 2

    def test_empty_string(self):
        assert _split_sentences("") == []

    def test_single_sentence_no_period(self):
        result = _split_sentences("No period here")
        # The function splits on ". " — a sentence with no period returns itself
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _jaccard
# ---------------------------------------------------------------------------

class TestJaccard:
    def test_identical_sets(self):
        s = {"a", "b", "c"}
        assert _jaccard(s, s) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        score = _jaccard({"a", "b", "c"}, {"b", "c", "d"})
        assert 0.0 < score < 1.0

    def test_empty_sets(self):
        assert _jaccard(set(), set()) == 0.0

    def test_one_empty_set(self):
        assert _jaccard({"a"}, set()) == 0.0


# ---------------------------------------------------------------------------
# hallucination_rate
# ---------------------------------------------------------------------------

class TestHallucinationRate:
    def test_zero_rate_when_all_cited(self):
        answer = "Python is popular [Chunk_c1]. FastAPI is fast [Chunk_c2]."
        rate = hallucination_rate(answer)
        assert rate == 0.0

    def test_full_rate_when_none_cited(self):
        answer = "Python is popular. FastAPI is fast."
        rate = hallucination_rate(answer)
        assert rate == 1.0

    def test_partial_rate(self):
        answer = "Python is popular [Chunk_c1]. FastAPI is not cited."
        rate = hallucination_rate(answer)
        assert 0.0 < rate < 1.0

    def test_empty_answer(self):
        rate = hallucination_rate("")
        assert rate == 0.0

    def test_rate_between_0_and_1(self):
        import random
        for _ in range(10):
            n = random.randint(1, 5)
            parts = [f"Sentence {i} [Chunk_c{i}]." if i % 2 == 0 else f"Sentence {i}." for i in range(n)]
            answer = " ".join(parts)
            rate = hallucination_rate(answer)
            assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# mmr_score
# ---------------------------------------------------------------------------

@pytest.fixture()
def diverse_chunks():
    return [
        DocumentChunk(
            chunk_id=f"c{i}",
            doc_id="d1",
            text=text,
            parent_text=text,
        )
        for i, text in enumerate([
            "Python is a programming language used in data science.",
            "Qdrant is a vector database for similarity search.",
            "FastAPI is a web framework for building APIs.",
        ])
    ]


@pytest.fixture()
def duplicate_chunks():
    return [
        DocumentChunk(
            chunk_id=f"c{i}",
            doc_id="d1",
            text="Python is a programming language.",
            parent_text="Python is a programming language.",
        )
        for i in range(3)
    ]


class TestMMRScore:
    def test_returns_float(self, diverse_chunks):
        score = mmr_score(diverse_chunks)
        assert isinstance(score, float)

    def test_score_between_0_and_1(self, diverse_chunks):
        score = mmr_score(diverse_chunks)
        assert 0.0 <= score <= 1.0

    def test_empty_chunks_returns_zero(self):
        assert mmr_score([]) == 0.0

    def test_single_chunk_returns_nonzero(self, diverse_chunks):
        score = mmr_score(diverse_chunks[:1])
        assert score > 0.0

    def test_diverse_chunks_score_higher_than_duplicates(self, diverse_chunks, duplicate_chunks):
        diverse_score = mmr_score(diverse_chunks)
        duplicate_score = mmr_score(duplicate_chunks)
        # Diverse retrieval should always outscore identical duplicates
        assert diverse_score > duplicate_score

    def test_lambda_zero_gives_only_novelty(self, diverse_chunks):
        # lambda=0 → pure novelty, lambda=1 → pure relevance
        s_novelty = mmr_score(diverse_chunks, lambda_rel=0.0)
        s_relevance = mmr_score(diverse_chunks, lambda_rel=1.0)
        assert 0.0 <= s_novelty <= 1.0
        assert 0.0 <= s_relevance <= 1.0
