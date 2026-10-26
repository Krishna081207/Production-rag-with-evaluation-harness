# tests/test_api.py
"""Integration tests for the FastAPI application using TestClient.

No real LLM / vector DB calls are made; we mock the heavy components so
tests run fast in CI.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Create a TestClient with all heavy singletons mocked."""

    with (
        patch("rag_hybrid.src.retriever.hybrid.SentenceTransformer"),
        patch("rag_hybrid.src.retriever.hybrid.CrossEncoder"),
        patch("rag_hybrid.src.retriever.hybrid.QdrantClient"),
        patch("rag_hybrid.src.eval.faithfulness.CrossEncoder"),
    ):
        import importlib

        from rag_hybrid.src.api import app as app_module
        importlib.reload(app_module)  # ensure fresh module with mocks in place

        from rag_hybrid.src.api.app import app
        yield TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, client):
        resp = client.get("/health")
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /query – input validation
# ---------------------------------------------------------------------------

class TestQueryValidation:
    def test_missing_body_returns_422(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_wrong_content_type_returns_422(self, client):
        resp = client.post("/query", data="plain text")
        assert resp.status_code in (422, 415)


# ---------------------------------------------------------------------------
# /query – mocked pipeline
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    @patch("rag_hybrid.src.api.app._rag")
    @patch("rag_hybrid.src.api.app._evaluator")
    @patch("rag_hybrid.src.generator.generator.Groq")
    def test_successful_query(self, MockGroq, mock_eval, mock_rag, client):
        from rag_hybrid.src.eval.faithfulness import CitationVerificationResult
        from rag_hybrid.src.retriever.hybrid import DocumentChunk

        chunk = DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            text="Python is popular.",
            parent_text="Python is popular.",
        )
        mock_rag.hybrid_retrieve.return_value = [chunk]

        # Mock Groq
        msg = MagicMock()
        msg.content = "Python is popular [Chunk_c1]."
        choice = MagicMock()
        choice.message = msg
        groq_resp = MagicMock()
        groq_resp.choices = [choice]
        MockGroq.return_value.chat.completions.create.return_value = groq_resp

        mock_eval.evaluate_generation.return_value = CitationVerificationResult(
            total_claims=1,
            verified_claims=1,
            faithfulness_score=1.0,
            violations=[],
        )

        resp = client.post("/query", json={"query": "What is Python?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "faithfulness_score" in body
        assert "citations" in body

    @patch("rag_hybrid.src.api.app._rag")
    @patch("rag_hybrid.src.api.app._evaluator")
    @patch("rag_hybrid.src.generator.generator.Groq")
    def test_empty_query_string_returns_400(self, MockGroq, mock_eval, mock_rag, client):
        resp = client.post("/query", json={"query": ""})
        assert resp.status_code == 400

    @patch("rag_hybrid.src.api.app._rag")
    @patch("rag_hybrid.src.api.app._evaluator")
    def test_no_chunks_retrieved_returns_fallback(self, mock_eval, mock_rag, client):
        from rag_hybrid.src.eval.faithfulness import CitationVerificationResult
        mock_rag.hybrid_retrieve.return_value = []
        mock_eval.evaluate_generation.return_value = CitationVerificationResult(
            total_claims=0,
            verified_claims=0,
            faithfulness_score=1.0,
            violations=[],
        )
        resp = client.post("/query", json={"query": "What is quantum computing?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert body["citations"] == []

    @patch("rag_hybrid.src.api.app._rag")
    @patch("rag_hybrid.src.api.app._evaluator")
    @patch("rag_hybrid.src.generator.generator.Groq")
    def test_faithfulness_score_in_range(self, MockGroq, mock_eval, mock_rag, client):
        from rag_hybrid.src.eval.faithfulness import CitationVerificationResult
        from rag_hybrid.src.retriever.hybrid import DocumentChunk

        chunk = DocumentChunk(
            chunk_id="c1", doc_id="d1",
            text="Test.", parent_text="Test.",
        )
        mock_rag.hybrid_retrieve.return_value = [chunk]

        msg = MagicMock()
        msg.content = "Test [Chunk_c1]."
        choice = MagicMock()
        choice.message = msg
        groq_resp = MagicMock()
        groq_resp.choices = [choice]
        MockGroq.return_value.chat.completions.create.return_value = groq_resp

        mock_eval.evaluate_generation.return_value = CitationVerificationResult(
            total_claims=1,
            verified_claims=1,
            faithfulness_score=0.75,
            violations=[],
        )

        resp = client.post("/query", json={"query": "Test question."})
        assert resp.status_code == 200
        score = resp.json()["faithfulness_score"]
        assert 0.0 <= score <= 1.0
