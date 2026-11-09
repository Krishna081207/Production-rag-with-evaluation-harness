# tests/test_core.py
"""Core unit tests for the rag_hybrid package.

These tests verify the three main building blocks:
1️⃣ Hybrid retrieval returns DocumentChunk objects.
2️⃣ Generation produces a GenerationResult with the expected fields.
3️⃣ FaithfulnessEvaluator returns a numeric score and optional violations.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_hybrid.src.eval.faithfulness import CitationVerificationResult, FaithfulnessEvaluator
from rag_hybrid.src.generator.generator import GenerationResult, generate_answer
from rag_hybrid.src.retriever.hybrid import DocumentChunk, ProductionHybridRAG


@pytest.fixture(scope="module")
def sample_chunks():
    """Create a tiny in‑memory corpus for testing."""
    return [
        DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            text="Python is a programming language.",
            parent_text="Python is a programming language. It is popular.",
        ),
        DocumentChunk(
            chunk_id="c2",
            doc_id="d1",
            text="FastAPI is a web framework.",
            parent_text="FastAPI is a web framework for building APIs.",
        ),
    ]

def test_hybrid_retrieval(sample_chunks):
    rag = ProductionHybridRAG()
    rag.index_documents(sample_chunks)
    results = rag.hybrid_retrieve("What is Python?", top_candidates=10, final_k=2)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(c, DocumentChunk) for c in results)

def test_generator_basic(sample_chunks, monkeypatch):
    rag = ProductionHybridRAG()
    rag.index_documents(sample_chunks)
    retrieved = rag.hybrid_retrieve("What is FastAPI?", top_candidates=10, final_k=1)
    monkeypatch.setenv("GROQ_API_KEY", "dummy-key")
    message = MagicMock(content=f"FastAPI is a web framework [{retrieved[0].chunk_id}].")
    response = MagicMock(choices=[MagicMock(message=message)])
    with patch("rag_hybrid.src.generator.generator.Groq") as mock_groq:
        mock_groq.return_value.chat.completions.create.return_value = response
        result: GenerationResult = generate_answer("What is FastAPI?", retrieved)
    assert isinstance(result, GenerationResult)
    assert isinstance(result.answer, str)
    assert "[Chunk_" in result.answer or not result.citations

def test_faithfulness_evaluator(sample_chunks):
    retrieved_map = {c.chunk_id: c.parent_text for c in sample_chunks}
    claims = [{"claim": "Python is a programming language.", "citation": "Chunk_c1"}]
    with patch("rag_hybrid.src.eval.faithfulness.CrossEncoder") as mock_encoder:
        mock_encoder.return_value.model.config.label2id = {"entailment": 1}
        mock_encoder.return_value.predict.return_value = np.array([[0.02, 0.9, 0.08]])
        evaluator = FaithfulnessEvaluator()
        res: CitationVerificationResult = evaluator.evaluate_generation(claims, retrieved_map)
    assert isinstance(res.faithfulness_score, float)
    assert 0.0 <= res.faithfulness_score <= 1.0
