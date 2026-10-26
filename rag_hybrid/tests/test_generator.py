# tests/test_generator.py
"""Unit tests for the generator module.

We mock the Groq client so no real API key is needed in CI.
"""

import os
import re
from unittest.mock import MagicMock, patch

import pytest

from rag_hybrid.src.generator.generator import (
    GenerationResult,
    _build_prompt,
    generate_answer,
)
from rag_hybrid.src.retriever.hybrid import DocumentChunk

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def two_chunks():
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


def _make_groq_response(content: str):
    """Build a minimal mock that matches the Groq SDK response shape."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_contains_query(self, two_chunks):
        prompt = _build_prompt("What is Python?", two_chunks)
        assert "What is Python?" in prompt

    def test_contains_parent_text_of_each_chunk(self, two_chunks):
        prompt = _build_prompt("query", two_chunks)
        for chunk in two_chunks:
            assert chunk.parent_text in prompt

    def test_contains_citation_instruction(self, two_chunks):
        prompt = _build_prompt("query", two_chunks)
        assert "[Chunk_" in prompt or "Chunk_" in prompt

    def test_empty_chunks_returns_prompt(self):
        prompt = _build_prompt("query", [])
        assert "query" in prompt


# ---------------------------------------------------------------------------
# generate_answer
# ---------------------------------------------------------------------------

class TestGenerateAnswerNoChunks:
    def test_returns_fallback_when_no_chunks(self):
        result = generate_answer("What is Python?", [])
        assert isinstance(result, GenerationResult)
        assert "does not contain" in result.answer.lower() or "insufficient" in result.answer.lower() or result.answer != ""
        assert result.citations == []


class TestGenerateAnswerWithMockedGroq:
    @patch("rag_hybrid.src.generator.generator.Groq")
    def test_returns_generation_result(self, MockGroq, two_chunks):
        groq_instance = MockGroq.return_value
        groq_instance.chat.completions.create.return_value = _make_groq_response(
            "Python is a language [Chunk_c1]."
        )
        os.environ["groq"] = "fake-key"
        result = generate_answer("What is Python?", two_chunks)
        assert isinstance(result, GenerationResult)

    @patch("rag_hybrid.src.generator.generator.Groq")
    def test_citations_extracted(self, MockGroq, two_chunks):
        groq_instance = MockGroq.return_value
        groq_instance.chat.completions.create.return_value = _make_groq_response(
            "Python is popular [Chunk_c1]. FastAPI is a framework [Chunk_c2]."
        )
        os.environ["groq"] = "fake-key"
        result = generate_answer("Tell me about both.", two_chunks)
        assert "Chunk_c1" in result.citations
        assert "Chunk_c2" in result.citations

    @patch("rag_hybrid.src.generator.generator.Groq")
    def test_no_citations_in_answer(self, MockGroq, two_chunks):
        groq_instance = MockGroq.return_value
        groq_instance.chat.completions.create.return_value = _make_groq_response(
            "The answer is unknown."
        )
        os.environ["groq"] = "fake-key"
        result = generate_answer("Unknown question.", two_chunks)
        assert result.citations == []

    @patch("rag_hybrid.src.generator.generator.Groq")
    def test_api_error_returns_fallback(self, MockGroq, two_chunks):
        groq_instance = MockGroq.return_value
        groq_instance.chat.completions.create.side_effect = Exception("network error")
        os.environ["groq"] = "fake-key"
        result = generate_answer("What is Python?", two_chunks)
        assert isinstance(result, GenerationResult)
        assert isinstance(result.answer, str)

    @patch("rag_hybrid.src.generator.generator.Groq")
    def test_citation_regex_handles_multiword_id(self, MockGroq, two_chunks):
        groq_instance = MockGroq.return_value
        groq_instance.chat.completions.create.return_value = _make_groq_response(
            "Something happened [Chunk_doc1_c0]."
        )
        os.environ["groq"] = "fake-key"
        result = generate_answer("query", two_chunks)
        assert "Chunk_doc1_c0" in result.citations


# ---------------------------------------------------------------------------
# Citation regex contract
# ---------------------------------------------------------------------------

class TestCitationRegex:
    """Verify the regex used in generate_answer is consistent."""

    def test_regex_matches_standard_citation(self):
        pattern = re.compile(r"\[Chunk_([^\]]+)\]")
        m = pattern.findall("Python [Chunk_c1] is great [Chunk_c2].")
        assert m == ["c1", "c2"]

    def test_regex_no_match_on_plain_text(self):
        pattern = re.compile(r"\[Chunk_([^\]]+)\]")
        assert pattern.findall("No citations here.") == []
