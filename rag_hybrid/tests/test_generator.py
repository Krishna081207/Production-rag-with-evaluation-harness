# tests/test_generator.py
"""Unit tests for the generator module.

We mock the Groq client so no real API key is needed in CI.
"""

import json
import re
from unittest.mock import MagicMock, patch

import pytest

from rag_hybrid.src.generator.generator import (
    ABSTENTION,
    GenerationResult,
    _build_prompt,
    extract_claims,
    generate_answer,
    without_citation_markers,
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


def _make_groq_response(content: str, finish_reason: str = "stop"):
    """Build a minimal mock that matches the Groq SDK response shape."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
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

    def test_contains_passages_and_nearby_context(self, two_chunks):
        prompt = _build_prompt("query", two_chunks)
        for chunk in two_chunks:
            assert chunk.text in prompt
        assert "It is popular." in prompt

    def test_shared_context_maps_each_id_to_its_own_passage(self, two_chunks):
        parent_text = " ".join(chunk.text for chunk in two_chunks)
        chunks = [chunk.model_copy(update={"parent_text": parent_text}) for chunk in two_chunks]

        prompt = _build_prompt("How do these technologies relate?", chunks)

        assert parent_text not in prompt
        for chunk in chunks:
            assert f'<passage id="Chunk_{chunk.chunk_id}">\n{chunk.text}\n</passage>' in prompt

    def test_long_parent_context_is_trimmed_around_the_exact_passage(self, two_chunks):
        chunk = two_chunks[0].model_copy(update={
            "parent_text": "distant " * 100 + "left " + two_chunks[0].text + " right " + "far " * 100,
        })
        prompt = _build_prompt("What is Python?", [chunk])
        assert prompt.count(chunk.text) == 1
        assert "left\n</context>" in prompt
        assert ">\nright " in prompt
        assert prompt.count("distant ") == 59
        assert prompt.count(" far") == 59

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

    def test_fallback_uses_query_relevant_child_sentence_and_attaches_citation(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("groq", raising=False)
        chunk = DocumentChunk(
            chunk_id="c2",
            doc_id="paper",
            text=(
                "retrieval costs and accu- racy. This approach enables evaluating RAG systems "
                "across a spectrum of cost constraints rather than at fixed operating points."
            ),
            parent_text=(
                "10 Front. Comput. Sci., 2025, 0(0): 1–18 Table 1. "
                "retrieval costs and accuracy. This approach enables evaluating RAG systems "
                "across a spectrum of cost constraints rather than at fixed operating points."
            ),
        )

        result = generate_answer("rag evaluation ways", [chunk])

        assert result.answer.startswith("This approach enables evaluating RAG systems")
        assert result.answer.endswith("[Chunk_c2].")
        assert extract_claims(result.answer) == [
            {
                "claim": (
                    "This approach enables evaluating RAG systems across a spectrum of "
                    "cost constraints rather than at fixed operating points."
                ),
                "citation": "Chunk_c2",
            }
        ]

    def test_claim_parser_keeps_citation_after_period_with_its_claim(self):
        assert extract_claims(
            "The paper reports an improvement. [Chunk_c1] The follow-up was inconclusive."
        ) == [
            {"claim": "The paper reports an improvement.", "citation": "Chunk_c1"},
            {"claim": "The follow-up was inconclusive.", "citation": ""},
        ]


class TestGenerateAnswerWithMockedGroq:
    @pytest.fixture(autouse=True)
    def provider_configuration(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")
        monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")

    @patch("rag_hybrid.src.generator.generator.Groq")
    def test_structured_answer_preserves_explanatory_paragraphs(self, MockGroq, two_chunks):
        create = MockGroq.return_value.chat.completions.create
        create.return_value = _make_groq_response(json.dumps({
            "abstain": False,
            "paragraphs": [
                [{"text": "Python is a programming language.", "citation": "Chunk_c1"}],
                [{"text": "FastAPI provides tools for building web APIs, e.g. route handlers.",
                  "citation": "Chunk_c2"}],
            ],
        }))

        result = generate_answer("How are Python and FastAPI different?", two_chunks)

        assert result.mode == "generated"
        assert result.answer == (
            "Python is a programming language [Chunk_c1].\n\n"
            "FastAPI provides tools for building web APIs, e.g. route handlers [Chunk_c2]."
        )
        assert result.citations == ["Chunk_c1", "Chunk_c2"]
        assert len(extract_claims(result.answer)) == 2
        create.assert_called_once()
        assert create.call_args.kwargs["response_format"]["json_schema"]["strict"] is True

    @patch("rag_hybrid.src.generator.generator.Groq")
    def test_retries_truncated_output_and_rejects_citation_only_retry(self, MockGroq, two_chunks):
        create = MockGroq.return_value.chat.completions.create
        create.side_effect = [
            _make_groq_response('{"abstain":false,"paragraphs":', finish_reason="length"),
            _make_groq_response("[Chunk_c1]."),
        ]

        result = generate_answer("What is Python?", two_chunks)

        assert result.mode == "extractive"
        assert result.fallback_reason == "citation_validation_error"
        assert result.answer == "Python is a programming language [Chunk_c1]."
        assert create.call_count == 2
        assert "response_format" not in create.call_args.kwargs

    @patch("rag_hybrid.src.generator.generator.Groq")
    def test_plain_model_abstention_has_the_correct_mode(self, MockGroq, two_chunks, monkeypatch):
        monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        MockGroq.return_value.chat.completions.create.return_value = _make_groq_response(ABSTENTION)

        result = generate_answer("What is quantum computing?", two_chunks)

        assert result.mode == "abstention"
        assert result.citations == []

    @patch("rag_hybrid.src.generator.generator.Groq")
    def test_returns_generation_result(self, MockGroq, two_chunks):
        groq_instance = MockGroq.return_value
        groq_instance.chat.completions.create.return_value = _make_groq_response(
            "Python is a language [Chunk_c1]."
        )
        result = generate_answer("What is Python?", two_chunks)
        assert isinstance(result, GenerationResult)

    @patch("rag_hybrid.src.generator.generator.Groq")
    def test_citations_extracted(self, MockGroq, two_chunks):
        groq_instance = MockGroq.return_value
        groq_instance.chat.completions.create.return_value = _make_groq_response(
            "Python is popular [Chunk_c1]. FastAPI is a framework [Chunk_c2]."
        )
        result = generate_answer("Tell me about both.", two_chunks)
        assert "Chunk_c1" in result.citations
        assert "Chunk_c2" in result.citations

    @patch("rag_hybrid.src.generator.generator.Groq")
    def test_no_citations_in_answer(self, MockGroq, two_chunks):
        groq_instance = MockGroq.return_value
        groq_instance.chat.completions.create.return_value = _make_groq_response(
            "The answer is unknown."
        )
        result = generate_answer("Unknown question.", two_chunks)
        assert result.mode == "abstention"
        assert result.citations == []
        assert result.answer == ABSTENTION

    @patch("rag_hybrid.src.generator.generator.Groq")
    def test_api_error_returns_fallback(self, MockGroq, two_chunks):
        groq_instance = MockGroq.return_value
        groq_instance.chat.completions.create.side_effect = Exception("network error")
        result = generate_answer("What is Python?", two_chunks)
        assert isinstance(result, GenerationResult)
        assert isinstance(result.answer, str)
        assert result.citations == ["Chunk_c1"]
        assert "[Chunk_c1]" in result.answer

    @patch("rag_hybrid.src.generator.generator.Groq")
    def test_citation_regex_handles_multiword_id(self, MockGroq, two_chunks):
        groq_instance = MockGroq.return_value
        groq_instance.chat.completions.create.return_value = _make_groq_response(
            "Something happened [Chunk_doc1_c0]."
        )
        chunk = DocumentChunk(
            chunk_id="doc1_c0",
            doc_id="doc1",
            text="Something happened.",
            parent_text="Something happened.",
        )
        result = generate_answer("query", [chunk])
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

    def test_reader_facing_prose_preserves_paragraphs_and_punctuation(self):
        text = (
            "Accuracy equals (TP + TN) / total [Chunk_c1] [Chunk_c2].\n\n"
            "Precision@5 is a separate metric (Chunk_c3) ."
        )
        assert without_citation_markers(text) == (
            "Accuracy equals (TP + TN) / total.\n\nPrecision@5 is a separate metric."
        )
