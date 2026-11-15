from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest

from rag_hybrid.src.generator.generator import ABSTENTION, GenerationResult
from rag_hybrid.src.ingestion import chunk_document_segments
from rag_hybrid.src.paper_qa import PaperQA
from rag_hybrid.src.retriever.hybrid import ProductionHybridRAG


@pytest.fixture
def rag(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("groq", raising=False)
    library = ProductionHybridRAG(vector_db="qdrant", qdrant_path=":memory:")
    monkeypatch.setattr(library, "_upsert_chunks", Mock())
    monkeypatch.setattr(library, "_delete_chunks", Mock())
    library.replace_sources(chunk_document_segments(
        "paper.pdf", [(3, "The objective is to measure flowering time in garden peas.")]
    ))
    return library


def test_prompt_alias_cache_tracks_replacement_and_removal(rag, monkeypatch):
    answerer = PaperQA(rag, Mock())
    retrieve = Mock(wraps=rag.hybrid_retrieve)
    monkeypatch.setattr(rag, "hybrid_retrieve", retrieve)

    original = answerer.answer("What does this PDF tell?", 5, mode="bm25")
    repeated = answerer.answer("WHAT IS THIS PAPER ABOUT?!", 5, mode="bm25")
    assert original["cache_status"] == "miss"
    assert repeated["cache_status"] == "hit"
    assert repeated["answer"] == original["answer"]
    assert repeated["answer"] == "The objective is to measure flowering time in garden peas."
    assert repeated["citation_details"][0]["page_number"] == 3
    retrieve.assert_called_once()

    rag.replace_sources(chunk_document_segments(
        "paper.pdf", [(7, "The objective is to measure steam pressure in industrial boilers.")]
    ))
    replacement = answerer.answer("What does this PDF tell me?", 5, mode="bm25")
    assert replacement["cache_status"] == "miss"
    assert "industrial boilers" in replacement["answer"]
    assert replacement["citation_details"][0]["page_number"] == 7

    rag.remove_sources(["paper.pdf"])
    deleted = answerer.answer("What does this PDF tell me?", 5, mode="bm25")
    assert deleted["cache_status"] == "miss"
    assert deleted["answer"] == ABSTENTION
    assert deleted["citations"] == []


@pytest.mark.parametrize("question", [
    "What does this PDF tell me about dinosaurs?",
    "What is this leading to?",
])
def test_abstains_when_paper_has_no_support_for_question(rag, question):
    result = PaperQA(rag, Mock()).answer(question, 5, mode="bm25")
    assert result["answer_mode"] == "abstention"
    assert result["answer"] == ABSTENTION


def test_transient_fallback_is_not_cached(rag, monkeypatch):
    generate = Mock(return_value=GenerationResult(
        answer=ABSTENTION, citations=[], mode="abstention", fallback_reason="provider_error"
    ))
    monkeypatch.setattr("rag_hybrid.src.paper_qa.generate_answer", generate)
    answerer = PaperQA(rag, Mock())

    for _ in range(2):
        assert answerer.answer("What does this PDF tell?", 5, mode="bm25")["cache_status"] == "miss"
    assert generate.call_count == 2


def test_concurrent_aliases_share_in_progress_work(rag, monkeypatch):
    waiting = Event()
    release = Event()
    started = Event()
    generate = Mock()

    class ObservableFuture(Future):
        def result(self, timeout=None):
            waiting.set()
            return super().result(timeout=timeout)

    def generate_once(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return GenerationResult(answer=ABSTENTION, citations=[], mode="abstention")

    generate.side_effect = generate_once
    monkeypatch.setattr("rag_hybrid.src.paper_qa.generate_answer", generate)
    monkeypatch.setattr("rag_hybrid.src.paper_qa.Future", ObservableFuture)
    answerer = PaperQA(rag, Mock())

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(answerer.answer, "What does this PDF tell?", 5, "bm25")
        try:
            assert started.wait(5)
            second = executor.submit(answerer.answer, "What is this paper about?", 5, "bm25")
            assert waiting.wait(5)
        finally:
            release.set()
        assert first.result()["cache_status"] == "miss"
        assert second.result()["cache_status"] == "shared"
    generate.assert_called_once()
