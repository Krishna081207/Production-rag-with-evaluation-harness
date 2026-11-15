"""Shared indexed-paper query pipeline for FastAPI and Streamlit."""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

from rag_hybrid.src.generator.generator import (
    extract_claims,
    generate_answer,
    without_citation_markers,
)

if TYPE_CHECKING:
    from rag_hybrid.src.eval.faithfulness import CitationVerificationResult
    from rag_hybrid.src.retriever.hybrid import ProductionHybridRAG

PAPER_OVERVIEW_QUESTION = "What does this PDF tell me?"
PAPER_IMPLICATIONS_QUESTION = "What is this leading to?"
_OVERVIEW = "Summarize the research question, approach, and main findings of the indexed papers."
_IMPLICATIONS = (
    "What implications, applications, or future work do the authors suggest in the indexed papers? "
    "Explain how each follows from the findings and distinguish proposed work from demonstrated results."
)
_QUESTIONS = {
    "what does this pdf tell": _OVERVIEW,
    "what does this pdf tell me": _OVERVIEW,
    "what is this pdf about": _OVERVIEW,
    "what is this paper about": _OVERVIEW,
    "summarize this pdf": _OVERVIEW,
    "what is this leading to": _IMPLICATIONS,
    "what does this lead to": _IMPLICATIONS,
    "what are the implications of this paper": _IMPLICATIONS,
}
_FALLBACK_TOPICS = {
    _OVERVIEW: "abstract research objective approach methods findings conclusion",
    _IMPLICATIONS: "implications applications future",
}
_TRANSIENT_FALLBACKS = {"provider_error", "citation_validation_error"}
_CACHE_SIZE = 128
_CACHE_TTL = 15 * 60
_Key = tuple[int, str, int, str, str, bool]


class PaperQA:
    """Cache answers for one library and share the work for concurrent repeat queries."""

    def __init__(
        self,
        rag: ProductionHybridRAG,
        evaluate: Callable[
            [list[dict[str, str]], dict[str, str]], CitationVerificationResult | None
        ],
    ):
        self.rag = rag
        self._evaluate = evaluate
        self._lock = threading.Lock()
        self._revision = None
        self._cache: OrderedDict[_Key, tuple[float, dict[str, Any]]] = OrderedDict()
        self._pending: dict[_Key, Future[dict[str, Any]]] = {}

    def answer(self, query: str, top_k: int, mode: str = "hybrid_rerank") -> dict[str, Any]:
        started = time.perf_counter()
        query = query.strip()
        alias = " ".join(query.lower().rstrip("?!. ").split())
        question = _QUESTIONS.get(alias, query)
        model = os.getenv("GROQ_MODEL", "").strip()
        has_provider = bool(os.getenv("GROQ_API_KEY", "").strip() or os.getenv("groq", "").strip())

        with self._lock:
            revision = self.rag.corpus_revision
            if revision != self._revision:
                self._cache.clear()
                self._revision = revision
            key = (revision, question, top_k, mode, model, has_provider)
            cached = self._cache.pop(key, None)
            if cached is not None and cached[0] > time.perf_counter():
                self._cache[key] = cached
                return self._response(cached[1], started, "hit")

            future = self._pending.get(key)
            if future is None:
                future = self._pending[key] = Future()
                compute = True
            else:
                compute = False

        if not compute:
            return self._response(future.result(), started, "shared")

        try:
            result = self._answer_uncached(question, top_k, mode)
        except BaseException as exc:
            future.set_exception(exc)
            raise
        else:
            with self._lock:
                if (
                    self.rag.corpus_revision == revision
                    and result["fallback_reason"] not in _TRANSIENT_FALLBACKS
                ):
                    self._cache[key] = (time.perf_counter() + _CACHE_TTL, result)
                    if len(self._cache) > _CACHE_SIZE:
                        self._cache.popitem(last=False)
            future.set_result(result)
        finally:
            with self._lock:
                self._pending.pop(key)
        return self._response(result, started, "miss")

    @staticmethod
    def _response(result: dict[str, Any], started: float, cache_status: str) -> dict[str, Any]:
        timings = result["latency_ms"] if cache_status == "miss" else {}
        return {
            **result,
            "cache_status": cache_status,
            "latency_ms": {**timings, "total": round((time.perf_counter() - started) * 1000, 2)},
        }

    def _answer_uncached(self, question: str, top_k: int, mode: str) -> dict[str, Any]:
        started = time.perf_counter()
        candidate_count = max(10, top_k * 2 if mode == "hybrid_rerank" else top_k)
        chunks = self.rag.hybrid_retrieve(
            question, top_candidates=candidate_count, final_k=top_k, mode=mode
        )
        retrieved_at = time.perf_counter()
        generation = generate_answer(question, chunks, fallback_query=_FALLBACK_TOPICS.get(question))
        generated_at = time.perf_counter()

        by_citation = {f"Chunk_{chunk.chunk_id}": chunk for chunk in chunks}
        passages = {
            citation: {
                "citation": citation,
                "source_name": chunk.source_name or chunk.doc_id,
                "page_number": chunk.page_number,
            }
            for citation, chunk in by_citation.items()
        }
        evaluation = None
        if generation.mode == "generated":
            evidence = {citation: chunk.text for citation, chunk in by_citation.items()}
            evaluation = self._evaluate(extract_claims(generation.answer), evidence)
        evaluated_at = time.perf_counter()
        return {
            "answer": without_citation_markers(generation.answer),
            "answer_mode": generation.mode,
            "citations": generation.citations,
            "citation_details": [passages[citation] for citation in generation.citations],
            "retrieved_sources": [
                {**passage, "excerpt": by_citation[citation].text}
                for citation, passage in passages.items()
            ],
            "faithfulness_score": evaluation.faithfulness_score if evaluation else None,
            "violations": (evaluation.violations or None) if evaluation else None,
            "fallback_reason": generation.fallback_reason,
            "latency_ms": {
                "retrieval": round((retrieved_at - started) * 1000, 2),
                "generation": round((generated_at - retrieved_at) * 1000, 2),
                "evidence_check": round((evaluated_at - generated_at) * 1000, 2),
            },
        }
