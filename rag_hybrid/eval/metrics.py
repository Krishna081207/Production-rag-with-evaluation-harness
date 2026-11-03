"""Deterministic metrics for retrieval and answer evaluation."""

import re
from collections.abc import Iterable, Sequence
from math import log2
from typing import Any


def normalize_text(value: str) -> str:
    """Normalize text for stable phrase and citation comparisons."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", value.lower())).strip()


def key_fact_match(answer: str, key_facts: Iterable[str]) -> bool:
    """Return whether every required fact appears in the generated answer."""
    normalized = normalize_text(answer)
    facts = [normalize_text(fact) for fact in key_facts if normalize_text(fact)]
    return bool(facts) and all(fact in normalized for fact in facts)


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 5) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def hit_rate_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 5) -> float:
    relevant = set(relevant_ids)
    return float(bool(relevant and set(retrieved_ids[:k]) & relevant))


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 5) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / log2(rank + 1)
        for rank, item_id in enumerate(retrieved_ids[:k], start=1)
        if item_id in relevant
    )
    ideal = sum(1.0 / log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0


def citation_match(citations: Iterable[str], gold_sources: Iterable[dict[str, Any]]) -> bool:
    """Match cited document/page pairs against the gold source set."""
    expected = {(str(source.get("paper")), source.get("page")) for source in gold_sources}
    actual = set()
    for citation in citations:
        parts = str(citation).split(":")
        if len(parts) == 2:
            actual.add((parts[0], int(parts[1]) if parts[1].isdigit() else parts[1]))
        else:
            actual.add((str(citation), None))
    return bool(expected) and bool(actual & expected)


def abstention_accuracy(answerable: bool, answer: str) -> float:
    refused = "does not contain sufficient evidence" in normalize_text(answer)
    return float(refused == (not answerable))


def summarize_latencies(latencies_ms: Sequence[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0}
    values = sorted(latencies_ms)

    def percentile(fraction: float) -> float:
        index = min(len(values) - 1, round((len(values) - 1) * fraction))
        return round(float(values[index]), 3)

    return {"p50": percentile(0.50), "p95": percentile(0.95)}


def retrieval_lift(full_rag_matches: Sequence[bool], llm_only_matches: Sequence[bool]) -> float:
    """Compute full-RAG key-fact accuracy minus the LLM-only accuracy."""
    if len(full_rag_matches) != len(llm_only_matches):
        raise ValueError("Both runs must contain the same number of questions")
    if not full_rag_matches:
        return 0.0
    return sum(full_rag_matches) / len(full_rag_matches) - sum(llm_only_matches) / len(
        llm_only_matches
    )
