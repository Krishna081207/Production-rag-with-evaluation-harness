# scripts/evaluate_harness.py
"""Evaluation harness for the rag_hybrid system.

This script runs a few example queries through the full RAG pipeline and
computes two key metrics:

* **Hallucination rate** – fraction of generated sentences that do *not*
  contain a citation placeholder ``[Chunk_<ID>]``.  A lower rate indicates the
  LLM respects the citation contract.
* **MMR (Maximum Marginal Relevance) score** – a simple diversity‑relevance
  measure for the retrieved chunks.  We approximate relevance by the rank order
  (higher rank => higher relevance) and diversity by the average Jaccard
  similarity of token sets between chunks.  The final score is normalized to
  ``[0, 1]`` where a higher value means more diverse and still relevant
  retrieval.

The script is deliberately lightweight and does not require any external data
files; it uses the ``ProductionHybridRAG`` class to index a tiny synthetic
corpus defined inline.  This makes it suitable for CI runs where a full data
set may not be available.
"""

import re
from typing import List, Tuple

from rag_hybrid.src.generator.generator import generate_answer
from rag_hybrid.src.retriever.hybrid import DocumentChunk, ProductionHybridRAG

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
CITATION_REGEX = re.compile(r"\[Chunk_([^\]]+)\]")


def _split_sentences(text: str) -> List[str]:
    """Very naive sentence splitter – split on period followed by space.
    Sufficient for the demo harness.
    """
    return [s.strip() for s in text.split(".") if s.strip()]


def hallucination_rate(answer: str) -> float:
    """Return the proportion of sentences *without* a citation.

    Parameters
    ----------
    answer: str
        The generated answer string.
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0
    without_citation = sum(1 for s in sentences if not CITATION_REGEX.search(s))
    return without_citation / len(sentences)


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def mmr_score(chunks: List[DocumentChunk], lambda_rel: float = 0.5) -> float:
    """Compute a simplified MMR‑like score.

    *Relevance* is approximated by the inverse of the rank (higher rank → higher
    relevance).  *Diversity* is measured by the average pairwise Jaccard
    similarity of token sets; we subtract that from 1 to get a novelty term.
    The final score is ``lambda * relevance - (1-lambda) * similarity`` and is
    scaled to ``[0, 1]``.
    """
    if not chunks:
        return 0.0
    # Relevance component – normalized sum of (1 / rank)
    relevance = sum(1.0 / (i + 1) for i in range(len(chunks)))
    relevance /= len(chunks)  # now in (0,1]

    # Diversity component – average pairwise Jaccard similarity of token sets
    token_sets = [set(c.text.lower().split()) for c in chunks]
    sims = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            sims.append(_jaccard(token_sets[i], token_sets[j]))
    if sims:
        avg_sim = sum(sims) / len(sims)
    else:
        avg_sim = 0.0
    novelty = 1.0 - avg_sim  # higher is better

    # Combine
    score = lambda_rel * relevance + (1 - lambda_rel) * novelty
    # Clamp to [0,1]
    return max(0.0, min(1.0, score))

# ---------------------------------------------------------------------------
# Simple synthetic corpus for CI – three short docs
# ---------------------------------------------------------------------------
SYNTHETIC_DOCS = [
    ("doc1", "Python is a programming language. It is dynamically typed and popular for data science."),
    ("doc2", "FastAPI is a modern web framework for building APIs with Python. It uses ASGI and type hints."),
    ("doc3", "Qdrant is a vector database that supports similarity search with cosine distance."),
]


def build_corpus() -> List[DocumentChunk]:
    rag = ProductionHybridRAG()
    all_chunks: List[DocumentChunk] = []
    for doc_id, txt in SYNTHETIC_DOCS:
        # Use a very small child chunk size for the demo
        all_chunks.extend(
            DocumentChunk(
                chunk_id=f"{doc_id}_c{i}",
                doc_id=doc_id,
                text=chunk,
                parent_text=txt,
            )
            for i, chunk in enumerate(txt.split(".")[:-1])  # drop empty last split
        )
    rag.index_documents(all_chunks)
    return all_chunks, rag


def evaluate_query(rag: ProductionHybridRAG, query: str) -> Tuple[float, float]:
    """Run a single query and return (hallucination_rate, mmr_score)."""
    retrieved = rag.hybrid_retrieve(query, top_candidates=5, final_k=3)
    gen_res = generate_answer(query, retrieved)
    h_rate = hallucination_rate(gen_res.answer)
    m_score = mmr_score(retrieved)
    return h_rate, m_score


def main():
    # Build a tiny in‑memory index – this is fast enough for CI.
    _, rag = build_corpus()

    sample_queries = [
        "What is Python?",
        "Explain FastAPI.",
        "What does Qdrant do?",
    ]

    for q in sample_queries:
        h_rate, m_score = evaluate_query(rag, q)
        print(f"Query: {q}")
        print(f"  Hallucination rate: {h_rate:.2%}")
        print(f"  MMR score: {m_score:.2%}\n")


if __name__ == "__main__":
    main()
