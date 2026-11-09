"""Run a tiny local demo and report citation coverage and lexical diversity.

These format and diversity proxies do not measure factual hallucination or
answer quality. Use rag_hybrid.eval.run_eval with a verified gold set for
retrieval metrics.
"""

import re
from typing import List, Tuple

from rag_hybrid.src.generator.generator import generate_answer
from rag_hybrid.src.retriever.hybrid import DocumentChunk, ProductionHybridRAG

CITATION_REGEX = re.compile(r"\[Chunk_[^\]]+\]")
ABSTENTION = "The provided documentation does not contain sufficient evidence to answer this inquiry."


def _split_sentences(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def uncited_sentence_rate(answer: str) -> float:
    sentences = _split_sentences(answer)
    if not sentences or answer.strip() == ABSTENTION:
        return 0.0
    uncited = sum(1 for sentence in sentences if not CITATION_REGEX.search(sentence))
    return uncited / len(sentences)


def hallucination_rate(answer: str) -> float:
    """Compatibility alias; missing citations are not proof of hallucination."""
    return uncited_sentence_rate(answer)


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def retrieval_diversity_proxy(chunks: List[DocumentChunk], lambda_rel: float = 0.5) -> float:
    """Combine rank position with lexical diversity; this is not true MMR."""
    if not chunks:
        return 0.0
    if not 0.0 <= lambda_rel <= 1.0:
        raise ValueError("lambda_rel must be between 0 and 1")
    relevance = sum(1.0 / (rank + 1) for rank in range(len(chunks))) / len(chunks)
    token_sets = [set(chunk.text.lower().split()) for chunk in chunks]
    similarities = [
        _jaccard(token_sets[i], token_sets[j])
        for i in range(len(token_sets))
        for j in range(i + 1, len(token_sets))
    ]
    novelty = 1.0 - (sum(similarities) / len(similarities) if similarities else 0.0)
    return max(0.0, min(1.0, lambda_rel * relevance + (1.0 - lambda_rel) * novelty))


def mmr_score(chunks: List[DocumentChunk], lambda_rel: float = 0.5) -> float:
    """Compatibility alias for the lexical diversity proxy."""
    return retrieval_diversity_proxy(chunks, lambda_rel=lambda_rel)


SYNTHETIC_DOCS = [
    (
        "retrieval_paper",
        "This fictional paper asks whether hybrid retrieval improves research evidence search. "
        "It compares BM25, dense retrieval, and reciprocal rank fusion on a small synthetic corpus.",
    ),
    (
        "methods_paper",
        "This fictional methods paper describes a study with manually reviewed questions and "
        "Recall at 5 and mean reciprocal rank as retrieval metrics.",
    ),
    (
        "limitations_paper",
        "This fictional paper notes that a small single-domain corpus limits generalization. "
        "It also says automated evidence checks do not replace domain expert review.",
    ),
]


def build_corpus() -> Tuple[List[DocumentChunk], ProductionHybridRAG]:
    rag = ProductionHybridRAG(vector_db="qdrant", qdrant_path=":memory:")
    all_chunks: List[DocumentChunk] = []
    for doc_id, text in SYNTHETIC_DOCS:
        sentences = [sentence.strip() for sentence in text.split(".") if sentence.strip()]
        all_chunks.extend(
            DocumentChunk(
                chunk_id=f"{doc_id}_c{index}",
                doc_id=doc_id,
                text=sentence,
                parent_text=text,
            )
            for index, sentence in enumerate(sentences)
        )
    rag.index_documents(all_chunks)
    return all_chunks, rag


def evaluate_query(rag: ProductionHybridRAG, query: str) -> Tuple[float, float]:
    retrieved = rag.hybrid_retrieve(query, top_candidates=5, final_k=3)
    generated = generate_answer(query, retrieved)
    return uncited_sentence_rate(generated.answer), retrieval_diversity_proxy(retrieved)


def main() -> None:
    _, rag = build_corpus()
    for question in (
        "What research question does the retrieval paper ask?",
        "Which metrics are used in the methods paper?",
        "What limitations do the fictional papers report?",
    ):
        uncited_rate, diversity = evaluate_query(rag, question)
        print(f"Question: {question}")
        print(f"  Uncited-sentence rate (format proxy): {uncited_rate:.2%}")
        print(f"  Retrieval-diversity proxy: {diversity:.2%}\n")


if __name__ == "__main__":
    main()
