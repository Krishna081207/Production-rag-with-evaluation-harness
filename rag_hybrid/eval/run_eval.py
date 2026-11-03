"""Run deterministic retrieval evaluations from a JSONL gold set.

The harness deliberately keeps generation optional: retrieval metrics can be
run without an LLM, which makes the ablation matrix reproducible on a laptop.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

try:
    from rag_hybrid.src.retriever.hybrid import DocumentChunk, ProductionHybridRAG
except ModuleNotFoundError:  # Supports running from an unpackaged repository checkout.
    from src.retriever.hybrid import DocumentChunk, ProductionHybridRAG

try:
    from .metrics import hit_rate_at_k, ndcg_at_k, recall_at_k, reciprocal_rank, summarize_latencies
except ImportError:  # Supports `python eval/run_eval.py` from the repository root.
    from metrics import hit_rate_at_k, ndcg_at_k, recall_at_k, reciprocal_rank, summarize_latencies

CONFIGS = {
    "C1": "dense",
    "C2": "bm25",
    "C3": "hybrid",
    "C4": "hybrid_rerank",
}


def load_gold(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_retriever(chunks: list[DocumentChunk]) -> ProductionHybridRAG:
    rag = ProductionHybridRAG()
    rag.index_documents(chunks)
    return rag


def run_config(
    rag: ProductionHybridRAG, gold: list[dict[str, Any]], mode: str, k: int = 5
) -> dict[str, Any]:
    recalls: list[float] = []
    hits: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    latencies: list[float] = []
    for row in gold:
        start = time.perf_counter()
        results = rag.retrieve(row["question"], mode=mode, k=k)
        latencies.append((time.perf_counter() - start) * 1000)
        retrieved = [chunk.chunk_id for chunk in results]
        relevant = row.get("gold_chunks", [])
        recalls.append(recall_at_k(retrieved, relevant, k))
        hits.append(hit_rate_at_k(retrieved, relevant, k))
        mrrs.append(reciprocal_rank(retrieved, relevant))
        ndcgs.append(ndcg_at_k(retrieved, relevant, k))

    count = len(gold) or 1
    return {
        "questions": len(gold),
        "recall@5": round(sum(recalls) / count, 4),
        "hit_rate@5": round(sum(hits) / count, 4),
        "mrr": round(sum(mrrs) / count, 4),
        "ndcg@5": round(sum(ndcgs) / count, 4),
        "latency_ms": summarize_latencies(latencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate C1-C4 retrieval configurations.")
    parser.add_argument("--gold", type=Path, required=True, help="Gold JSONL file.")
    parser.add_argument(
        "--chunks", type=Path, required=True, help="JSONL file with chunk_id, doc_id, text, parent_text."
    )
    args = parser.parse_args()
    chunks = [DocumentChunk.model_validate(json.loads(line)) for line in args.chunks.open(encoding="utf-8")]
    gold = load_gold(args.gold)
    rag = build_retriever(chunks)
    print("| Config | Questions | Recall@5 | Hit rate@5 | MRR | nDCG@5 | p50 ms | p95 ms |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for config, mode in CONFIGS.items():
        result = run_config(rag, gold, mode)
        latency = result["latency_ms"]
        print(
            f"| {config} | {result['questions']} | {result['recall@5']:.3f} | "
            f"{result['hit_rate@5']:.3f} | {result['mrr']:.3f} | {result['ndcg@5']:.3f} | "
            f"{latency['p50']:.1f} | {latency['p95']:.1f} |"
        )


if __name__ == "__main__":
    main()
