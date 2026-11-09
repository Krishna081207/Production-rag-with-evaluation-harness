"""Run deterministic retrieval evaluations from a JSONL gold set.

The harness deliberately keeps generation optional: retrieval metrics can be
run without an LLM, which makes the ablation matrix reproducible on a laptop.
"""

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any

from rag_hybrid.eval.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize_latencies,
)
from rag_hybrid.src.retriever.hybrid import DocumentChunk, ProductionHybridRAG

CONFIGS = {
    "C1": "dense",
    "C2": "bm25",
    "C3": "hybrid",
    "C4": "hybrid_rerank",
}


def load_gold(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("question"), str)
                or not row["question"].strip()
            ):
                raise ValueError(f"Gold row {line_number} must be an object with a question")
            gold_chunks = row.get("gold_chunks")
            if (
                not isinstance(gold_chunks, list)
                or not gold_chunks
                or any(not isinstance(chunk_id, str) or not chunk_id.strip() for chunk_id in gold_chunks)
            ):
                raise ValueError(
                    f"Gold row {line_number} must include a non-empty list of chunk IDs"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"Gold set is empty: {path}")
    return rows


def build_retriever(chunks: list[DocumentChunk]) -> ProductionHybridRAG:
    rag = ProductionHybridRAG(
        vector_db="qdrant",
        collection_name=f"eval-{uuid.uuid4().hex[:8]}",
        qdrant_path=":memory:",
    )
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
    # Warm model initialization is excluded from per-query latency.
    rag.retrieve(gold[0]["question"], mode=mode, k=k)
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
        f"recall@{k}": round(sum(recalls) / count, 4),
        f"hit_rate@{k}": round(sum(hits) / count, 4),
        "mrr": round(sum(mrrs) / count, 4),
        f"ndcg@{k}": round(sum(ndcgs) / count, 4),
        "latency_ms": summarize_latencies(latencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate C1-C4 retrieval configurations.")
    parser.add_argument("--gold", type=Path, required=True, help="Gold JSONL file.")
    parser.add_argument(
        "--chunks", type=Path, required=True, help="JSONL file with chunk_id, doc_id, text, parent_text."
    )
    parser.add_argument("--k", type=int, default=5, help="Number of final chunks to score.")
    parser.add_argument("--output", type=Path, help="Optional Markdown report path.")
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be a positive integer")
    chunks: list[DocumentChunk] = []
    with args.chunks.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunks.append(DocumentChunk.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid chunk record in {args.chunks}:{line_number}") from exc
    if not chunks:
        parser.error("--chunks must contain at least one chunk")
    gold = load_gold(args.gold)
    available_chunk_ids = {chunk.chunk_id for chunk in chunks}
    unknown_gold_ids = sorted(
        {chunk_id for row in gold for chunk_id in row["gold_chunks"]} - available_chunk_ids
    )
    if unknown_gold_ids:
        parser.error(f"Gold set references chunk IDs missing from --chunks: {', '.join(unknown_gold_ids)}")
    rag = build_retriever(chunks)
    lines = [
        f"| Config | Questions | Recall@{args.k} | Hit rate@{args.k} | MRR | nDCG@{args.k} | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config, mode in CONFIGS.items():
        result = run_config(rag, gold, mode, k=args.k)
        latency = result["latency_ms"]
        lines.append(
            f"| {config} | {result['questions']} | {result[f'recall@{args.k}']:.3f} | "
            f"{result[f'hit_rate@{args.k}']:.3f} | {result['mrr']:.3f} | "
            f"{result[f'ndcg@{args.k}']:.3f} | "
            f"{latency['p50']:.1f} | {latency['p95']:.1f} |"
        )
    report = "\n".join(lines) + "\n"
    print(report, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
