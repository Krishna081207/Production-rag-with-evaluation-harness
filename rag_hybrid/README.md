# Retrieval‑Augmented Generation (Hybrid) Project

A production‑ready RAG system that combines **dense vector search** (Qdrant/pgvector) with **BM25 sparse retrieval**, a **cross‑encoder reranker**, and a **faithfulness evaluation harness**.

## Overview
- **Hybrid retrieval** – dense ANN + BM25 with linear interpolation.
- **Reranking** – cross‑encoder (MiniLM‑MarCo) for top‑N passages.
- **Chunking** – 300‑word overlapping windows, citation IDs.
- **LLM generator** – OpenAI/Anthropic API with in‑line citations.
- **Eval harness** – RAGAS, faithfulness, recall@k, latency.

## Directory Layout
```
rag_hybrid/
├─ data/           # raw documents (PDF, txt …)
├─ chunks/         # pre‑processed passages (jsonl)
├─ embeddings/     # optional dense vectors
├─ scripts/        # CLI utilities (ingest, retrieve, generate, evaluate)
├─ src/            # core library
│   ├─ retriever/
│   ├─ reranker/
│   ├─ generator/
│   ├─ eval/
│   └─ api/
├─ docker-compose.yml
├─ pyproject.toml
└─ README.md
```

## Getting Started
```bash
# Build services
docker compose up -d

# Install Python package (editable)
pip install -e .

# Index your corpus
python scripts/ingest.py --source data/
```

---
*For detailed component docs, see the `src/` modules.*
