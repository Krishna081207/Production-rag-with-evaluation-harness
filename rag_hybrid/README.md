# rag_hybrid — Production RAG with Evaluation Harness

> **Hybrid Dense (Pinecone / Qdrant) + BM25 retrieval → RRF fusion → Cross-Encoder reranking → Groq LLM → NLI faithfulness evaluation**

[![CI](https://github.com/yourorg/rag_hybrid/actions/workflows/ci.yml/badge.svg)](https://github.com/yourorg/rag_hybrid/actions/workflows/ci.yml)

---

## Architecture

```
Documents
   │
   ▼
scripts/ingest.py          # chunk → embed → index
   │
   ├── Dense Vector DB: Pinecone (Cloud Serverless) OR Qdrant (BGE-small-en-v1.5)
   └── Sparse BM25 index (rank-bm25)
         │
         ▼
   RRF fusion (k=60)
         │
         ▼
   Cross-Encoder reranker  (ms-marco-MiniLM-L-6-v2)
         │
         ▼
   Groq LLM (llama3-70b)  ← citation contract [Chunk_<ID>]
         │
         ▼
   NLI faithfulness check  (nli-deberta-v3-small)
         │
         ▼
   FastAPI /query  +  Streamlit UI
```

---

## Quickstart

```bash
# 1. Clone & install
git clone https://github.com/yourorg/rag_hybrid
cd rag_hybrid
pip install -e ".[dev]"

# 2. Configure environment (.env)
PINECONE_API_KEY=your_pinecone_api_key   # or db_key
VECTOR_DB=pinecone                       # 'pinecone' or 'qdrant'
PINECONE_INDEX_NAME=rag-hybrid
GROQ_API_KEY=your_groq_api_key

# 3. Index your documents (drop .txt / .md / .pdf files into data/)
# For Pinecone:
python scripts/ingest.py --vector-db pinecone --source data/

# For Qdrant (in-memory):
python scripts/ingest.py --vector-db qdrant --source data/

# 4a. Run the FastAPI backend
uvicorn rag_hybrid.src.api.app:app --reload

# 4b. Run the Streamlit UI (in a second terminal)
streamlit run scripts/app_streamlit.py
```

### Docker Compose (all services)

```bash
docker compose up --build
```

| Service | Port |
|---|---|
| FastAPI API | http://localhost:8000 |
| Streamlit UI | http://localhost:8501 |

---

## Vector Database Backends

`rag_hybrid` supports pluggable vector databases:

| Feature | Pinecone | Qdrant |
|---|---|---|
| **Deployment** | Managed Serverless Cloud | Local / In-memory / Self-hosted |
| **Dimension** | 384 (cosine) | 384 (cosine) |
| **Index Auto-creation** | ✅ Yes (`ServerlessSpec`) | ✅ Yes |
| **Metadata payload** | Chunk & parent context | Chunk & parent context |
| **Configuration** | `VECTOR_DB=pinecone` | `VECTOR_DB=qdrant` |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Faithfulness score** | NLI entailment probability of each cited claim vs. its source chunk |
| **Hallucination rate** | Fraction of generated sentences without a `[Chunk_<ID>]` citation |
| **MMR score** | λ·relevance + (1−λ)·novelty — measures diversity of retrieved chunks |

Run the harness manually:

```bash
python scripts/evaluate_harness.py
```

---

## CI/CD Pipeline

Four jobs run on every push to `main`/`develop` and all PRs:

1. **Lint** — `ruff check`
2. **Test** — `pytest` on Python 3.10 + 3.11 with coverage upload
3. **Evaluate** — evaluation harness with a **hallucination gate (≤ 50%)**
4. **Docker build** — smoke-tests the image builds cleanly

### Required GitHub Secrets

| Secret | Required | Purpose |
|---|---|---|
| `PINECONE_API_KEY` | Optional | Pinecone cloud vector DB |
| `GROQ_API_KEY` | Optional | Real LLM calls in CI (tests use mocks if absent) |
| `CODECOV_TOKEN` | Optional | Coverage badge |

---

## Environment Variables

| Variable | Description |
|---|---|
| `PINECONE_API_KEY` / `db_key` | Pinecone API key |
| `VECTOR_DB` | Vector backend: `pinecone` or `qdrant` (default: `qdrant`) |
| `PINECONE_INDEX_NAME` | Pinecone index name (default: `rag-hybrid`) |
| `GROQ_API_KEY` | Groq API key |
| `RAG_API_URL` | Streamlit → API URL (default: `http://localhost:8000`) |
