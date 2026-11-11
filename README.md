# Research Paper Understanding Assistant

A paper-reading assistant built on hybrid retrieval and an evaluation harness. Upload a research paper, ask about its research question, contributions, methods, results, or limitations, and inspect the retrieved source and PDF page behind each answer.

The project combines dense retrieval, BM25, reciprocal rank fusion, cross-encoder reranking, cited answer generation, and NLI-based evidence checks. The included evaluation data is synthetic smoke data; it demonstrates the harness and does not measure research-paper answer quality.

## How it works

```text
PDF / DOCX / Markdown / text
        │
        ├── extract text (PDF page numbers are preserved)
        ├── create child chunks with parent context
        └── embed + persist in Qdrant or Pinecone
                    │
             Dense + BM25 retrieval
                    │
                 RRF fusion
                    │
          Cross-encoder reranking
                    │
       Grounded answer with chunk citations
                    │
   NLI evidence support check + source/page details
```

## Quick start

Requires Python 3.10 or 3.11. On Windows, create and activate a virtual environment, then install the project from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Edit `.env` and set a long random `ADMIN_API_KEY` to enable uploads. Set `GROQ_API_KEY` for synthesized answers; without it, the application returns up to three query-relevant cited paper sentences and the chat exposes the retrieved passage text with its chunk and page reference. The default local setup uses Qdrant at `rag_hybrid/data/qdrant` and saves the sparse corpus under `rag_hybrid/data/`.

The chat UI can search either your indexed papers or the public web. For web search, set `TAVILY_API_KEY` in `.env`; web queries are sent to Tavily only when **Web search** is selected. The [Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search) returns an answer with linked source snippets. Retrieval models load in the background after API startup, and the UI shows retrieval, generation, and evidence-check timing for paper answers.

For local testing, start both services from one terminal:

```powershell
python -m rag_hybrid.scripts.run_local
```

The launcher waits for FastAPI readiness before starting Streamlit and prints startup errors in the same terminal. Open <http://127.0.0.1:8501>. Press Ctrl+C to stop both services. To run them separately instead, start FastAPI with `python -m uvicorn rag_hybrid.src.api.app:app --reload` and Streamlit with `python -m streamlit run rag_hybrid/scripts/app_streamlit.py` in separate terminals.

Open <http://localhost:8501>, upload a paper, and ask a question such as:

- What problem does this paper address, and what are its contributions?
- How was the study designed? What baselines or datasets were used?
- What are the main findings, and which measurements support them?
- What limitations do the authors acknowledge?

PDF citations include page numbers. DOCX, Markdown, and text sources are cited by file and chunk; these formats do not provide stable page locations.

## Ingest from a directory

Put papers in `rag_hybrid/data/papers/` or provide another source directory. The CLI accepts PDF, DOCX, Markdown, and UTF-8 text files. Re-running ingestion is idempotent for unchanged files, replaces an updated file at the same path, and preserves other sources. Give paper versions distinct filenames when you want both indexed.

```powershell
python rag_hybrid/scripts/ingest.py --source rag_hybrid/data/papers
```

For Pinecone, install the optional backend and configure `VECTOR_DB=pinecone`, `PINECONE_API_KEY`, `PINECONE_CLOUD`, and `PINECONE_REGION` in `.env`:

```powershell
python -m pip install -e ".[pinecone,dev]"
python rag_hybrid/scripts/ingest.py --vector-db pinecone --source rag_hybrid/data/papers
```

## API

`GET /health` is a liveness check. `GET /ready` reports indexed chunk/source counts and model warmup status. Uploads require the configured admin key. If the same unchanged paper is uploaded again, the UI reports that its chunks are already indexed; scanned PDFs need OCR before upload:

```powershell
curl.exe -X POST http://localhost:8000/documents `
  -H "X-Admin-Key: replace-with-your-admin-key" `
  -F "files=@rag_hybrid/data/papers/example.pdf"
```

Ask a question with `POST /query`:

```powershell
curl.exe -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  -d '{"query":"What are the paper’s main contributions?"}'
```

The response includes the answer, chunk citations, file/page details, an evidence-support score, and any unsupported-claim notes. The NLI score is an automated estimate and should be checked against the cited passage for consequential use.

Optional public web search is available at `POST /web-search` when `TAVILY_API_KEY` is configured. It returns an answer and a list of source titles, URLs, and snippets. `/ready` also reports whether background retrieval-model warmup has completed.

`GET /documents` lists indexed source names and chunk counts. The Streamlit sidebar can remove selected papers or clear the full library. Both operations update Qdrant/Pinecone and the saved BM25 corpus; deletion requires `X-Admin-Key`. The clear-all action requires an explicit confirmation in the UI.

## Evaluation

Run the retrieval ablation harness:

```powershell
python -m rag_hybrid.eval.run_eval `
  --chunks rag_hybrid/eval/data/sample_chunks.jsonl `
  --gold rag_hybrid/eval/data/sample_gold.jsonl
```

The report compares dense retrieval, BM25, hybrid RRF, and hybrid RRF with reranking using Recall@k, hit rate@k, MRR, nDCG@k, and latency. Supply a JSONL chunk corpus and a manually reviewed gold file for meaningful results. Gold rows use `{"question":"...","gold_chunks":["chunk-id"]}`; the harness rejects unknown chunk IDs. The checked-in fictional paper excerpts are only a wiring smoke test.

Run the local citation-format and diversity demo with:

```powershell
python rag_hybrid/scripts/evaluate_harness.py
```

Its citation and diversity metrics are format/lexical proxies, not factuality or true MMR scores.

## Docker Compose

Set `.env` values first, especially `ADMIN_API_KEY`, then start both services:

```powershell
docker compose up --build
```

| Service | Address |
|---|---|
| Streamlit | <http://localhost:8501> |
| FastAPI | <http://localhost:8000> |
| API docs | <http://localhost:8000/docs> |

Compose persists the local Qdrant database and corpus in the `rag_data` volume. For a shared or multi-replica deployment, configure a remote Qdrant service or Pinecone. Put the API and UI behind your organization’s access control and TLS before exposing them beyond a trusted network; the admin key protects ingestion, while query access is otherwise open.

## Production deployment (Render)

The included [`render.yaml`](render.yaml) creates two services in Singapore: a public Streamlit web service and a private FastAPI service. The API stores Qdrant, the BM25 corpus, and the embedding-model cache on a 10 GB persistent disk. Only the Streamlit service is reachable from the public internet. The UI requires Streamlit OIDC sign-in and an explicit email allowlist before it exposes any RAG features; until those values are configured, it shows a locked setup page. Streamlit OIDC supports providers such as Google, Microsoft Entra ID, Okta, and Auth0 ([Streamlit authentication](https://docs.streamlit.io/develop/concepts/connections/authentication)).

Create a Render Blueprint from this repository and deploy `render.yaml`. Render persistent disks require a paid service plan ([persistent disk limits](https://render.com/docs/disks)). After the first deploy:

1. Copy the Streamlit service URL from Render. In your identity provider, register `https://<your-streamlit-host>/oauth2callback` as an allowed redirect URI. For a custom domain, use the custom-domain callback URI and set `OIDC_REDIRECT_URI` on the Streamlit service to that exact URL.
2. On the Streamlit service, set `OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET`, and `AUTHORIZED_EMAILS`. Enter authorized email addresses as a comma-separated list. The Blueprint generates `OIDC_COOKIE_SECRET` and the API's `ADMIN_API_KEY` for you. Google is the default identity provider; set `OIDC_SERVER_METADATA_URL` on the Streamlit service for another OIDC provider.
3. Set `GROQ_API_KEY` and `TAVILY_API_KEY` on the private API service if you want generated answers and web search. Without them, paper queries use the cited extractive fallback and web search is unavailable.
4. Redeploy the Streamlit service after setting authentication values, then upload your papers through the app. Local Qdrant and paper data are not copied by deployment; the new hosted library starts empty.

Every allowlisted account uses the same paper library. This layout runs one API instance because Render disks cannot be shared or horizontally scaled; the disk provides persistence and daily snapshots, but does not provide high availability or a database backup strategy ([disk behavior](https://render.com/docs/disks), [scaling limits](https://render.com/docs/blueprint-spec#scaling)). For larger or higher-availability production use, move vectors to a managed Qdrant cluster and move the BM25 corpus to shared durable storage before scaling the API. Review the service plan and disk costs in Render before creating the Blueprint.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `VECTOR_DB` | `qdrant` or `pinecone` | `qdrant` |
| `QDRANT_PATH` | Local persistent Qdrant path; `:memory:` disables persistence | unset (in-memory) |
| `QDRANT_URL` | Remote Qdrant URL, when used instead of a local path | unset |
| `QDRANT_COLLECTION_NAME` | Local or remote Qdrant collection | `rag-hybrid` |
| `RAG_DATA_DIR` | Directory for the saved BM25 corpus | `rag_hybrid/data` |
| `EMBEDDING_MODEL` | Sentence-transformer embedding model (must output 384 dimensions) | `BAAI/bge-small-en-v1.5` |
| `RERANKER_MODEL` | Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `GROQ_API_KEY` | Optional key for generated answers | unset (extractive fallback) |
| `GROQ_MODEL` | Groq model ID | `openai/gpt-oss-20b` |
| `TAVILY_API_KEY` | Optional Tavily key for web search mode | unset |
| `ADMIN_API_KEY` | Required key for API document uploads | unset (uploads disabled) |
| `MAX_UPLOAD_BYTES` | Maximum size per file; combined request limit is twice this | `10485760` |
| `PINECONE_API_KEY` | Optional Pinecone key | unset |
| `PINECONE_INDEX_NAME` | Pinecone index name | `rag-hybrid` |
| `PINECONE_CLOUD` / `PINECONE_REGION` | Pinecone serverless location | `aws` / `us-east-1` |
| `RAG_API_URL` | API base URL used by Streamlit | `http://localhost:8000` |

## Repository layout

```text
.github/workflows/ci.yml       GitHub Actions workflow
rag_hybrid/src/api/            FastAPI service
rag_hybrid/src/ingestion.py    File parsing and page-aware chunking
rag_hybrid/src/retriever/      Dense + BM25 retrieval and reranking
rag_hybrid/src/generator/      Grounded answer generation
rag_hybrid/src/eval/           Citation evidence checks
rag_hybrid/eval/               Retrieval metrics, smoke corpus, and harness
rag_hybrid/scripts/            Ingestion and Streamlit commands
```

Scanned PDFs need OCR before ingestion. Tables and figures in PDFs may not be captured by text extraction, so verify the cited page in the original paper. Research-paper quality depends on domain-relevant chunking, a human-reviewed evaluation set, and manual review of cited evidence.
