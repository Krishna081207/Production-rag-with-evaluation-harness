"""FastAPI application exposing the RAG pipeline.

Endpoints
---------
GET /health
    Simple health check.
POST /query
    Body: {"query": "..."}
    Returns a JSON with the generated answer, the list of citations, and a faithfulness score.
"""

import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Local imports – the repository root is added to ``PYTHONPATH`` when the
# package is installed in editable mode (``pip install -e .``).  Therefore we
# can import using the top‑level ``rag_hybrid`` package name.
from rag_hybrid.src.retriever.hybrid import ProductionHybridRAG, DocumentChunk
from rag_hybrid.src.generator.generator import generate_answer
from rag_hybrid.src.eval.faithfulness import FaithfulnessEvaluator

app = FastAPI(title="rag_hybrid API", version="0.1.0")

# Initialise a single global RAG instance – in production you would manage a
# connection pool, handle re‑indexing, etc.
_rag = ProductionHybridRAG()
_evaluator = FaithfulnessEvaluator()

# ---------------------------------------------------------------------
#   Request / response models
# ---------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    faithfulness_score: float
    violations: list[str] | None = None

# ---------------------------------------------------------------------
#   Helper utilities
# ---------------------------------------------------------------------
_CITATION_RE = re.compile(r"\[Chunk_(?P<cid>[^\]]+)\]")

def _extract_claims(answer: str) -> list[dict[str, str]]:
    """Very naive claim extraction.

    We treat the whole answer as a single claim and associate it with the
    first citation that appears in the text.  This is sufficient for the demo
    and matches the contract described in the project specification.
    """
    match = _CITATION_RE.search(answer)
    citation = f"Chunk_{match.group('cid')}" if match else ""
    claim = answer.replace(match.group(0), "").strip() if match else answer.strip()
    return [{"claim": claim, "citation": citation}]

# ---------------------------------------------------------------------
#   Endpoints
# ---------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Query string empty")

    # Retrieve top‑5 chunks (adjustable)
    retrieved_chunks = _rag.hybrid_retrieve(req.query, top_candidates=20, final_k=5)

    # Generation step – deterministic stub for now
    gen_result = generate_answer(req.query, retrieved_chunks)

    # Build mapping from chunk_id -> parent_text for evaluation
    retrieved_map = {c.chunk_id: c.parent_text for c in retrieved_chunks}

    # Evaluate faithfulness
    claims = _extract_claims(gen_result.answer)
    eval_res = _evaluator.evaluate_generation(claims, retrieved_map)

    return QueryResponse(
        answer=gen_result.answer,
        citations=gen_result.citations,
        faithfulness_score=eval_res.faithfulness_score,
        violations=eval_res.violations if eval_res.violations else None,
    )
