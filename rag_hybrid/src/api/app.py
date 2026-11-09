"""FastAPI application for the hybrid RAG pipeline."""

import hmac
import logging
import os
import re
import threading
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from rag_hybrid.src.config import max_upload_bytes
from rag_hybrid.src.corpus_store import load_corpus, save_corpus
from rag_hybrid.src.eval.faithfulness import FaithfulnessEvaluator
from rag_hybrid.src.generator.generator import ABSTENTION, generate_answer
from rag_hybrid.src.ingestion import (
    SUPPORTED_EXTENSIONS,
    chunk_document_segments,
    read_document_segments,
)
from rag_hybrid.src.retriever.hybrid import DocumentChunk, ProductionHybridRAG

logger = logging.getLogger(__name__)
app = FastAPI(title="Research Paper Understanding RAG", version="0.2.0")

_rag: ProductionHybridRAG | None = None
_evaluator: FaithfulnessEvaluator | None = None
_initialization_lock = threading.RLock()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=3000)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Query must contain non-whitespace characters")
        return stripped


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    citation_details: list[dict[str, str | int | None]]
    faithfulness_score: float | None
    violations: list[str] | None = None


class UploadResponse(BaseModel):
    files: list[str]
    chunks_indexed: int
    total_chunks: int


def _get_rag() -> ProductionHybridRAG:
    global _rag
    if _rag is None:
        with _initialization_lock:
            if _rag is None:
                rag = ProductionHybridRAG()
                saved_chunks = load_corpus()
                if saved_chunks:
                    if rag.has_vectors():
                        rag.restore_corpus(saved_chunks)
                    else:
                        # Recover a corpus file written before a vector upsert completed.
                        rag.index_documents(saved_chunks)
                _rag = rag
    return _rag


def _get_evaluator() -> FaithfulnessEvaluator:
    global _evaluator
    if _evaluator is None:
        with _initialization_lock:
            if _evaluator is None:
                _evaluator = FaithfulnessEvaluator()
    return _evaluator


def _replace_and_save(rag: ProductionHybridRAG, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Replace source versions and persist the matching sparse-corpus snapshot."""
    with _initialization_lock:
        added = rag.replace_sources(chunks)
        save_corpus(rag.corpus_chunks)
        return added


def _evaluate_claims(
    claims: list[dict[str, str]], retrieved_map: dict[str, str]
):
    """Initialize and run the NLI model off the async event loop."""
    return _get_evaluator().evaluate_generation(claims, retrieved_map)


def _extract_claims(answer: str) -> list[dict[str, str]]:
    """Create one verifiable claim per sentence and cited chunk ID."""
    if answer.strip() == ABSTENTION:
        return []

    claims: list[dict[str, str]] = []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", answer.strip()) if part.strip()]
    for sentence in sentences:
        cited_ids = re.findall(r"\[Chunk_([^\]]+)\]", sentence)
        claim = re.sub(r"\[Chunk_[^\]]+\]", "", sentence).strip()
        if not claim:
            continue
        if cited_ids:
            claims.extend({"claim": claim, "citation": f"Chunk_{chunk_id}"} for chunk_id in cited_ids)
        else:
            claims.append({"claim": claim, "citation": ""})
    return claims


def _safe_filename(filename: str) -> str:
    portable = filename.replace("\\", "/")
    name = Path(portable).name.strip()
    if not name or name in {".", ".."}:
        raise ValueError("A valid filename is required")
    return name


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness endpoint; does not load models or connect to external services."""
    return {"status": "ok"}


@app.get("/ready")
async def readiness_check() -> dict[str, str | int]:
    try:
        rag = await run_in_threadpool(_get_rag)
    except Exception as exc:
        logger.exception("RAG backend is not ready")
        raise HTTPException(status_code=503, detail="RAG backend is not ready") from exc
    return {"status": "ready", "indexed_chunks": len(rag.corpus_chunks)}


@app.post("/documents", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> UploadResponse:
    """Parse, chunk, and index uploads; ingestion requires an explicit admin key."""
    expected_key = os.getenv("ADMIN_API_KEY", "")
    if not expected_key:
        raise HTTPException(status_code=503, detail="Document uploads are disabled until ADMIN_API_KEY is set")
    if not x_admin_key or not hmac.compare_digest(x_admin_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    if len(files) > 10:
        raise HTTPException(status_code=413, detail="A maximum of 10 files may be uploaded at once")

    try:
        file_limit = max_upload_bytes()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="MAX_UPLOAD_BYTES is invalid") from exc

    indexed_files: list[str] = []
    chunks: list[DocumentChunk] = []
    total_bytes = 0
    for upload in files:
        try:
            filename = _safe_filename(upload.filename or "")
            if filename in indexed_files:
                raise HTTPException(status_code=400, detail=f"Duplicate filename in upload: {filename}")
            suffix = Path(filename).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
                raise HTTPException(status_code=400, detail=f"{filename}: supported types are {supported}")

            content = await upload.read(file_limit + 1)
            if len(content) > file_limit:
                raise HTTPException(status_code=413, detail=f"{filename}: file exceeds the upload limit")
            total_bytes += len(content)
            if total_bytes > file_limit * 2:
                raise HTTPException(status_code=413, detail="Combined upload exceeds the request limit")

            _, segments = await run_in_threadpool(read_document_segments, filename, content)
            document_chunks = chunk_document_segments(filename, segments)
            if not document_chunks:
                raise ValueError("Document contains no indexable text")
            indexed_files.append(filename)
            chunks.extend(document_chunks)
        except HTTPException:
            raise
        except (ValueError, RuntimeError, ImportError) as exc:
            raise HTTPException(status_code=400, detail=f"{upload.filename or 'upload'}: {exc}") from exc
        finally:
            await upload.close()

    try:
        rag = await run_in_threadpool(_get_rag)
        added = await run_in_threadpool(_replace_and_save, rag, chunks)
    except Exception as exc:
        logger.exception("Document indexing failed")
        raise HTTPException(status_code=503, detail="Document indexing failed; check backend configuration") from exc

    return UploadResponse(
        files=indexed_files,
        chunks_indexed=len(added),
        total_chunks=len(rag.corpus_chunks),
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    try:
        rag = await run_in_threadpool(_get_rag)
        retrieved_chunks = await run_in_threadpool(
            rag.hybrid_retrieve,
            req.query,
            top_candidates=20,
            final_k=5,
        )
        generation = await run_in_threadpool(generate_answer, req.query, retrieved_chunks)
        claims = _extract_claims(generation.answer)
        retrieved_map = {
            key: chunk.parent_text
            for chunk in retrieved_chunks
            for key in (chunk.chunk_id, f"Chunk_{chunk.chunk_id}")
        }
        if claims:
            evaluation = await run_in_threadpool(
                _evaluate_claims,
                claims,
                retrieved_map,
            )
        else:
            evaluation = None
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=503, detail="The RAG service could not complete the query") from exc

    chunks_by_citation = {f"Chunk_{chunk.chunk_id}": chunk for chunk in retrieved_chunks}
    citation_details = []
    for citation in generation.citations:
        chunk = chunks_by_citation.get(citation)
        if chunk:
            citation_details.append(
                {
                    "citation": citation,
                    "source_name": chunk.source_name or chunk.doc_id,
                    "page_number": chunk.page_number,
                }
            )

    return QueryResponse(
        answer=generation.answer,
        citations=generation.citations,
        citation_details=citation_details,
        faithfulness_score=evaluation.faithfulness_score if evaluation else None,
        violations=evaluation.violations or None if evaluation else None,
    )
