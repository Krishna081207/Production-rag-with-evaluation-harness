"""Hybrid dense and BM25 retrieval with RRF fusion and cross-encoder reranking."""

from __future__ import annotations

import os
import re
import threading
import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder as CrossEncoderType
    from sentence_transformers import SentenceTransformer as SentenceTransformerType

# Kept as patchable sentinels so imports stay light until a model is first used.
CrossEncoder: Any = None
SentenceTransformer: Any = None

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
EMBEDDING_DIMENSION = 384
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class DocumentChunk(BaseModel):
    """A retrievable child chunk and its broader generation context."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    text: str
    parent_text: str
    source_name: str | None = None
    page_number: int | None = Field(default=None, ge=1)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


class ProductionHybridRAG:
    """Hybrid retriever backed by Qdrant or Pinecone.

    Qdrant uses a remote server when ``QDRANT_URL`` is set, persistent local
    storage when ``QDRANT_PATH`` is set, and in-memory storage otherwise.
    Model weights are loaded only when indexing or retrieval needs them.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        vector_db: str | None = None,
        pinecone_api_key: str | None = None,
        pinecone_cloud: str | None = None,
        pinecone_region: str | None = None,
        qdrant_url: str | None = None,
        qdrant_path: str | None = None,
        embedding_model: str | None = None,
        reranker_model: str | None = None,
    ):
        self.vector_db = (vector_db or os.getenv("VECTOR_DB", "qdrant")).strip().lower()
        if self.vector_db not in {"qdrant", "pinecone"}:
            raise ValueError("VECTOR_DB must be either 'qdrant' or 'pinecone'")

        self.collection_name = collection_name or (
            os.getenv("PINECONE_INDEX_NAME", "rag-hybrid")
            if self.vector_db == "pinecone"
            else os.getenv("QDRANT_COLLECTION_NAME", "rag-hybrid")
        )
        self.embedding_model_name = embedding_model or EMBEDDING_MODEL
        self.reranker_model_name = reranker_model or RERANKER_MODEL
        self._dense_model: SentenceTransformerType | None = None
        self._reranker: CrossEncoderType | None = None
        self._model_lock = threading.RLock()
        self.bm25: BM25Okapi | None = None
        self.corpus_chunks: list[DocumentChunk] = []
        self._corpus_lock = threading.RLock()

        self.pinecone = None
        self.pinecone_index = None
        self.qdrant = None

        if self.vector_db == "pinecone":
            self._initialize_pinecone(
                pinecone_api_key=pinecone_api_key,
                pinecone_cloud=pinecone_cloud,
                pinecone_region=pinecone_region,
            )
        else:
            self._initialize_qdrant(qdrant_url=qdrant_url, qdrant_path=qdrant_path)

    @property
    def dense_model(self) -> SentenceTransformer:
        with self._model_lock:
            if self._dense_model is None:
                model_class = SentenceTransformer
                if model_class is None:
                    from sentence_transformers import SentenceTransformer as model_class

                self._dense_model = model_class(self.embedding_model_name)
            return self._dense_model

    @property
    def reranker(self) -> CrossEncoder:
        with self._model_lock:
            if self._reranker is None:
                model_class = CrossEncoder
                if model_class is None:
                    from sentence_transformers import CrossEncoder as model_class

                self._reranker = model_class(self.reranker_model_name)
            return self._reranker

    def _initialize_qdrant(self, qdrant_url: str | None, qdrant_path: str | None) -> None:
        resolved_url = qdrant_url or os.getenv("QDRANT_URL")
        resolved_path = qdrant_path or os.getenv("QDRANT_PATH")
        if qdrant_path == ":memory:":
            self.qdrant = QdrantClient(":memory:")
        elif resolved_url:
            self.qdrant = QdrantClient(url=resolved_url)
        elif resolved_path and resolved_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(resolved_path)), exist_ok=True)
            self.qdrant = QdrantClient(path=resolved_path)
        else:
            self.qdrant = QdrantClient(":memory:")

        if not self.qdrant.collection_exists(self.collection_name):
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
            )

    def _initialize_pinecone(
        self,
        pinecone_api_key: str | None,
        pinecone_cloud: str | None,
        pinecone_region: str | None,
    ) -> None:
        api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY") or os.getenv("db_key")
        if not api_key:
            raise ValueError("PINECONE_API_KEY is required when VECTOR_DB=pinecone")
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:
            raise RuntimeError(
                "Pinecone support is optional; install it with `pip install -e '.[pinecone]'`."
            ) from exc

        self.pinecone = Pinecone(api_key=api_key)
        if not self.pinecone.has_index(self.collection_name):
            self.pinecone.create_index(
                name=self.collection_name,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=pinecone_cloud or os.getenv("PINECONE_CLOUD", "aws"),
                    region=pinecone_region or os.getenv("PINECONE_REGION", "us-east-1"),
                ),
            )
        self.pinecone_index = self.pinecone.Index(self.collection_name)

    def _encode(self, texts: list[str]):
        embeddings = self.dense_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        shape = getattr(embeddings, "shape", ())
        if len(shape) != 2 or shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding model must return {EMBEDDING_DIMENSION}-dimensional vectors; got {shape}"
            )
        return embeddings

    def _upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        embeddings = self._encode([chunk.text for chunk in chunks])

        if self.vector_db == "pinecone":
            vectors = [
                {
                    "id": chunk.chunk_id,
                    "values": embeddings[index].tolist(),
                    "metadata": chunk.model_dump(),
                }
                for index, chunk in enumerate(chunks)
            ]
            for start in range(0, len(vectors), 100):
                self.pinecone_index.upsert(vectors=vectors[start : start + 100])
            return

        points = [
            PointStruct(
                id=uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id),
                vector=embeddings[index].tolist(),
                payload=chunk.model_dump(),
            )
            for index, chunk in enumerate(chunks)
        ]
        self.qdrant.upsert(collection_name=self.collection_name, points=points, wait=True)

    def _build_bm25(self) -> None:
        tokenized = [_tokenize(chunk.text) for chunk in self.corpus_chunks]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def index_documents(self, chunks: Iterable[DocumentChunk]) -> None:
        """Set the in-process corpus and upsert vectors into the configured store."""
        new_chunks = list(chunks)
        ids = [chunk.chunk_id for chunk in new_chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Document chunk IDs must be unique")
        if any(
            not chunk.chunk_id.strip() or not chunk.text.strip() or not _tokenize(chunk.text)
            for chunk in new_chunks
        ):
            raise ValueError("Document chunks must have non-empty IDs and searchable text")

        with self._corpus_lock:
            if new_chunks:
                self._upsert_chunks(new_chunks)
            self.corpus_chunks = new_chunks
            self.bm25 = None
            if new_chunks:
                self._build_bm25()

    def restore_corpus(self, chunks: Iterable[DocumentChunk]) -> None:
        """Restore sparse-retrieval data from disk without re-embedding vectors."""
        restored = list(chunks)
        ids = [chunk.chunk_id for chunk in restored]
        if len(ids) != len(set(ids)):
            raise ValueError("Document chunk IDs must be unique")
        if any(
            not chunk.chunk_id.strip() or not chunk.text.strip() or not _tokenize(chunk.text)
            for chunk in restored
        ):
            raise ValueError("Document chunks must have non-empty IDs and searchable text")
        with self._corpus_lock:
            self.corpus_chunks = restored
            self._build_bm25()

    def has_vectors(self) -> bool:
        """Return whether the configured vector collection/index contains points."""
        if self.vector_db == "qdrant":
            return self.qdrant.count(collection_name=self.collection_name, exact=True).count > 0
        stats = self.pinecone_index.describe_index_stats()
        total = getattr(stats, "total_vector_count", None)
        if total is None and isinstance(stats, dict):
            total = stats.get("total_vector_count", 0)
        return bool(total)

    def _delete_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        if self.vector_db == "pinecone":
            ids = [chunk.chunk_id for chunk in chunks]
            for start in range(0, len(ids), 1000):
                self.pinecone_index.delete(ids=ids[start : start + 1000])
            return

        point_ids = [
            uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)
            for chunk in chunks
        ]
        self.qdrant.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=point_ids),
            wait=True,
        )

    def remove_sources(self, source_names: Iterable[str]) -> list[DocumentChunk]:
        """Remove selected source files from the vector and sparse indexes."""
        targets = {name.strip() for name in source_names if name.strip()}
        if not targets:
            return []

        with self._corpus_lock:
            removed = [
                chunk
                for chunk in self.corpus_chunks
                if (chunk.source_name or chunk.doc_id) in targets
            ]
            if not removed:
                return []

            self._delete_chunks(removed)
            removed_ids = {chunk.chunk_id for chunk in removed}
            self.corpus_chunks = [
                chunk for chunk in self.corpus_chunks if chunk.chunk_id not in removed_ids
            ]
            self._build_bm25()
            return removed

    def replace_sources(self, chunks: Iterable[DocumentChunk]) -> list[DocumentChunk]:
        """Replace uploaded source files by name, removing stale chunks on re-upload."""
        incoming: list[DocumentChunk] = []
        incoming_by_id: dict[str, DocumentChunk] = {}
        for chunk in chunks:
            if not chunk.chunk_id.strip() or not chunk.text.strip() or not _tokenize(chunk.text):
                raise ValueError("Document chunks must have non-empty IDs and searchable text")
            previous = incoming_by_id.get(chunk.chunk_id)
            if previous and previous != chunk:
                raise ValueError(f"Conflicting content for chunk ID {chunk.chunk_id!r}")
            if previous is None:
                incoming.append(chunk)
                incoming_by_id[chunk.chunk_id] = chunk
        if not incoming:
            return []

        def source_key(chunk: DocumentChunk) -> str:
            return chunk.source_name or chunk.doc_id

        incoming_sources = {source_key(chunk) for chunk in incoming}
        with self._corpus_lock:
            existing_by_source: dict[str, dict[str, DocumentChunk]] = {}
            for chunk in self.corpus_chunks:
                key = source_key(chunk)
                if key in incoming_sources:
                    existing_by_source.setdefault(key, {})[chunk.chunk_id] = chunk

            incoming_by_source: dict[str, dict[str, DocumentChunk]] = {}
            for chunk in incoming:
                incoming_by_source.setdefault(source_key(chunk), {})[chunk.chunk_id] = chunk

            changed_sources = {
                key
                for key in incoming_sources
                if existing_by_source.get(key, {}) != incoming_by_source.get(key, {})
            }
            if not changed_sources:
                return []

            to_upsert = [chunk for chunk in incoming if source_key(chunk) in changed_sources]
            new_ids = {chunk.chunk_id for chunk in to_upsert}
            to_remove = [
                chunk
                for key in changed_sources
                for chunk in existing_by_source.get(key, {}).values()
                if chunk.chunk_id not in new_ids
            ]
            self._upsert_chunks(to_upsert)
            self._delete_chunks(to_remove)
            self.corpus_chunks = [
                chunk for chunk in self.corpus_chunks if source_key(chunk) not in changed_sources
            ] + to_upsert
            self._build_bm25()
            return to_upsert

    def append_documents(self, chunks: Iterable[DocumentChunk]) -> list[DocumentChunk]:
        """Upsert only new chunk IDs, then extend the sparse corpus."""
        with self._corpus_lock:
            existing_by_id = {chunk.chunk_id: chunk for chunk in self.corpus_chunks}
            added: list[DocumentChunk] = []
            seen = set(existing_by_id)
            for chunk in chunks:
                if not chunk.chunk_id.strip() or not chunk.text.strip() or not _tokenize(chunk.text):
                    raise ValueError("Document chunks must have non-empty IDs and searchable text")
                if chunk.chunk_id not in seen:
                    added.append(chunk)
                    seen.add(chunk.chunk_id)
                    existing_by_id[chunk.chunk_id] = chunk
                elif existing_by_id[chunk.chunk_id] != chunk:
                    raise ValueError(f"Conflicting content for chunk ID {chunk.chunk_id!r}")
            if not added:
                return []

            self._upsert_chunks(added)
            self.corpus_chunks.extend(added)
            self._build_bm25()
            return added

    def _dense_results(self, query: str, top_k: int) -> tuple[list[str], dict[str, DocumentChunk]]:
        query_vector = self._encode([QUERY_INSTRUCTION + query])[0].tolist()
        chunk_map: dict[str, DocumentChunk] = {}

        if self.vector_db == "pinecone":
            result = self.pinecone_index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
            )
            matches = getattr(result, "matches", None)
            if matches is None and isinstance(result, dict):
                matches = result.get("matches", [])
            chunk_ids = []
            for match in matches or []:
                metadata = getattr(match, "metadata", None)
                match_id = getattr(match, "id", None)
                if isinstance(match, dict):
                    metadata = metadata or match.get("metadata")
                    match_id = match_id or match.get("id")
                metadata = metadata or {}
                chunk_id = str(metadata.get("chunk_id") or match_id or "")
                if not chunk_id:
                    continue
                chunk_ids.append(chunk_id)
                if metadata.get("text"):
                    page_number = metadata.get("page_number")
                    chunk_map[chunk_id] = DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=str(metadata.get("doc_id", "document")),
                        text=str(metadata["text"]),
                        parent_text=str(metadata.get("parent_text") or metadata["text"]),
                        source_name=metadata.get("source_name"),
                        page_number=int(page_number) if page_number is not None else None,
                    )
            return chunk_ids, chunk_map

        hits = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points
        chunk_ids = []
        for hit in hits:
            payload = hit.payload or {}
            chunk_id = str(payload.get("chunk_id", ""))
            if not chunk_id:
                continue
            chunk_ids.append(chunk_id)
            if payload.get("text"):
                page_number = payload.get("page_number")
                chunk_map[chunk_id] = DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=str(payload.get("doc_id", "document")),
                    text=str(payload["text"]),
                    parent_text=str(payload.get("parent_text") or payload["text"]),
                    source_name=payload.get("source_name"),
                    page_number=int(page_number) if page_number is not None else None,
                )
        return chunk_ids, chunk_map

    def hybrid_retrieve(
        self,
        query: str,
        top_candidates: int = 20,
        final_k: int = 5,
        mode: str = "hybrid_rerank",
    ) -> list[DocumentChunk]:
        """Retrieve using dense, BM25, RRF, or RRF with cross-encoder reranking."""
        allowed_modes = {"dense", "bm25", "hybrid", "hybrid_rerank"}
        if mode not in allowed_modes:
            raise ValueError(f"Unknown retrieval mode {mode!r}; choose one of {sorted(allowed_modes)}")
        if top_candidates <= 0 or final_k <= 0 or not query.strip():
            return []

        with self._corpus_lock:
            corpus_chunks = list(self.corpus_chunks)
            bm25_index = self.bm25

        dense_ids: list[str] = []
        dense_chunk_map: dict[str, DocumentChunk] = {}
        if mode in {"dense", "hybrid", "hybrid_rerank"}:
            dense_ids, dense_chunk_map = self._dense_results(query.strip(), top_candidates)

        sparse_ids: list[str] = []
        if mode in {"bm25", "hybrid", "hybrid_rerank"} and bm25_index and corpus_chunks:
            scores = bm25_index.get_scores(_tokenize(query))
            ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            sparse_ids = [
                corpus_chunks[index].chunk_id
                for index in ranked_indices[:top_candidates]
                if _tokenize(corpus_chunks[index].text)
            ]

        if mode == "dense":
            candidate_ids = dense_ids[:top_candidates]
        elif mode == "bm25":
            candidate_ids = sparse_ids[:top_candidates]
        else:
            rrf_scores: dict[str, float] = {}
            for ranking in (dense_ids, sparse_ids):
                for rank, chunk_id in enumerate(ranking, start=1):
                    rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)
            candidate_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_candidates]

        id_to_chunk = {chunk.chunk_id: chunk for chunk in corpus_chunks}
        id_to_chunk.update(dense_chunk_map)
        candidates = [id_to_chunk[chunk_id] for chunk_id in candidate_ids if chunk_id in id_to_chunk]
        if not candidates:
            return []
        if mode != "hybrid_rerank":
            return candidates[:final_k]

        scores = self.reranker.predict([[query, chunk.text] for chunk in candidates])
        reranked = sorted(range(len(candidates)), key=lambda index: float(scores[index]), reverse=True)
        return [candidates[index] for index in reranked[:final_k]]

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid_rerank",
        k: int = 5,
        top_candidates: int = 20,
    ) -> list[DocumentChunk]:
        """Stable public method used by the evaluation harness."""
        return self.hybrid_retrieve(query, top_candidates=top_candidates, final_k=k, mode=mode)
