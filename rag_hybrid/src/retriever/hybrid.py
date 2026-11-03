'''Hybrid Retrieval & Reranking implementation.

- Dense vector search via Qdrant or Pinecone (Sentence-Transformer embeddings)
- Sparse BM25 search (rank-bm25)
- Reciprocal Rank Fusion (RRF) to combine both rankings
- Cross-Encoder reranker (MiniLM-MS-MARCO style) for final top-N selection
'''

import os
import uuid
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()


# ----------------------------------------------------------------------
#   Data models
# ----------------------------------------------------------------------
class DocumentChunk(BaseModel):
    """A child chunk that will be indexed.

    Attributes
    ----------
    chunk_id: str
        Unique identifier for the child chunk.
    doc_id: str
        Identifier of the parent document.
    text: str
        The child chunk text (used for dense & sparse retrieval).
    parent_text: str
        The surrounding parent chunk (used for generation context).
    """

    chunk_id: str
    doc_id: str
    text: str
    parent_text: str


# ----------------------------------------------------------------------
#   Hybrid retriever class
# ----------------------------------------------------------------------
class ProductionHybridRAG:
    """End-to-end hybrid retrieval pipeline supporting Qdrant and Pinecone.

    Parameters
    ----------
    collection_name: str, optional
        Name of the collection / index (defaults to "rag-hybrid" for Pinecone
        or "private_corpus" for Qdrant).
    vector_db: str, optional
        Vector database backend to use: "qdrant" or "pinecone".
        If omitted, defaults to the VECTOR_DB env variable or "qdrant".
    pinecone_api_key: str, optional
        Pinecone API key. If omitted, read from PINECONE_API_KEY or db_key env vars.
    pinecone_cloud: str, optional
        Pinecone serverless cloud provider (default "aws").
    pinecone_region: str, optional
        Pinecone serverless region (default "us-east-1").
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        vector_db: Optional[str] = None,
        pinecone_api_key: Optional[str] = None,
        pinecone_cloud: str = "aws",
        pinecone_region: str = "us-east-1",
    ):
        # Resolve vector DB choice
        db_choice = (vector_db or os.getenv("VECTOR_DB", "qdrant")).strip().lower()
        self.vector_db = db_choice

        # Default collection / index name
        if collection_name:
            self.collection_name = collection_name
        elif self.vector_db == "pinecone":
            self.collection_name = os.getenv("PINECONE_INDEX_NAME", "rag-hybrid")
        else:
            self.collection_name = "private_corpus"

        self.dense_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_chunks: List[DocumentChunk] = []

        # Initialize vector database client
        if self.vector_db == "pinecone":
            from pinecone import Pinecone, ServerlessSpec

            api_key = (
                pinecone_api_key
                or os.getenv("PINECONE_API_KEY")
                or os.getenv("db_key")
            )
            if not api_key:
                raise ValueError(
                    "Pinecone API key not found. Set PINECONE_API_KEY or db_key in your .env file."
                )

            self.pinecone = Pinecone(api_key=api_key)
            # Create serverless index if it doesn't already exist
            if not self.pinecone.has_index(self.collection_name):
                self.pinecone.create_index(
                    name=self.collection_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region),
                )
            self.pinecone_index = self.pinecone.Index(self.collection_name)
            self.qdrant = None
        else:
            # Default: in-memory Qdrant client
            self.qdrant = QdrantClient(":memory:")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            self.pinecone = None
            self.pinecone_index = None

    # ------------------------------------------------------------------
    #   Indexing
    # ------------------------------------------------------------------
    def index_documents(self, chunks: List[DocumentChunk]) -> None:
        """Ingest dense vectors into the vector DB and build the BM25 index."""
        self.corpus_chunks = chunks
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.dense_model.encode(texts, show_progress_bar=False)

        # ---- Dense vectors ----
        if self.vector_db == "pinecone":
            vectors = []
            for idx, chunk in enumerate(chunks):
                vectors.append(
                    {
                        "id": chunk.chunk_id,
                        "values": embeddings[idx].tolist(),
                        "metadata": chunk.model_dump(),
                    }
                )
            # Upsert in batches of 100
            for i in range(0, len(vectors), 100):
                self.pinecone_index.upsert(vectors=vectors[i : i + 100])
        else:
            points = []
            for idx, chunk in enumerate(chunks):
                points.append(
                    PointStruct(
                        id=uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id),
                        vector=embeddings[idx].tolist(),
                        payload=chunk.model_dump(),
                    )
                )
            self.qdrant.upsert(collection_name=self.collection_name, points=points)

        # ---- Sparse BM25 ----
        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def append_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add uploaded chunks while retaining documents already indexed."""
        if not chunks:
            return
        existing_ids = {chunk.chunk_id for chunk in self.corpus_chunks}
        new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
        if new_chunks:
            self.index_documents(self.corpus_chunks + new_chunks)

    # ------------------------------------------------------------------
    #   Retrieval
    # ------------------------------------------------------------------
    def hybrid_retrieve(
        self,
        query: str,
        top_candidates: int = 20,
        final_k: int = 5,
        mode: str = "hybrid_rerank",
    ) -> List[DocumentChunk]:
        """Retrieve chunks using one of the documented ablation configurations.

        ``dense`` (C1), ``bm25`` (C2), ``hybrid`` (C3), and
        ``hybrid_rerank`` (C4) are supported. The default retains the original
        hybrid-plus-reranker behavior.
        """
        if mode not in {"dense", "bm25", "hybrid", "hybrid_rerank"}:
            raise ValueError(
                f"Unknown retrieval mode {mode!r}; expected dense, bm25, hybrid, or hybrid_rerank"
            )
        if top_candidates <= 0 or final_k <= 0:
            return []

        dense_cids: List[str] = []
        dense_chunk_map: Dict[str, DocumentChunk] = {}

        # ---- Dense retrieval ----
        if mode in {"dense", "hybrid", "hybrid_rerank"}:
            query_vec = self.dense_model.encode(query).tolist()
            if self.vector_db == "pinecone":
                query_res = self.pinecone_index.query(
                    vector=query_vec,
                    top_k=top_candidates,
                    include_metadata=True,
                )
                matches = getattr(query_res, "matches", None)
                if matches is None and isinstance(query_res, dict):
                    matches = query_res.get("matches", [])

                for match in matches or []:
                    meta = (
                        getattr(match, "metadata", None)
                        or (match.get("metadata") if isinstance(match, dict) else {})
                        or {}
                    )
                    match_id = getattr(match, "id", None) or (
                        match.get("id") if isinstance(match, dict) else ""
                    )
                    cid = str(meta.get("chunk_id") or match_id)
                    if cid:
                        dense_cids.append(cid)
                        if cid not in dense_chunk_map and "text" in meta:
                            dense_chunk_map[cid] = DocumentChunk(
                                chunk_id=cid,
                                doc_id=meta.get("doc_id", "doc"),
                                text=meta.get("text", ""),
                                parent_text=meta.get("parent_text", meta.get("text", "")),
                            )
            else:
                dense_hits = self.qdrant.query_points(
                    collection_name=self.collection_name,
                    query=query_vec,
                    limit=top_candidates,
                ).points
                dense_cids = [str(hit.payload["chunk_id"]) for hit in dense_hits]

        # ---- Sparse BM25 retrieval ----
        sparse_cids: List[str] = []
        if mode in {"bm25", "hybrid", "hybrid_rerank"} and self.bm25 is not None and self.corpus_chunks:
            tokenized_q = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_q)  # type: ignore[arg-type]
            sparse_idx = sorted(
                range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
            )[:top_candidates]
            sparse_cids = [self.corpus_chunks[idx].chunk_id for idx in sparse_idx]

        # ---- Select a ranked candidate pool ----
        if mode == "dense":
            fused_ids = dense_cids[:top_candidates]
        elif mode == "bm25":
            fused_ids = sparse_cids[:top_candidates]
        else:
            k_smooth = 60
            rrf: Dict[str, float] = {}
            for rank, cid in enumerate(dense_cids, start=1):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k_smooth + rank)
            for rank, cid in enumerate(sparse_cids, start=1):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k_smooth + rank)
            fused_ids = sorted(rrf, key=rrf.get, reverse=True)[:top_candidates]

        id_to_chunk = {c.chunk_id: c for c in self.corpus_chunks}
        # Fall back to metadata received from Pinecone if not in local corpus_chunks
        id_to_chunk.update(dense_chunk_map)

        candidates = [id_to_chunk[cid] for cid in fused_ids if cid in id_to_chunk]
        if not candidates:
            return []

        candidates = candidates[:top_candidates]
        if mode != "hybrid_rerank":
            return candidates[:final_k]

        # ---- Cross-Encoder reranking ----
        pairs = [[query, c.text] for c in candidates]
        cross_scores = self.reranker.predict(pairs)
        rerank_idx = sorted(
            range(len(cross_scores)), key=lambda i: cross_scores[i], reverse=True
        )[:final_k]

        return [candidates[i] for i in rerank_idx]

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid_rerank",
        k: int = 5,
        top_candidates: int = 20,
    ) -> List[DocumentChunk]:
        """Stable public entry point used by the evaluation harness."""
        return self.hybrid_retrieve(
            query,
            top_candidates=top_candidates,
            final_k=k,
            mode=mode,
        )


"""End of ProductionHybridRAG implementation."""
