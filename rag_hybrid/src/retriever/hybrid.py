'''Hybrid Retrieval & Reranking implementation.

- Dense vector search via Qdrant (Sentence‑Transformer embeddings)
- Sparse BM25 search (rank‑bm25)
- Reciprocal Rank Fusion (RRF) to combine both rankings
- Cross‑Encoder reranker (MiniLM‑MarCo style) for final top‑N selection
'''

import uuid
from typing import List, Dict

from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


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
    """End‑to‑end hybrid retrieval pipeline.

    Parameters
    ----------
    collection_name: str, optional
        Name of the Qdrant collection (default ``"private_corpus"``).
    """

    def __init__(self, collection_name: str = "private_corpus"):
        self.collection_name = collection_name
        # In‑memory Qdrant for demo purposes – replace with a persistent host for production.
        self.qdrant = QdrantClient(":memory:")
        self.dense_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.bm25: BM25Okapi | None = None
        self.corpus_chunks: List[DocumentChunk] = []

        # Create collection if it does not exist.
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    # ------------------------------------------------------------------
    #   Indexing
    # ------------------------------------------------------------------
    def index_documents(self, chunks: List[DocumentChunk]) -> None:
        """Ingest both dense vectors (Qdrant) and sparse BM25 data.

        This method stores the child chunks in Qdrant and builds a BM25 index over
        the raw ``text`` field.
        """
        self.corpus_chunks = chunks

        # ---- Dense vectors ----
        texts = [c.text for c in chunks]
        embeddings = self.dense_model.encode(texts, show_progress_bar=False)

        points = []
        for idx, chunk in enumerate(chunks):
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                    vector=embeddings[idx].tolist(),
                    payload=chunk.model_dump(),
                )
            )
        self.qdrant.upsert(collection_name=self.collection_name, points=points)

        # ---- Sparse BM25 ----
        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    # ------------------------------------------------------------------
    #   Retrieval
    # ------------------------------------------------------------------
    def hybrid_retrieve(
        self,
        query: str,
        top_candidates: int = 20,
        final_k: int = 5,
    ) -> List[DocumentChunk]:
        """Return ``final_k`` candidate chunks after RRF fusion and cross‑encoder rerank.
        """
        # ---- Dense retrieval ----
        query_vec = self.dense_model.encode(query).tolist()
        dense_hits = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            limit=top_candidates,
        )

        # ---- Sparse BM25 retrieval ----
        tokenized_q = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_q)  # type: ignore[arg-type]
        sparse_idx = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:top_candidates]

        # ---- Reciprocal Rank Fusion (RRF) ----
        k_smooth = 60
        rrf: Dict[str, float] = {}

        # Dense contribution
        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit.payload["chunk_id"]
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k_smooth + rank)

        # Sparse contribution
        for rank, idx in enumerate(sparse_idx, start=1):
            cid = self.corpus_chunks[idx].chunk_id
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k_smooth + rank)

        # Keep top candidates for reranking
        fused_ids = sorted(rrf, key=rrf.get, reverse=True)[:top_candidates]
        id_to_chunk = {c.chunk_id: c for c in self.corpus_chunks}
        candidates = [id_to_chunk[cid] for cid in fused_ids]

        # ---- Cross‑Encoder reranking ----
        pairs = [[query, c.text] for c in candidates]
        cross_scores = self.reranker.predict(pairs)
        rerank_idx = sorted(
            range(len(cross_scores)), key=lambda i: cross_scores[i], reverse=True
        )[:final_k]

        return [candidates[i] for i in rerank_idx]

"""End of ProductionHybridRAG implementation."""
