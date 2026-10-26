# tests/test_retriever.py
"""Unit tests for the hybrid retriever (ProductionHybridRAG).

All tests use an in-memory Qdrant instance – no external services required.
"""

import pytest

from rag_hybrid.src.retriever.hybrid import DocumentChunk, ProductionHybridRAG

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def three_chunks():
    return [
        DocumentChunk(
            chunk_id="c1",
            doc_id="doc1",
            text="Python is a dynamically typed programming language.",
            parent_text="Python is a dynamically typed programming language. It is widely used.",
        ),
        DocumentChunk(
            chunk_id="c2",
            doc_id="doc2",
            text="FastAPI is a high-performance web framework for building APIs.",
            parent_text="FastAPI is a high-performance web framework for building APIs with Python.",
        ),
        DocumentChunk(
            chunk_id="c3",
            doc_id="doc3",
            text="Qdrant stores dense vectors for similarity search.",
            parent_text="Qdrant stores dense vectors and supports cosine similarity search.",
        ),
    ]


@pytest.fixture(scope="module")
def indexed_rag(three_chunks):
    rag = ProductionHybridRAG()
    rag.index_documents(three_chunks)
    return rag


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

class TestIndexing:
    def test_index_populates_corpus(self, three_chunks):
        rag = ProductionHybridRAG()
        rag.index_documents(three_chunks)
        assert len(rag.corpus_chunks) == 3

    def test_bm25_built_after_index(self, indexed_rag):
        assert indexed_rag.bm25 is not None

    def test_reindex_replaces_corpus(self, three_chunks):
        rag = ProductionHybridRAG()
        rag.index_documents(three_chunks[:1])
        assert len(rag.corpus_chunks) == 1
        rag.index_documents(three_chunks)
        assert len(rag.corpus_chunks) == 3


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class TestRetrieval:
    def test_returns_list_of_document_chunks(self, indexed_rag):
        results = indexed_rag.hybrid_retrieve("What is Python?", top_candidates=10, final_k=2)
        assert isinstance(results, list)
        assert all(isinstance(c, DocumentChunk) for c in results)

    def test_final_k_respected(self, indexed_rag):
        for k in (1, 2, 3):
            results = indexed_rag.hybrid_retrieve("Python", top_candidates=10, final_k=k)
            assert len(results) == k

    def test_top_candidates_capped_at_corpus_size(self, indexed_rag):
        # top_candidates > corpus size should not crash
        results = indexed_rag.hybrid_retrieve("API", top_candidates=100, final_k=2)
        assert len(results) == 2

    def test_relevant_chunk_surfaces(self, indexed_rag):
        """The Python chunk should be in the top result for a Python query."""
        results = indexed_rag.hybrid_retrieve("programming language Python", top_candidates=10, final_k=1)
        assert results[0].chunk_id == "c1"

    def test_fastapi_query_surfaces_c2(self, indexed_rag):
        results = indexed_rag.hybrid_retrieve("FastAPI web framework", top_candidates=10, final_k=1)
        assert results[0].chunk_id == "c2"

    def test_chunk_ids_unique_in_result(self, indexed_rag):
        results = indexed_rag.hybrid_retrieve("Python", top_candidates=10, final_k=3)
        ids = [c.chunk_id for c in results]
        assert len(ids) == len(set(ids))

    def test_parent_text_preserved(self, indexed_rag, three_chunks):
        results = indexed_rag.hybrid_retrieve("Python", top_candidates=10, final_k=1)
        expected = {c.chunk_id: c.parent_text for c in three_chunks}
        assert results[0].parent_text == expected[results[0].chunk_id]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestRetrieverEdgeCases:
    def test_single_chunk_corpus(self):
        rag = ProductionHybridRAG()
        chunk = DocumentChunk(
            chunk_id="only",
            doc_id="d1",
            text="Only document in the corpus.",
            parent_text="Only document in the corpus.",
        )
        rag.index_documents([chunk])
        results = rag.hybrid_retrieve("document", top_candidates=5, final_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "only"

    def test_numeric_only_query(self, indexed_rag):
        """Should not crash on a query containing only numbers."""
        results = indexed_rag.hybrid_retrieve("42 0 999", top_candidates=5, final_k=2)
        assert isinstance(results, list)

    def test_empty_query_string(self, indexed_rag):
        """Empty query should not crash, though results may be arbitrary."""
        results = indexed_rag.hybrid_retrieve("", top_candidates=5, final_k=1)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Pinecone integration
# ---------------------------------------------------------------------------

class TestPineconeRetriever:
    def test_missing_api_key_raises_error(self, monkeypatch):
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        monkeypatch.delenv("db_key", raising=False)
        with pytest.raises(ValueError, match="Pinecone API key not found"):
            ProductionHybridRAG(vector_db="pinecone", pinecone_api_key=None)

    def test_pinecone_index_and_retrieve(self, monkeypatch, three_chunks):
        from unittest.mock import MagicMock, patch

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.has_index.return_value = False
        mock_pc.Index.return_value = mock_index

        # Mock query return
        mock_query_res = MagicMock()
        mock_query_res.matches = [
            MagicMock(
                id="c1",
                metadata={
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "text": "Python is a dynamically typed programming language.",
                    "parent_text": "Python is a dynamically typed programming language. It is widely used.",
                },
            )
        ]
        mock_index.query.return_value = mock_query_res

        with patch("pinecone.Pinecone", return_value=mock_pc):
            rag = ProductionHybridRAG(
                vector_db="pinecone",
                pinecone_api_key="mock_key",
                collection_name="test-rag-hybrid",
            )
            assert rag.vector_db == "pinecone"
            mock_pc.create_index.assert_called_once()

            # Test indexing
            rag.index_documents(three_chunks)
            assert mock_index.upsert.called

            # Test retrieval
            results = rag.hybrid_retrieve("Python", top_candidates=5, final_k=1)
            assert len(results) == 1
            assert results[0].chunk_id == "c1"

