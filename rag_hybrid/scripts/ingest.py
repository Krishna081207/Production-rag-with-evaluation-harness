import argparse
import glob
import os
from pathlib import Path

from rag_hybrid.src.corpus_store import load_corpus, save_corpus
from rag_hybrid.src.ingestion import chunk_document_segments, read_document_segments
from rag_hybrid.src.retriever.hybrid import DocumentChunk, ProductionHybridRAG


def load_raw_documents(data_dir: str) -> list[tuple[str, list[tuple[int | None, str]]]]:
    """Load supported documents as (relative path, page segments) tuples."""
    docs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "**/*"), recursive=True)):
        if os.path.isfile(path) and Path(path).suffix.lower() in {".txt", ".md", ".pdf", ".docx"}:
            with open(path, "rb") as file:
                _, segments = read_document_segments(path, file.read())
            doc_id = os.path.relpath(path, data_dir).replace(os.sep, "/")
            docs.append((doc_id, segments))
    return docs


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the vector database.")
    parser.add_argument(
        "--source",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Path to documents directory (default: ../data)",
    )
    parser.add_argument(
        "--vector-db",
        choices=["qdrant", "pinecone"],
        default=os.getenv("VECTOR_DB", "qdrant"),
        help="Vector DB backend to use: 'pinecone' or 'qdrant' (default: env VECTOR_DB or qdrant)",
    )
    parser.add_argument(
        "--index-name",
        default=None,
        help="Collection / index name (defaults to rag-hybrid)",
    )
    args = parser.parse_args()

    rag = ProductionHybridRAG(collection_name=args.index_name, vector_db=args.vector_db)
    saved_chunks = load_corpus()
    if saved_chunks:
        if rag.has_vectors():
            rag.restore_corpus(saved_chunks)
        else:
            rag.index_documents(saved_chunks)

    new_chunks: list[DocumentChunk] = []
    for doc_id, segments in load_raw_documents(args.source):
        new_chunks.extend(chunk_document_segments(doc_id, segments))

    if not new_chunks:
        print("No documents found in", args.source)
        return

    added = rag.replace_sources(new_chunks)
    save_corpus(rag.corpus_chunks)
    print(
        f"Indexed {len(added)} new child chunks across "
        f"{len(set(c.doc_id for c in added))} new documents into {rag.vector_db.upper()} "
        f"(index: {rag.collection_name})."
    )


if __name__ == "__main__":
    main()
