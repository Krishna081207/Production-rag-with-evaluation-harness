import argparse
import glob
import os
from typing import List

from rag_hybrid.src.retriever.hybrid import DocumentChunk, ProductionHybridRAG


def load_raw_documents(data_dir: str) -> List[tuple[str, str]]:
    """Load plain-text documents from *data_dir*.
    Returns a list of (doc_id, text) tuples.
    """
    docs = []
    for path in glob.glob(os.path.join(data_dir, "**/*"), recursive=True):
        if os.path.isfile(path) and path.lower().endswith((".txt", ".md", ".pdf")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            doc_id = os.path.relpath(path, data_dir).replace(os.sep, "_")
            docs.append((doc_id, txt))
    return docs


def chunk_document(doc_id: str, text: str, child_len: int = 300) -> List[DocumentChunk]:
    """Create hierarchical parent/child chunks.

    * child_len – number of words per child chunk (default 300).
    * parent chunk – concatenates up to 4 consecutive child chunks (~1200 words).
    """
    words = text.split()
    child_chunks = []
    for i in range(0, len(words), child_len):
        child_text = " ".join(words[i : i + child_len])
        child_id = f"{doc_id}_c{i // child_len}"
        child_chunks.append((child_id, child_text))

    # Build parent chunks (group of up to 4 child chunks)
    parent_chunks: List[DocumentChunk] = []
    for i in range(0, len(child_chunks), 4):
        group = child_chunks[i : i + 4]
        parent_text = " ".join(c[1] for c in group)
        for child_id, child_text in group:
            parent_chunks.append(
                DocumentChunk(
                    chunk_id=child_id,
                    doc_id=doc_id,
                    text=child_text,
                    parent_text=parent_text,
                )
            )
    return parent_chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the vector database.")
    parser.add_argument(
        "--source",
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="Path to documents directory (default: ../data)",
    )
    parser.add_argument(
        "--vector-db",
        choices=["qdrant", "pinecone"],
        default=os.getenv("VECTOR_DB", "pinecone"),
        help="Vector DB backend to use: 'pinecone' or 'qdrant' (default: env VECTOR_DB or pinecone)",
    )
    parser.add_argument(
        "--index-name",
        default=None,
        help="Collection / index name (defaults to rag-hybrid for Pinecone or private_corpus for Qdrant)",
    )
    args = parser.parse_args()

    rag = ProductionHybridRAG(collection_name=args.index_name, vector_db=args.vector_db)
    all_chunks: List[DocumentChunk] = []
    for doc_id, txt in load_raw_documents(args.source):
        all_chunks.extend(chunk_document(doc_id, txt))

    if not all_chunks:
        print("No documents found in", args.source)
        return

    rag.index_documents(all_chunks)
    print(
        f"Indexed {len(all_chunks)} child chunks across "
        f"{len(set(c.doc_id for c in all_chunks))} documents into {rag.vector_db.upper()} "
        f"(index: {rag.collection_name})."
    )


if __name__ == "__main__":
    main()
