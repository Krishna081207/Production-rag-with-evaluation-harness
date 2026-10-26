import os
import glob
from typing import List

from sentence_transformers import SentenceTransformer

from rag_hybrid.src.retriever.hybrid import DocumentChunk, ProductionHybridRAG


def load_raw_documents(data_dir: str) -> List[tuple[str, str]]:
    """Load plain‑text documents from *data_dir*.
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
    * parent chunk – concatenates up to 4 consecutive child chunks (≈1200 words).
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
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    rag = ProductionHybridRAG()
    all_chunks: List[DocumentChunk] = []
    for doc_id, txt in load_raw_documents(data_dir):
        all_chunks.extend(chunk_document(doc_id, txt))
    if not all_chunks:
        print("No documents found in", data_dir)
        return
    rag.index_documents(all_chunks)
    print(f"Indexed {len(all_chunks)} child chunks across {len(set(c.doc_id for c in all_chunks))} documents.")


if __name__ == "__main__":
    main()
