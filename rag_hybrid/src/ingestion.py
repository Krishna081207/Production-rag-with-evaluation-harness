"""Document parsing and hierarchical chunking for the RAG pipeline."""

from io import BytesIO
from pathlib import Path
from typing import List

from rag_hybrid.src.retriever.hybrid import DocumentChunk


def read_document_bytes(filename: str, content: bytes) -> str:
    """Extract text from a supported uploaded document."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        import fitz

        with fitz.open(stream=content, filetype="pdf") as document:
            return "\n".join(page.get_text("text") for page in document)
    if suffix == ".docx":
        from docx import Document

        document = Document(BytesIO(content))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    if suffix in {".txt", ".md"}:
        return content.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported document type: {suffix or 'unknown'}")


def chunk_document(doc_id: str, text: str, child_len: int = 300) -> List[DocumentChunk]:
    """Create child chunks with parent context for retrieval and generation."""
    words = text.split()
    child_chunks = [
        (f"{doc_id}_c{i // child_len}", " ".join(words[i : i + child_len]))
        for i in range(0, len(words), child_len)
    ]

    chunks: List[DocumentChunk] = []
    for i in range(0, len(child_chunks), 4):
        group = child_chunks[i : i + 4]
        parent_text = " ".join(text for _, text in group)
        chunks.extend(
            DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=child_text,
                parent_text=parent_text,
            )
            for chunk_id, child_text in group
        )
    return chunks
