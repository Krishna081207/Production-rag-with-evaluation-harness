"""Document parsing and page-aware hierarchical chunking for research sources."""

import hashlib
import re
from collections.abc import Iterable
from io import BytesIO
from pathlib import PurePosixPath

from rag_hybrid.src.retriever.hybrid import DocumentChunk

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def read_document_segments(filename: str, content: bytes) -> tuple[str, list[tuple[int | None, str]]]:
    """Extract text and page segments, preserving PDF page numbers for citations."""
    portable_name = filename.replace("\\", "/")
    suffix = PurePosixPath(portable_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {suffix or 'unknown'}")
    if not content:
        raise ValueError("Document is empty")

    if suffix == ".pdf":
        try:
            import fitz

            with fitz.open(stream=content, filetype="pdf") as document:
                if document.is_encrypted:
                    raise ValueError("Encrypted PDF files are not supported")
                segments = [
                    (page_index + 1, page.get_text("text").strip())
                    for page_index, page in enumerate(document)
                ]
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Could not read the PDF document") from exc
    elif suffix == ".docx":
        try:
            from docx import Document

            document = Document(BytesIO(content))
            paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
            tables = [
                "\n".join("\t".join(cell.text.strip() for cell in row.cells) for row in table.rows)
                for table in document.tables
            ]
            text = "\n".join([*paragraphs, *tables]).strip()
        except Exception as exc:
            raise ValueError("Could not read the DOCX document") from exc
        segments = [(None, text)]
    else:
        try:
            text = content.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("Text files must use UTF-8 encoding") from exc
        segments = [(None, text)]

    segments = [(page, text.strip()) for page, text in segments if text.strip()]
    if not segments:
        raise ValueError("No readable text was found in the document")
    return "\n".join(text for _, text in segments), segments


def read_document_bytes(filename: str, content: bytes) -> str:
    """Extract plain text from a supported document."""
    text, _ = read_document_segments(filename, content)
    return text


def _stable_doc_id(filename: str, text: str) -> str:
    portable_name = filename.replace("\\", "/")
    source_path = PurePosixPath(portable_name)
    base = re.sub(r"[^A-Za-z0-9_-]+", "-", source_path.stem).strip("-_") or "document"
    identity = f"{source_path.as_posix()}\0{text}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"{base}-{digest}"


def _chunk_text(
    filename: str,
    text: str,
    stable_id: str,
    child_len: int,
    page_number: int | None,
) -> list[DocumentChunk]:
    words = text.split()
    if not words:
        return []

    source_name = PurePosixPath(filename.replace("\\", "/")).as_posix()
    page_prefix = f"_p{page_number}" if page_number is not None else ""
    child_chunks = [
        (f"{stable_id}{page_prefix}_c{index // child_len}", " ".join(words[index : index + child_len]))
        for index in range(0, len(words), child_len)
    ]

    chunks: list[DocumentChunk] = []
    for index in range(0, len(child_chunks), 4):
        group = child_chunks[index : index + 4]
        parent_text = " ".join(child_text for _, child_text in group)
        chunks.extend(
            DocumentChunk(
                chunk_id=chunk_id,
                doc_id=stable_id,
                text=child_text,
                parent_text=parent_text,
                source_name=source_name,
                page_number=page_number,
            )
            for chunk_id, child_text in group
        )
    return chunks


def chunk_document(doc_id: str, text: str, child_len: int = 300) -> list[DocumentChunk]:
    """Create hierarchical chunks from plain text or a non-paginated document."""
    if child_len <= 0:
        raise ValueError("child_len must be a positive integer")
    stable_id = _stable_doc_id(doc_id, text)
    return _chunk_text(doc_id, text, stable_id, child_len, page_number=None)


def chunk_document_segments(
    doc_id: str,
    segments: Iterable[tuple[int | None, str]],
    child_len: int = 300,
) -> list[DocumentChunk]:
    """Chunk extracted page segments with one stable document ID and page citations."""
    if child_len <= 0:
        raise ValueError("child_len must be a positive integer")
    pages = [(page_number, text.strip()) for page_number, text in segments if text.strip()]
    if not pages:
        return []
    full_text = "\n".join(text for _, text in pages)
    stable_id = _stable_doc_id(doc_id, full_text)
    return [
        chunk
        for page_number, text in pages
        for chunk in _chunk_text(doc_id, text, stable_id, child_len, page_number)
    ]
