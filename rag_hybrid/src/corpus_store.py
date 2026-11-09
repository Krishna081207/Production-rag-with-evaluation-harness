"""Atomic JSONL persistence for the in-process sparse retrieval corpus."""

import json
from pathlib import Path
from typing import Iterable

from rag_hybrid.src.config import corpus_path
from rag_hybrid.src.retriever.hybrid import DocumentChunk


def load_corpus(path: Path | None = None) -> list[DocumentChunk]:
    source = path or corpus_path()
    if not source.exists():
        return []

    chunks: list[DocumentChunk] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunks.append(DocumentChunk.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid corpus record at {source}:{line_number}") from exc
    return chunks


def save_corpus(chunks: Iterable[DocumentChunk], path: Path | None = None) -> None:
    destination = path or corpus_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
        handle.flush()
    temporary.replace(destination)
