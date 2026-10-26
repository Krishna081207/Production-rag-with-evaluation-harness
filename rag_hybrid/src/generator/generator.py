# Generator implementation for RAG pipeline
"""Generation utilities.

The generator follows a strict citation contract:
- Each factual sentence must end with a citation ``[Chunk_<ID>]``.
- If the context does not contain enough information, a fallback message is returned.
"""

import os
from typing import List, Dict
from pydantic import BaseModel

from rag_hybrid.src.retriever.hybrid import DocumentChunk


class GenerationResult(BaseModel):
    answer: str
    citations: List[str]


def assemble_context(chunks: List[DocumentChunk]) -> str:
    """Create a single context string from the parent chunks.
    The parent_text provides enough surrounding material for the LLM.
    """
    # Preserve order; here we simply join unique parent texts.
    seen = set()
    parts = []
    for c in chunks:
        if c.parent_text not in seen:
            seen.add(c.parent_text)
            parts.append(c.parent_text)
    return "\n\n".join(parts)


def generate_answer(query: str, retrieved_chunks: List[DocumentChunk]) -> GenerationResult:
    """Generate a grounded answer for *query*.

    For demonstration purposes we use a very simple rule‑based stub:
    - If at least one chunk is retrieved, we concatenate the first two sentences
      from the first parent chunk and append the appropriate citation.
    - If no chunks are found we return the fallback message.
    """
    if not retrieved_chunks:
        return GenerationResult(
            answer="The provided documentation does not contain sufficient evidence to answer this inquiry.",
            citations=[],
        )

    # Use the parent text of the top‑ranked chunk as context.
    parent_text = retrieved_chunks[0].parent_text
    # Simple split into sentences (naïve). Take first two sentences.
    sentences = [s.strip() for s in parent_text.split('.') if s]
    selected = ". ".join(sentences[:2])
    if selected:
        selected = selected.strip() + "."
    citation = f"[Chunk_{retrieved_chunks[0].chunk_id}]"
    answer = f"{selected} {citation}"
    return GenerationResult(answer=answer, citations=[citation])
