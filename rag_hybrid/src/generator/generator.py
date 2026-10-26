# Generator implementation for RAG pipeline
"""Generation utilities.

The generator follows a strict citation contract:
- Each factual sentence must end with a citation ``[Chunk_<ID>]``.
- If the context does not contain enough information, a fallback message is returned.
"""

import os
from typing import List

# Load environment variables (requires python-dotenv in dependencies)
from dotenv import load_dotenv
from pydantic import BaseModel

from rag_hybrid.src.retriever.hybrid import DocumentChunk

load_dotenv()

# Groq client – ensure groq SDK is installed.
try:
    from groq import Groq
except ImportError as e:
    raise ImportError("Groq SDK not installed. Add 'groq' to pyproject dependencies.") from e

class GenerationResult(BaseModel):
    answer: str
    citations: List[str]

def _build_prompt(query: str, chunks: List[DocumentChunk]) -> str:
    """Create a prompt that enforces the citation contract.

    The prompt includes the user query and the parent texts of the supplied chunks.
    """
    context = "\n\n".join(chunk.parent_text for chunk in chunks)
    return f"""You are a factual assistant. Answer the user's question using ONLY the provided context. Cite each factual sentence with the corresponding chunk identifier in the form [Chunk_<ID>].

User query: {query}

Context:
{context}

If the context does not contain sufficient information, respond with: \"The provided documentation does not contain sufficient evidence to answer this inquiry.\"
"""

def generate_answer(query: str, retrieved_chunks: List[DocumentChunk]) -> GenerationResult:
    """Generate a grounded answer using Groq LLM.

    Parameters
    ----------
    query: str
        User question.
    retrieved_chunks: List[DocumentChunk]
        Ranked chunks from the hybrid retriever.
    """
    if not retrieved_chunks:
        return GenerationResult(
            answer="The provided documentation does not contain sufficient evidence to answer this inquiry.",
            citations=[],
        )

    # Build prompt from top‑N chunks (use all provided)
    prompt = _build_prompt(query, retrieved_chunks)

    client = Groq(api_key=os.getenv("GROQ_API_KEY") or os.getenv("groq"))
    try:
        response = client.chat.completions.create(
            model="llama3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        answer_text = response.choices[0].message.content.strip()
    except Exception as e:
        answer_text = "The provided documentation does not contain sufficient evidence to answer this inquiry."
        print(f"Groq API error: {e}")

    # Extract citations (Chunk_<ID>) from the answer
    import re
    found = re.findall(r"\[Chunk_([^\]]+)\]", answer_text)
    citations = [f"Chunk_{c}" for c in found]

    return GenerationResult(answer=answer_text, citations=citations)
