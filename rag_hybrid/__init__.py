# Top‑level package for rag_hybrid.
# Exposes the sub‑modules under ``rag_hybrid.src``.
# Users can import like ``from rag_hybrid.src.retriever.hybrid import ProductionHybridRAG``.
"""rag_hybrid – production‑grade Retrieval‑Augmented Generation toolkit.

This package follows a ``src`` layout.  The actual implementation lives in
``rag_hybrid/src``; the top‑level ``rag_hybrid`` module re‑exports the ``src``
namespace for convenience.
"""

# Re‑export the ``src`` namespace so that ``import rag_hybrid.src`` works.
from . import src  # noqa: F401
