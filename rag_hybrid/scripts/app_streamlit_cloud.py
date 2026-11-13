"""Streamlit Community Cloud entry point for the free, in-memory demo."""

import os
import sys
from pathlib import Path

os.environ["RAG_INLINE_ONLY"] = "true"
os.environ.setdefault("RAG_INLINE_RETRIEVAL_MODE", "hybrid")

# Streamlit Cloud can start a nested entry point with only its script directory
# on sys.path. Add the repository root so the rag_hybrid package is importable.
repository_root = str(Path(__file__).resolve().parents[2])
if repository_root not in sys.path:
    sys.path.insert(0, repository_root)

from rag_hybrid.scripts import app_streamlit  # noqa: F401,E402
