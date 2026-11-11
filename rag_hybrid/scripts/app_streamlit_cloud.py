"""Streamlit Community Cloud entry point for the free, in-memory demo."""

import os

os.environ["RAG_INLINE_ONLY"] = "true"

from rag_hybrid.scripts import app_streamlit  # noqa: F401,E402
