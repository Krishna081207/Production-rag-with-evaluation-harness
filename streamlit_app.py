"""Root Streamlit Community Cloud entry point for the research paper demo."""

import os
import runpy
import sys
from pathlib import Path

REPOSITORY_ROOT = str(Path(__file__).resolve().parent)
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

os.environ["RAG_INLINE_ONLY"] = "true"
os.environ.setdefault("RAG_INLINE_RETRIEVAL_MODE", "hybrid")

runpy.run_path(
    str(Path(REPOSITORY_ROOT) / "rag_hybrid" / "scripts" / "app_streamlit.py"),
    run_name="__main__",
)
