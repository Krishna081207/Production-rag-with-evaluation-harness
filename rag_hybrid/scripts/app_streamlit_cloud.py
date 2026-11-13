"""Streamlit Community Cloud entry point for the free, in-memory demo."""

import os
import runpy
import sys
from pathlib import Path

os.environ["RAG_INLINE_ONLY"] = "true"
os.environ.setdefault("RAG_INLINE_RETRIEVAL_MODE", "hybrid")

# Streamlit Cloud can start a nested entry point with only its script directory
# on sys.path. Add the repository root so the rag_hybrid package is importable.
repository_root = Path(__file__).resolve().parents[2]
repository_root_text = str(repository_root)
if repository_root_text not in sys.path:
    sys.path.insert(0, repository_root_text)

# Streamlit executes this file as a script, so importing another Streamlit entry
# point as a package can leave partially initialized package modules behind.
# Execute the shared UI file directly instead; this also keeps existing Cloud
# deployments that still point at this legacy nested path working.
runpy.run_path(
    str(repository_root / "rag_hybrid" / "scripts" / "app_streamlit.py"),
    run_name="__main__",
)
