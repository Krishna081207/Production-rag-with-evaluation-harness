# conftest.py – root-level pytest configuration
"""Ensure the repo root and scripts/ directory are on sys.path.

This allows tests to import from `scripts.*` without needing an installed
package, and also ensures `rag_hybrid.*` imports work correctly regardless
of whether the package is installed in editable mode.
"""
import os
import sys
from pathlib import Path

# Ensure test suite runs offline with in-memory Qdrant unless testing Pinecone explicitly
os.environ.setdefault("VECTOR_DB", "qdrant")

# Repo root (where this conftest.py lives)
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

