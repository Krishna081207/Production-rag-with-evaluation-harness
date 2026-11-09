"""Ensure the checkout root is importable when tests run from the package tree."""
import os
import sys
from pathlib import Path

# Ensure test suite runs offline with in-memory Qdrant unless testing Pinecone explicitly
os.environ.setdefault("VECTOR_DB", "qdrant")

# Checkout root (one level above the rag_hybrid package).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

