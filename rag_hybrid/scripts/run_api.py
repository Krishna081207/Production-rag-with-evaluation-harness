# Script to launch the FastAPI server for rag_hybrid
"""Run the RAG API.

Usage:
    python scripts/run_api.py
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "rag_hybrid.src.api.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,
    )
