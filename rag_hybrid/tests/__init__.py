"""Comprehensive unit + integration tests for rag_hybrid.

Test plan
---------
1. test_retriever.py  – hybrid retrieval, RRF fusion, edge-cases
2. test_generator.py  – generation contract, citation extraction, empty-chunk fallback
3. test_evaluator.py  – faithfulness scorer, violations, edge-cases
4. test_harness.py    – MMR score, hallucination-rate helpers
5. test_api.py        – FastAPI /health and /query endpoints (TestClient, no real LLM)
"""
