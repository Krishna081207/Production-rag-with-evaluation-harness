"""Streamlit front-end for research paper understanding.

Calls the FastAPI backend at http://localhost:8000 (or the URL defined by
``RAG_API_URL`` environment variable).  When the API is not reachable, it
falls back to running the pipeline inline so the UI always works in demo mode.
"""

import os
import time
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: Inter, system-ui, sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e2e8f0;
    }

    /* Hero card */
    .hero-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(12px);
    }

    /* Answer box */
    .answer-box {
        background: rgba(99, 102, 241, 0.12);
        border-left: 4px solid #6366f1;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        line-height: 1.75;
    }

    /* Metric card */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        text-align: center;
    }

    /* Citation chip */
    .chip {
        display: inline-block;
        background: rgba(99,102,241,0.25);
        border: 1px solid rgba(99,102,241,0.5);
        border-radius: 999px;
        padding: 2px 12px;
        font-size: 0.78rem;
        margin: 2px 4px;
        color: #c7d2fe;
    }

    /* Violation warning */
    .violation {
        background: rgba(239,68,68,0.12);
        border-left: 4px solid #ef4444;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        color: #fca5a5;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(0,0,0,0.35);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.6rem;
        font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    .stTextInput > div > input {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 10px;
        color: #e2e8f0;
    }

    .stSpinner > div { color: #6366f1 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar – configuration
# ---------------------------------------------------------------------------
#upload documents
st.markdown("## 📄 Upload research sources")
stfile = st.file_uploader(
    "Upload papers or supporting documents",
    type=["pdf", "txt", "md", "docx"],
    accept_multiple_files=True,
)
index_uploads = st.button(
    "Index uploaded documents",
    disabled=not stfile,
    help="Parse, chunk, embed, and store the selected documents for search.",
)
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    api_url = st.text_input(
        "API base URL",
        value=os.getenv("RAG_API_URL", "http://localhost:8000"),
        help="URL of the running FastAPI backend",
    )

    top_k = st.slider("Retrieved chunks (top-k)", min_value=1, max_value=10, value=5)
    show_violations = st.checkbox("Show citation violations", value=True)
    st.divider()

    st.markdown("### 🧪 Mode")
    mode = st.radio(
        "Backend",
        ["API (FastAPI)", "Inline (no server)"],
        index=0,
        help="Inline mode uses an in-memory demo corpus; uploaded papers are temporary. Use API mode for persistent storage.",
    )

    st.divider()
    st.markdown(
        "<small style='color:#94a3b8;'>Research Paper Assistant · Hybrid Dense+BM25+RRF · "
        "Cross-Encoder reranker · NLI faithfulness</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin:0;font-size:2rem;background:linear-gradient(90deg,#a5b4fc,#c4b5fd);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            📄 Research Paper Assistant
        </h1>
        <p style="color:#94a3b8;margin-top:0.5rem;">
            Ask about a paper’s research question, contributions, methods, results, and limitations.
            Answers link back to retrieved source chunks and PDF pages.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Query history (session state)
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Main query input
# ---------------------------------------------------------------------------
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "Your question",
        placeholder="e.g. What are the paper’s key findings?",
        label_visibility="collapsed",
    )
with col_btn:
    submit = st.button("Ask →", use_container_width=True)

# ---------------------------------------------------------------------------
# Helper: call FastAPI backend
# ---------------------------------------------------------------------------

def call_api(q: str) -> dict:
    """POST to /query endpoint and return the JSON response."""
    resp = requests.post(
        f"{api_url.rstrip('/')}/query",
        json={"query": q},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def upload_documents(files: list) -> dict:
    """Send selected files to the API ingestion pipeline."""
    payload = [
        ("files", (file.name, file.getvalue(), file.type or "application/octet-stream"))
        for file in files
    ]
    resp = requests.post(
        f"{api_url.rstrip('/')}/documents",
        files=payload,
        headers={"X-Admin-Key": os.getenv("ADMIN_API_KEY", "")},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()

# ---------------------------------------------------------------------------
# Helper: inline pipeline (no server required)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models…")
def _load_inline_pipeline():
    """Load the RAG components once and cache them."""
    import re

    from rag_hybrid.src.eval.faithfulness import FaithfulnessEvaluator
    from rag_hybrid.src.generator.generator import generate_answer
    from rag_hybrid.src.retriever.hybrid import DocumentChunk, ProductionHybridRAG

    rag = ProductionHybridRAG(vector_db="qdrant", qdrant_path=":memory:")
    evaluator = FaithfulnessEvaluator()

    # Seed a tiny demo corpus so the UI works out of the box
    demo_chunks = [
        DocumentChunk(
            chunk_id="demo_retrieval_p1_c0",
            doc_id="demo_retrieval",
            source_name="synthetic_demo_paper.pdf",
            page_number=1,
            text="Research question: Does combining dense semantic retrieval and BM25 improve evidence retrieval for a research assistant?",
            parent_text="Research question: Does combining dense semantic retrieval and BM25 improve evidence retrieval for a research assistant? The synthetic demonstration compares each method on a small illustrative corpus.",
        ),
        DocumentChunk(
            chunk_id="demo_retrieval_p2_c0",
            doc_id="demo_retrieval",
            source_name="synthetic_demo_paper.pdf",
            page_number=2,
            text="Method: The synthetic study compares BM25, dense retrieval, and reciprocal rank fusion using Recall@5 on manually labeled questions.",
            parent_text="Method: The synthetic study compares BM25, dense retrieval, and reciprocal rank fusion using Recall@5 on manually labeled questions. This example contains no real experimental results.",
        ),
        DocumentChunk(
            chunk_id="demo_retrieval_p3_c0",
            doc_id="demo_retrieval",
            source_name="synthetic_demo_paper.pdf",
            page_number=3,
            text="Limitation: Results from a small synthetic example cannot establish how retrieval methods perform across scientific fields or full-length papers.",
            parent_text="Limitation: Results from a small synthetic example cannot establish how retrieval methods perform across scientific fields or full-length papers.",
        ),
    ]
    rag.index_documents(demo_chunks)
    return rag, evaluator, generate_answer, re


def call_inline(q: str) -> dict:
    rag, evaluator, generate_answer, _re = _load_inline_pipeline()
    retrieved = rag.hybrid_retrieve(q, top_candidates=10, final_k=top_k)
    gen_res = generate_answer(q, retrieved)
    retrieved_map = {
        key: c.parent_text
        for c in retrieved
        for key in (c.chunk_id, f"Chunk_{c.chunk_id}")
    }
    claims = []
    abstention = "The provided documentation does not contain sufficient evidence to answer this inquiry."
    if gen_res.answer != abstention:
        for sentence in _re.split(r"(?<=[.!?])\s+", gen_res.answer.strip()):
            sentence = sentence.strip()
            if not sentence:
                continue
            cited_ids = _re.findall(r"\[Chunk_([^\]]+)\]", sentence)
            claim = _re.sub(r"\[Chunk_[^\]]+\]", "", sentence).strip()
            claims.extend(
                {"claim": claim, "citation": f"Chunk_{chunk_id}"}
                for chunk_id in cited_ids
            )
            if not cited_ids and claim:
                claims.append({"claim": claim, "citation": ""})
    eval_res = evaluator.evaluate_generation(claims, retrieved_map)
    chunks_by_citation = {f"Chunk_{chunk.chunk_id}": chunk for chunk in retrieved}
    return {
        "answer": gen_res.answer,
        "citations": gen_res.citations,
        "citation_details": [
            {
                "citation": citation,
                "source_name": chunks_by_citation[citation].source_name or chunks_by_citation[citation].doc_id,
                "page_number": chunks_by_citation[citation].page_number,
            }
            for citation in gen_res.citations
            if citation in chunks_by_citation
        ],
        "faithfulness_score": eval_res.faithfulness_score,
        "violations": eval_res.violations or None,
    }


def index_inline_documents(files: list) -> dict[str, Any]:
    """Parse and index selected documents in the cached inline pipeline."""
    from rag_hybrid.src.ingestion import chunk_document_segments, read_document_segments

    rag, _, _, _ = _load_inline_pipeline()
    all_chunks = []
    indexed_files = []
    for file in files:
        _, segments = read_document_segments(file.name, file.getvalue())
        chunks = chunk_document_segments(file.name, segments)
        if not chunks:
            raise ValueError(f"{file.name}: document is empty")
        indexed_files.append(file.name)
        all_chunks.extend(chunks)
    added = rag.replace_sources(all_chunks)
    return {
        "files": indexed_files,
        "chunks_indexed": len(added),
        "total_chunks": len(rag.corpus_chunks),
    }


if index_uploads and stfile:
    with st.spinner("Parsing, chunking, embedding, and indexing…"):
        try:
            upload_result = (
                upload_documents(stfile)
                if mode == "API (FastAPI)"
                else index_inline_documents(stfile)
            )
            st.success(
                f"Indexed {upload_result['chunks_indexed']} chunks from "
                f"{len(upload_result['files'])} document(s). "
                f"Total indexed chunks: {upload_result['total_chunks']}."
            )
        except requests.RequestException as exc:
            st.error(f"Document indexing failed: {exc}")
        except (KeyError, ValueError) as exc:
            st.error(f"Document indexing failed: {exc}")
        except Exception as exc:
            st.error(f"Document indexing failed: {exc}")


# ---------------------------------------------------------------------------
# Process query
# ---------------------------------------------------------------------------
if submit and query.strip():
    with st.spinner("Retrieving and generating…"):
        t0 = time.perf_counter()
        error = None
        result = None
        try:
            if mode == "API (FastAPI)":
                result = call_api(query.strip())
            else:
                result = call_inline(query.strip())
        except Exception as exc:
            error = str(exc)
        elapsed = time.perf_counter() - t0

    if error:
        st.error(f"❌ Error: {error}")
        if mode == "API (FastAPI)":
            st.info("💡 Make sure the API server is running: `uvicorn rag_hybrid.src.api.app:app --reload`")
    else:
        assert result is not None
        # Save to history
        st.session_state.history.insert(
            0,
            {"query": query, "result": result, "elapsed": elapsed},
        )

        # ---- Display result ----
        st.markdown("---")

        # Answer
        with st.container(border=True):
            st.markdown(result["answer"])

        # Metrics row
        c1, c2, c3 = st.columns(3)
        faith = result.get("faithfulness_score")
        faith_color = (
            "#22c55e" if faith is not None and faith >= 0.8
            else "#f59e0b" if faith is not None and faith >= 0.5
            else "#94a3b8" if faith is None
            else "#ef4444"
        )
        faith_text = f"{faith:.0%}" if faith is not None else "N/A"
        with c1:
            st.markdown(
                f'<div class="metric-card"><div style="font-size:1.8rem;font-weight:700;color:{faith_color};">'
                f'{faith_text}</div><div style="color:#94a3b8;font-size:0.85rem;">Evidence support</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            n_cit = len(result.get("citations", []))
            st.markdown(
                f'<div class="metric-card"><div style="font-size:1.8rem;font-weight:700;color:#a5b4fc;">'
                f'{n_cit}</div><div style="color:#94a3b8;font-size:0.85rem;">Citations</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="metric-card"><div style="font-size:1.8rem;font-weight:700;color:#c4b5fd;">'
                f'{elapsed:.2f}s</div><div style="color:#94a3b8;font-size:0.85rem;">Latency</div></div>',
                unsafe_allow_html=True,
            )

        # Citations
        if result.get("citations"):
            st.markdown("**📎 Sources**")
            details = result.get("citation_details") or []
            if details:
                labels = []
                for item in details:
                    label = item["source_name"]
                    if item.get("page_number") is not None:
                        label += f", p. {item['page_number']}"
                    labels.append(f"{label} ({item['citation']})")
                st.caption(" · ".join(labels))
            else:
                st.caption(" · ".join(str(c) for c in result["citations"]))

        # Violations
        if show_violations and result.get("violations"):
            st.markdown("**⚠️ Citation violations**")
            for v in result["violations"]:
                st.warning(v)

# ---------------------------------------------------------------------------
# Query history
# ---------------------------------------------------------------------------
# if st.session_state.history:
#     st.markdown("---")
#     with st.expander(f"🕑 Query history ({len(st.session_state.history)} queries)", expanded=False):
#         for i, h in enumerate(st.session_state.history):
#             st.markdown(
#                 f"**{i+1}. {h['query']}** — "
#                 f"faithfulness: `{h['result'].get('faithfulness_score', 0):.0%}` | "
#                 f"latency: `{h['elapsed']:.2f}s`"
#             )
