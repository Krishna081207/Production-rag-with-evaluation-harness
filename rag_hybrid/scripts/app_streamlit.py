"""Streamlit front-end for the rag_hybrid RAG pipeline.

Calls the FastAPI backend at http://localhost:8000 (or the URL defined by
``RAG_API_URL`` environment variable).  When the API is not reachable, it
falls back to running the pipeline inline so the UI always works in demo mode.
"""

import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🔍 Production RAG – Hybrid Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
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
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/openai.svg",
        width=40,
    )
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
        help="Inline mode runs the pipeline directly in this process — useful when the API server is not running.",
    )

    vdb_choice = st.selectbox(
        "Vector Database (Inline mode)",
        ["pinecone", "qdrant"],
        index=0 if os.getenv("VECTOR_DB", "pinecone").lower() == "pinecone" else 1,
        help="Select Pinecone for cloud vector index or Qdrant for local in-memory search.",
    )

    st.divider()
    st.markdown(
        "<small style='color:#94a3b8;'>Production RAG · Hybrid Dense+BM25+RRF · "
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
            🔍 Production RAG · Hybrid Search
        </h1>
        <p style="color:#94a3b8;margin-top:0.5rem;">
            Dense (BGE) + BM25 → RRF fusion → Cross-Encoder reranking → Groq LLM
            → NLI faithfulness evaluation
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
        placeholder="e.g. What is Python used for?",
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
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Helper: inline pipeline (no server required)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models…")
def _load_inline_pipeline(vector_db: str = "pinecone"):
    """Load the RAG components once and cache them."""
    import re

    from rag_hybrid.src.eval.faithfulness import FaithfulnessEvaluator
    from rag_hybrid.src.generator.generator import generate_answer
    from rag_hybrid.src.retriever.hybrid import DocumentChunk, ProductionHybridRAG

    rag = ProductionHybridRAG(vector_db=vector_db)
    evaluator = FaithfulnessEvaluator()

    # Seed a tiny demo corpus so the UI works out of the box
    demo_chunks = [
        DocumentChunk(
            chunk_id="demo_c1",
            doc_id="demo",
            text="Python is a high-level, dynamically typed programming language popular in data science and web development.",
            parent_text="Python is a high-level, dynamically typed programming language popular in data science and web development.",
        ),
        DocumentChunk(
            chunk_id="demo_c2",
            doc_id="demo",
            text="FastAPI is a modern, fast web framework for building APIs with Python based on type hints.",
            parent_text="FastAPI is a modern, fast web framework for building APIs with Python based on type hints.",
        ),
        DocumentChunk(
            chunk_id="demo_c3",
            doc_id="demo",
            text="Pinecone and Qdrant are vector databases designed for high-performance similarity search.",
            parent_text="Pinecone and Qdrant are vector databases designed for high-performance similarity search.",
        ),
    ]
    rag.index_documents(demo_chunks)
    return rag, evaluator, generate_answer, re


def call_inline(q: str, vector_db: str = "pinecone") -> dict:
    rag, evaluator, generate_answer, _re = _load_inline_pipeline(vector_db)
    retrieved = rag.hybrid_retrieve(q, top_candidates=10, final_k=top_k)
    gen_res = generate_answer(q, retrieved)
    retrieved_map = {c.chunk_id: c.parent_text for c in retrieved}
    match = _re.search(r"\[Chunk_(?P<cid>[^\]]+)\]", gen_res.answer)
    citation = f"Chunk_{match.group('cid')}" if match else ""
    claim = gen_res.answer.replace(match.group(0), "").strip() if match else gen_res.answer
    claims = [{"claim": claim, "citation": citation}]
    eval_res = evaluator.evaluate_generation(claims, retrieved_map)
    return {
        "answer": gen_res.answer,
        "citations": gen_res.citations,
        "faithfulness_score": eval_res.faithfulness_score,
        "violations": eval_res.violations or None,
    }


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
                result = call_inline(query.strip(), vector_db=vdb_choice)
        except Exception as exc:
            error = str(exc)
        elapsed = time.perf_counter() - t0

    if error:
        st.error(f"❌ Error: {error}")
        if mode == "API (FastAPI)":
            st.info("💡 Make sure the API server is running: `uvicorn rag_hybrid.src.api.app:app --reload`")
    else:
        # Save to history
        st.session_state.history.insert(
            0,
            {"query": query, "result": result, "elapsed": elapsed},
        )

        # ---- Display result ----
        st.markdown("---")

        # Answer
        st.markdown(
            f'<div class="answer-box">{result["answer"]}</div>',
            unsafe_allow_html=True,
        )

        # Metrics row
        c1, c2, c3 = st.columns(3)
        faith = result.get("faithfulness_score", 0.0)
        faith_color = "#22c55e" if faith >= 0.8 else "#f59e0b" if faith >= 0.5 else "#ef4444"
        with c1:
            st.markdown(
                f'<div class="metric-card"><div style="font-size:1.8rem;font-weight:700;color:{faith_color};">'
                f'{faith:.0%}</div><div style="color:#94a3b8;font-size:0.85rem;">Faithfulness</div></div>',
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
            st.markdown("**📎 Citations**")
            chips = "".join(f'<span class="chip">{c}</span>' for c in result["citations"])
            st.markdown(chips, unsafe_allow_html=True)

        # Violations
        if show_violations and result.get("violations"):
            st.markdown("**⚠️ Citation violations**")
            for v in result["violations"]:
                st.markdown(f'<div class="violation">⚠️ {v}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Query history
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.markdown("---")
    with st.expander(f"🕑 Query history ({len(st.session_state.history)} queries)", expanded=False):
        for i, h in enumerate(st.session_state.history):
            st.markdown(
                f"**{i+1}. {h['query']}** — "
                f"faithfulness: `{h['result'].get('faithfulness_score', 0):.0%}` | "
                f"latency: `{h['elapsed']:.2f}s`"
            )
