"""Chat-style Streamlit front end for the research paper assistant."""

from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_START_COMMAND = "python -m rag_hybrid.scripts.run_local"

for secret_name in ("GROQ_API_KEY", "GROQ_MODEL"):
    if not os.getenv(secret_name):
        try:
            secret_value = st.secrets.get(secret_name)
        except Exception:
            secret_value = None
        if secret_value:
            os.environ[secret_name] = str(secret_value)

st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: Inter, system-ui, sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #000000 60%, #071a3d 85%, #064d35 100%);
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] { background: rgba(0,0,0,0.35); }
    [data-testid="stChatMessage"] { background: rgba(255,255,255,0.045); border: 1px solid rgba(255,255,255,0.08); }
    .hero-card {
        background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px; padding: 1.5rem 2rem; margin: 0.5rem 0 1.5rem;
        backdrop-filter: blur(12px);
    }
    .metric-card {
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px; padding: 0.8rem 1rem; text-align: center;
    }
    .stButton > button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6); color: white;
        border: none; border-radius: 10px; font-weight: 600;
    }
    .stSpinner > div { color: #6366f1 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def require_authenticated_user() -> None:
    """Require a configured OIDC login and an explicitly allowed email in production."""
    if os.getenv("AUTH_REQUIRED", "false").strip().lower() != "true":
        return

    if os.getenv("RAG_OIDC_CONFIGURED") != "true":
        st.title("Research Paper Assistant")
        st.error(
            "Sign-in is not configured yet. Set the OIDC client credentials and "
            "AUTHORIZED_EMAILS on the hosting service to unlock this workspace."
        )
        st.stop()

    if not st.user.is_logged_in:
        st.title("Research Paper Assistant")
        st.write("Sign in with your organization's identity provider to continue.")
        st.button("Sign in", on_click=st.login, type="primary")
        st.stop()

    user_email = str(getattr(st.user, "email", "") or "").strip().casefold()
    allowed_emails = {
        email.strip().casefold()
        for email in os.getenv("AUTHORIZED_EMAILS", "").split(",")
        if email.strip()
    }
    if not user_email or user_email not in allowed_emails:
        st.title("Research Paper Assistant")
        st.error("Your account is not on this workspace's access list.")
        st.button("Sign out", on_click=st.logout)
        st.stop()

    st.sidebar.caption(f"Signed in as {user_email}")
    st.sidebar.button("Sign out", on_click=st.logout)


require_authenticated_user()


def _raise_for_api_error(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            detail = response.text.strip()
        raise RuntimeError(detail or str(exc)) from exc


def call_api(base_url: str, query: str, top_k: int) -> dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/query",
        json={"query": query, "top_k": top_k},
        timeout=180,
    )
    _raise_for_api_error(response)
    return response.json()


def call_web_search(base_url: str, query: str) -> dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/web-search",
        json={"query": query},
        timeout=30,
    )
    _raise_for_api_error(response)
    return response.json()


def call_readiness(base_url: str) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/ready", timeout=3)
        _raise_for_api_error(response)
        return response.json()
    except (requests.RequestException, RuntimeError, ValueError):
        return None


def call_documents(base_url: str) -> dict[str, Any]:
    response = requests.get(f"{base_url.rstrip('/')}/documents", timeout=10)
    _raise_for_api_error(response)
    return response.json()


def remove_api_documents(
    base_url: str, source_names: list[str] | None = None, clear_all: bool = False
) -> dict[str, Any]:
    response = requests.delete(
        f"{base_url.rstrip('/')}/documents",
        json={"source_names": source_names or [], "clear_all": clear_all},
        headers={"X-Admin-Key": os.getenv("ADMIN_API_KEY", "")},
        timeout=30,
    )
    _raise_for_api_error(response)
    return response.json()


def upload_documents(base_url: str, files: list) -> dict[str, Any]:
    payload = [
        ("files", (file.name, file.getvalue(), file.type or "application/octet-stream"))
        for file in files
    ]
    response = requests.post(
        f"{base_url.rstrip('/')}/documents",
        files=payload,
        headers={"X-Admin-Key": os.getenv("ADMIN_API_KEY", "")},
        timeout=180,
    )
    _raise_for_api_error(response)
    return response.json()


@st.cache_resource(show_spinner="Loading local models…")
def _load_inline_pipeline():
    """Load the RAG components once and cache them."""
    from importlib import import_module

    FaithfulnessEvaluator = import_module(
        "rag_hybrid.src.eval.faithfulness"
    ).FaithfulnessEvaluator
    generator = import_module("rag_hybrid.src.generator.generator")
    extract_claims = generator.extract_claims
    generate_answer = generator.generate_answer
    retriever = import_module("rag_hybrid.src.retriever.hybrid")
    DocumentChunk = retriever.DocumentChunk
    ProductionHybridRAG = retriever.ProductionHybridRAG

    rag = ProductionHybridRAG(vector_db="qdrant", qdrant_path=":memory:")
    evaluator = (
        FaithfulnessEvaluator()
        if os.getenv("GROQ_API_KEY") or os.getenv("groq")
        else None
    )
    demo_chunks = [
        DocumentChunk(
            chunk_id="demo_retrieval_p1_c0",
            doc_id="demo_retrieval",
            source_name="synthetic_demo_paper.pdf",
            page_number=1,
            text="Research question: Does combining dense semantic retrieval and BM25 improve evidence retrieval for a research assistant?",
            parent_text="Research question: Does combining dense semantic retrieval and BM25 improve evidence retrieval for a research assistant? The synthetic demonstration compares retrieval methods.",
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
    return rag, evaluator, generate_answer, extract_claims


def call_inline(query: str, top_k: int) -> dict[str, Any]:
    rag, evaluator, generate_answer, extract_claims = _load_inline_pipeline()
    st.session_state.inline_pipeline_loaded = True
    retrieval_mode = os.getenv("RAG_INLINE_RETRIEVAL_MODE", "hybrid_rerank")
    retrieved = rag.hybrid_retrieve(
        query, top_candidates=10, final_k=top_k, mode=retrieval_mode
    )
    generation = generate_answer(query, retrieved)
    retrieved_map = {
        key: chunk.text
        for chunk in retrieved
        for key in (chunk.chunk_id, f"Chunk_{chunk.chunk_id}")
    }
    claims = extract_claims(generation.answer)
    evaluation = (
        evaluator.evaluate_generation(claims, retrieved_map)
        if generation.mode == "generated" and evaluator is not None
        else None
    )
    chunks_by_citation = {f"Chunk_{chunk.chunk_id}": chunk for chunk in retrieved}
    return {
        "answer": generation.answer,
        "answer_mode": generation.mode,
        "citations": generation.citations,
        "citation_details": [
            {
                "citation": citation,
                "source_name": chunks_by_citation[citation].source_name
                or chunks_by_citation[citation].doc_id,
                "page_number": chunks_by_citation[citation].page_number,
            }
            for citation in generation.citations
            if citation in chunks_by_citation
        ],
        "faithfulness_score": evaluation.faithfulness_score if evaluation else None,
        "violations": evaluation.violations or None if evaluation else None,
        "fallback_reason": generation.fallback_reason,
        "latency_ms": {},
        "retrieved_sources": [
            {
                "citation": f"Chunk_{chunk.chunk_id}",
                "source_name": chunk.source_name or chunk.doc_id,
                "page_number": chunk.page_number,
                "excerpt": chunk.text,
            }
            for chunk in retrieved
        ],
    }


def index_inline_documents(files: list) -> dict[str, Any]:
    """Parse and index selected documents in the cached inline pipeline."""
    from importlib import import_module

    ingestion = import_module("rag_hybrid.src.ingestion")
    chunk_document_segments = ingestion.chunk_document_segments
    read_document_segments = ingestion.read_document_segments

    rag, _, _, _ = _load_inline_pipeline()
    st.session_state.inline_pipeline_loaded = True
    chunks = []
    names = []
    for file in files:
        _, segments = read_document_segments(file.name, file.getvalue())
        document_chunks = chunk_document_segments(file.name, segments)
        if not document_chunks:
            raise ValueError(f"{file.name}: document contains no indexable text")
        names.append(file.name)
        chunks.extend(document_chunks)
    added = rag.replace_sources(chunks)
    return {"files": names, "chunks_indexed": len(added), "total_chunks": len(rag.corpus_chunks)}


def render_assistant_message(message: dict[str, Any], show_violations: bool) -> None:
    result = message.get("result")
    if not result:
        st.error(message.get("error", "The request failed."))
        if message.get("api_unavailable"):
            st.info("Start the API in a terminal and leave it running while you use the UI:")
            st.code(API_START_COMMAND, language="powershell")
        return

    if result.get("answer_mode") == "extractive":
        fallback_reason = result.get("fallback_reason")
        if fallback_reason == "provider_error":
            st.info(
                "Groq couldn't complete the answer request. Showing a cited answer from the "
                "retrieved paper text. Check the Streamlit app logs for the provider error."
            )
        elif fallback_reason == "citation_validation_error":
            st.info(
                "Groq's answer still lacked valid citations after a retry. Showing a cited "
                "answer selected from the retrieved paper text."
            )
        elif fallback_reason == "no_api_key" or not (
            os.getenv("GROQ_API_KEY") or os.getenv("groq")
        ):
            st.info(
                "No answer-generation model is configured. This cited answer is selected from "
                "the paper text that matched your question. Set GROQ_API_KEY for a generated answer."
            )
        else:
            st.info(
                "A validated model answer wasn't available. Showing a cited answer selected "
                "from the retrieved paper text. Check the Streamlit app logs for details."
            )
    st.markdown(result.get("answer", "No answer returned."))
    if message.get("source_mode") == "Web search":
        sources = result.get("sources", [])
        st.caption(f"Web search · {len(sources)} source(s) · {message['elapsed']:.2f}s")
        for index, source in enumerate(sources, start=1):
            title = source.get("title") or source.get("url") or f"Source {index}"
            st.markdown(f"{index}. [{title}]({source.get('url', '')})")
            if source.get("content"):
                st.caption(source["content"])
        return

    faithfulness = result.get("faithfulness_score")
    citations = result.get("citations", [])
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Evidence support", f"{faithfulness:.0%}" if faithfulness is not None else "N/A")
    metric2.metric("Citations", len(citations))
    metric3.metric("Answer time", f"{message['elapsed']:.2f}s")

    details = result.get("citation_details") or []
    if details:
        labels = []
        for item in details:
            label = item.get("source_name") or item.get("citation", "Source")
            if item.get("page_number") is not None:
                label += f", p. {item['page_number']}"
            if item.get("citation"):
                label += f" ({item['citation']})"
            labels.append(label)
        st.markdown("**📎 Paper sources**")
        for label in labels:
            st.markdown(f"- {label}")
    elif citations:
        st.caption("Citations: " + " · ".join(citations))

    retrieved_sources = result.get("retrieved_sources") or []
    if retrieved_sources:
        with st.expander(
            f"Retrieved paper passages ({len(retrieved_sources)})",
            expanded=result.get("answer_mode") == "extractive",
        ):
            for source in retrieved_sources:
                label = source.get("source_name") or "Paper"
                if source.get("page_number") is not None:
                    label += f", p. {source['page_number']}"
                if source.get("citation"):
                    label += f" · {source['citation']}"
                st.markdown(f"**{label}**")
                st.text(source.get("excerpt", ""))

    timings = result.get("latency_ms") or {}
    if timings:
        with st.expander("Latency details"):
            st.caption(" · ".join(f"{name}: {value:.0f} ms" for name, value in timings.items()))

    if show_violations and result.get("violations"):
        with st.expander("Citation checks", expanded=False):
            for violation in result["violations"]:
                st.warning(violation)


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    inline_only = os.getenv("RAG_INLINE_ONLY", "false").strip().lower() == "true"
    if inline_only:
        api_url = ""
        backend = "Inline demo"
        search_mode = "Indexed papers"
        st.caption("Free demo mode · papers are kept in this running session only")
    else:
        api_url = st.text_input(
            "API base URL",
            value=os.getenv("RAG_API_URL", "http://localhost:8000"),
            help="URL of the FastAPI backend",
            disabled=os.getenv("RAG_LOCK_API_URL", "false").strip().lower() == "true",
        )
        backend_options = (
            ["API (FastAPI)"]
            if os.getenv("RAG_ALLOW_INLINE_DEMO", "true").strip().lower() != "true"
            else ["API (FastAPI)", "Inline demo"]
        )
        default_backend = os.getenv("RAG_DEFAULT_BACKEND", backend_options[0])
        if default_backend not in backend_options:
            default_backend = backend_options[0]
        backend = st.radio(
            "Backend",
            backend_options,
            index=backend_options.index(default_backend),
            help="API mode keeps indexed papers in the persistent vector database. Inline demo is temporary and uses synthetic content.",
        )
        search_mode = st.radio(
            "Search in",
            ["Indexed papers", "Web search"],
            horizontal=True,
            help="Web search sends the question to Tavily only after you select this option and submit it.",
        )
    top_k = st.slider("Retrieved paper chunks", min_value=1, max_value=10, value=5)
    show_violations = st.checkbox("Show citation checks", value=False)
    st.divider()
    st.markdown("### 📚 Your papers")
    if backend == "API (FastAPI)":
        library_status = call_readiness(api_url)
        if library_status:
            try:
                document_data = call_documents(api_url)
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                document_data = None
                st.warning(f"Could not load the indexed paper list: {exc}")

            st.caption(
                f"Library: {library_status.get('indexed_chunks', 0)} chunks from "
                f"{library_status.get('indexed_sources', 0)} source(s)."
            )
            if not library_status.get("indexed_chunks"):
                st.info("No papers are indexed yet. Upload a text-based PDF, DOCX, Markdown, or text file below.")
            elif library_status.get("model_warmup_status") == "pending":
                st.caption("Retrieval models are warming up in the background.")

            if document_data and document_data["documents"]:
                documents = document_data["documents"]
                source_names = [document["source_name"] for document in documents]
                chunk_counts = {
                    document["source_name"]: document["chunks"] for document in documents
                }
                selected_sources = st.multiselect(
                    "Remove selected papers",
                    options=source_names,
                    format_func=lambda name: f"{name} · {chunk_counts[name]} chunks",
                    key=f"remove_sources_{len(source_names)}_{document_data['total_chunks']}",
                )
                if st.button("Remove selected", disabled=not selected_sources):
                    try:
                        removed = remove_api_documents(api_url, source_names=selected_sources)
                        st.session_state.library_action_notice = (
                            f"Removed {removed['chunks_removed']} chunks from "
                            f"{len(removed['sources_removed'])} paper(s). "
                            f"Library total: {removed['total_chunks']} chunks."
                        )
                        st.rerun()
                    except (requests.RequestException, RuntimeError, ValueError) as exc:
                        st.error(f"Could not remove selected papers: {exc}")

                confirm_key = f"confirm_clear_library_{document_data['total_chunks']}"
                confirm_clear = st.checkbox(
                    "Confirm clearing every indexed paper",
                    key=confirm_key,
                )
                if st.button(
                    "Clear entire library",
                    disabled=not confirm_clear,
                    type="secondary",
                ):
                    try:
                        removed = remove_api_documents(api_url, clear_all=True)
                        st.session_state.library_action_notice = (
                            f"Cleared {removed['chunks_removed']} chunks from "
                            f"{len(removed['sources_removed'])} paper(s)."
                        )
                        st.rerun()
                    except (requests.RequestException, RuntimeError, ValueError) as exc:
                        st.error(f"Could not clear the library: {exc}")

            if "library_action_notice" in st.session_state:
                st.success(st.session_state.pop("library_action_notice"))
        else:
            st.warning("FastAPI is unavailable. Start the local services to view and upload your library.")
            st.code(API_START_COMMAND, language="powershell")
    elif st.session_state.get("inline_pipeline_loaded"):
        inline_rag, _, _, _ = _load_inline_pipeline()
        inline_counts: dict[str, int] = {}
        for chunk in inline_rag.corpus_chunks:
            source_name = chunk.source_name or chunk.doc_id
            if source_name != "synthetic_demo_paper.pdf":
                inline_counts[source_name] = inline_counts.get(source_name, 0) + 1
        st.caption(
            f"Temporary inline uploads: {sum(inline_counts.values())} chunks from "
            f"{len(inline_counts)} paper(s)."
        )
        if inline_counts:
            selected_inline = st.multiselect(
                "Remove selected inline papers",
                options=sorted(inline_counts),
                format_func=lambda name: f"{name} · {inline_counts[name]} chunks",
            )
            if st.button("Remove selected inline papers", disabled=not selected_inline):
                removed = inline_rag.remove_sources(selected_inline)
                st.session_state.library_action_notice = (
                    f"Removed {len(removed)} inline chunks from {len(selected_inline)} paper(s)."
                )
                st.rerun()
            confirm_inline_clear = st.checkbox("Confirm clearing all inline uploads")
            if st.button("Clear inline uploads", disabled=not confirm_inline_clear):
                removed = inline_rag.remove_sources(list(inline_counts))
                st.session_state.library_action_notice = (
                    f"Removed {len(removed)} inline chunks from {len(inline_counts)} paper(s)."
                )
                st.rerun()
    else:
        st.caption("Inline demo uploads are temporary and reset when the app restarts.")

    if "library_action_notice" in st.session_state:
        st.success(st.session_state.pop("library_action_notice"))

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, Markdown, or text",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
    )
    index_uploads = st.button(
        "Index uploaded papers",
        disabled=not uploaded_files,
        use_container_width=True,
    )
    st.divider()
    st.caption("Hybrid dense + BM25 retrieval · source citations · evidence checks")

st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin:0;font-size:2rem;background:linear-gradient(90deg,#a5b4fc,#c4b5fd);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            📄 Research Paper Assistant
        </h1>
        <p style="color:#cbd5e1;margin:0.5rem 0 0;">
            Chat with your indexed papers or switch to web search for broader research.
            Paper answers include citations and page numbers.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if index_uploads and uploaded_files:
    try:
        with st.spinner("Reading, chunking, and indexing papers…"):
            if backend == "API (FastAPI)":
                indexed = upload_documents(api_url, uploaded_files)
            else:
                indexed = index_inline_documents(uploaded_files)
        if indexed["chunks_indexed"]:
            st.sidebar.success(
                f"Added {indexed['chunks_indexed']} chunks from {len(indexed['files'])} file(s). "
                f"Library total: {indexed['total_chunks']} chunks."
            )
        else:
            st.sidebar.info(
                f"No new chunks were needed; the uploaded content is already indexed. "
                f"Library total: {indexed['total_chunks']} chunks."
            )
    except requests.ConnectionError:
        st.sidebar.error(f"Cannot connect to FastAPI at {api_url.rstrip('/')}")
        st.sidebar.code(API_START_COMMAND, language="powershell")
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        st.sidebar.error(f"Indexing failed: {exc}")
    except Exception as exc:
        st.sidebar.error(f"Indexing failed: {exc}")

if not st.session_state.messages:
    st.markdown(
        "Ask a question to get started. Try *“What problem does this paper address?”* or "
        "*“What evidence supports its main result?”*"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_assistant_message(message, show_violations)

placeholder = "Search the web…" if search_mode == "Web search" else "Ask about your papers…"
prompt = st.chat_input(placeholder)

if prompt and prompt.strip():
    query = prompt.strip()
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    new_message: dict[str, Any] = {"role": "assistant", "source_mode": search_mode}
    if search_mode == "Web search" and backend != "API (FastAPI)":
        new_message["error"] = "Web search requires the API (FastAPI) backend. Select it in Settings."
    else:
        if search_mode == "Indexed papers" and backend == "API (FastAPI)":
            readiness = call_readiness(api_url)
            if readiness and not readiness.get("models_ready", False):
                if readiness.get("model_warmup_status") == "failed":
                    st.info("Background model warmup failed. This request may load models on demand; check the API terminal for details.")
                else:
                    st.info("Retrieval models are still warming up. The first answer may take longer.")
        started = time.perf_counter()
        try:
            with st.spinner("Searching and preparing your answer…"):
                if search_mode == "Web search":
                    response = call_web_search(api_url, query)
                elif backend == "API (FastAPI)":
                    response = call_api(api_url, query, top_k)
                else:
                    response = call_inline(query, top_k)
            new_message["result"] = response
        except requests.ConnectionError as exc:
            new_message["error"] = f"Cannot connect to FastAPI at {api_url.rstrip('/')}: {exc}"
            new_message["api_unavailable"] = True
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            new_message["error"] = str(exc)
        except Exception as exc:
            new_message["error"] = str(exc)
        new_message["elapsed"] = time.perf_counter() - started

    st.session_state.messages.append(new_message)
    with st.chat_message("assistant"):
        render_assistant_message(new_message, show_violations)
