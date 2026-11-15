"""Chat-style Streamlit front end for the research paper assistant."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

from rag_hybrid.src.generator.generator import CITATION_RE, without_citation_markers
from rag_hybrid.src.paper_qa import PAPER_IMPLICATIONS_QUESTION, PAPER_OVERVIEW_QUESTION, PaperQA

load_dotenv()
logger = logging.getLogger(__name__)
API_START_COMMAND = "python -m rag_hybrid.scripts.run_local"
INLINE_PIPELINE_CACHE_VERSION = "paper-qa-v1"
DEMO_PAPER = "synthetic_demo_paper.pdf"

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


@st.cache_resource
def _inline_retrieval_models():
    from rag_hybrid.src.retriever.hybrid import EMBEDDING_MODEL, RERANKER_MODEL, RetrievalModels

    return RetrievalModels(EMBEDDING_MODEL, RERANKER_MODEL)


def _load_inline_pipeline(*, include_demo: bool = True) -> PaperQA:
    """Share model weights while keeping each session's papers and answers together."""
    if INLINE_PIPELINE_CACHE_VERSION in st.session_state:
        return st.session_state[INLINE_PIPELINE_CACHE_VERSION]

    from rag_hybrid.src.retriever.hybrid import DocumentChunk, ProductionHybridRAG

    rag = ProductionHybridRAG(
        vector_db="qdrant", qdrant_path=":memory:", models=_inline_retrieval_models()
    )
    demo_chunks = [
        DocumentChunk(
            chunk_id="demo_retrieval_p1_c0",
            doc_id="demo_retrieval",
            source_name=DEMO_PAPER,
            page_number=1,
            text="Research question: Does combining dense semantic retrieval and BM25 improve evidence retrieval for a research assistant?",
            parent_text="Research question: Does combining dense semantic retrieval and BM25 improve evidence retrieval for a research assistant? The synthetic demonstration compares retrieval methods.",
        ),
        DocumentChunk(
            chunk_id="demo_retrieval_p2_c0",
            doc_id="demo_retrieval",
            source_name=DEMO_PAPER,
            page_number=2,
            text="Method: The synthetic study compares BM25, dense retrieval, and reciprocal rank fusion using Recall@5 on manually labeled questions.",
            parent_text="Method: The synthetic study compares BM25, dense retrieval, and reciprocal rank fusion using Recall@5 on manually labeled questions. This example contains no real experimental results.",
        ),
        DocumentChunk(
            chunk_id="demo_retrieval_p3_c0",
            doc_id="demo_retrieval",
            source_name=DEMO_PAPER,
            page_number=3,
            text="Limitation: Results from a small synthetic example cannot establish how retrieval methods perform across scientific fields or full-length papers.",
            parent_text="Limitation: Results from a small synthetic example cannot establish how retrieval methods perform across scientific fields or full-length papers.",
        ),
    ]
    if include_demo:
        rag.index_documents(demo_chunks)
    paper_qa = PaperQA(rag, _evaluate_inline)
    st.session_state[INLINE_PIPELINE_CACHE_VERSION] = paper_qa
    return paper_qa


@st.cache_resource(show_spinner=False)
def _load_inline_evaluator():
    from rag_hybrid.src.eval.faithfulness import FaithfulnessEvaluator

    return FaithfulnessEvaluator()


def _evaluate_inline(claims: list[dict[str, str]], evidence: dict[str, str]):
    try:
        return _load_inline_evaluator().evaluate_generation(claims, evidence)
    except Exception:
        logger.exception("Could not run citation checks for the inline answer")
        return None


def call_inline(query: str, top_k: int) -> dict[str, Any]:
    retrieval_mode = os.getenv("RAG_INLINE_RETRIEVAL_MODE", "hybrid_rerank")
    return _load_inline_pipeline().answer(query, top_k, mode=retrieval_mode)


def index_inline_documents(files: list) -> dict[str, Any]:
    """Parse and index selected documents in the cached inline pipeline."""
    from importlib import import_module

    ingestion = import_module("rag_hybrid.src.ingestion")
    chunk_document_segments = ingestion.chunk_document_segments
    read_document_segments = ingestion.read_document_segments

    chunks = []
    names = []
    for file in files:
        _, segments = read_document_segments(file.name, file.getvalue())
        document_chunks = chunk_document_segments(file.name, segments)
        if not document_chunks:
            raise ValueError(f"{file.name}: document contains no indexable text")
        names.append(file.name)
        chunks.extend(document_chunks)
    rag = _load_inline_pipeline(include_demo=False).rag
    added = rag.replace_sources(chunks)
    if any(chunk.doc_id == "demo_retrieval" for chunk in rag.corpus_chunks):
        rag.remove_sources([DEMO_PAPER])
    return {"files": names, "chunks_indexed": len(added), "total_chunks": len(rag.corpus_chunks)}


def _source_label(source: dict[str, Any]) -> str:
    name = source.get("source_name") or "Paper"
    label = re.sub(r"([\\`*_\[\]<>])", r"\\\1", name)
    page = source.get("page_number")
    return f"{label}, p. {page}" if page is not None else label


def _paper_references(
    citations: list[str], result: dict[str, Any]
) -> list[tuple[str, list[str]]]:
    """Group cited passages by file and page in citation order."""
    sources = {}
    for items in (result.get("retrieved_sources"), result.get("citation_details")):
        for item in items or []:
            citation = item.get("citation")
            if citation:
                sources.setdefault(citation, {}).update(item)

    references: dict[tuple[str, int | None], tuple[str, list[str]]] = {}
    for citation in citations:
        source = sources.get(citation, {})
        location = (source.get("source_name") or citation, source.get("page_number"))
        _, excerpts = references.setdefault(location, (_source_label(source), []))
        if (excerpt := source.get("excerpt")) and excerpt not in excerpts:
            excerpts.append(excerpt)
    return list(references.values())


def render_assistant_message(message: dict[str, Any], show_violations: bool) -> None:
    result = message.get("result")
    if not result:
        st.error(message.get("error", "The request failed."))
        if message.get("api_unavailable"):
            st.info("Start the API in a terminal and leave it running while you use the UI:")
            st.code(API_START_COMMAND, language="powershell")
        return

    answer = result.get("answer", "No answer returned.")
    if message.get("source_mode") == "Web search":
        st.markdown(answer)
        sources = result.get("sources", [])
        st.caption(f"Web search · {len(sources)} source(s) · {message['elapsed']:.2f}s")
        for index, source in enumerate(sources, start=1):
            title = source.get("title") or source.get("url") or f"Source {index}"
            st.markdown(f"{index}. [{title}]({source.get('url', '')})")
            if source.get("content"):
                st.caption(source["content"])
        return

    citations = result.get("citations") or [f"Chunk_{id_}" for id_ in CITATION_RE.findall(answer)]
    references = _paper_references(citations, result)
    st.markdown(without_citation_markers(answer))
    if result.get("answer_mode") == "extractive":
        if result.get("fallback_reason") == "no_api_key":
            st.caption("Passages from the papers · Set GROQ_API_KEY to enable synthesized explanations.")
        else:
            st.caption("Passages from the papers · A synthesized explanation is currently unavailable.")

    if references:
        st.markdown("**Sources**")
        for label, excerpts in references:
            if excerpts:
                with st.expander(label):
                    for excerpt in excerpts:
                        st.text(excerpt)
            else:
                st.markdown(label)

    cited = set(citations)
    other_passages = [
        source for source in result.get("retrieved_sources") or []
        if source.get("citation") not in cited
    ]
    if other_passages:
        with st.expander(f"Other retrieved passages ({len(other_passages)})"):
            for source in other_passages:
                st.markdown(f"**{_source_label(source)}**")
                st.text(source.get("excerpt", ""))

    with st.expander("Answer details"):
        faithfulness = result.get("faithfulness_score")
        if faithfulness is not None:
            st.caption(f"Evidence support {faithfulness:.0%}")
        st.caption(f"Answer time {message['elapsed']:.2f}s")
        if result.get("cache_status") == "hit":
            st.caption("Reused an answer for this paper library")
        if timings := result.get("latency_ms"):
            st.caption(" · ".join(f"{name} {value:.0f} ms" for name, value in timings.items()))

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
    elif INLINE_PIPELINE_CACHE_VERSION in st.session_state:
        inline_rag = _load_inline_pipeline().rag
        inline_counts: dict[str, int] = {}
        for chunk in inline_rag.corpus_chunks:
            source_name = chunk.source_name or chunk.doc_id
            if chunk.doc_id != "demo_retrieval":
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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_assistant_message(message, show_violations)

placeholder = "Search the web…" if search_mode == "Web search" else "Ask about your papers…"
quick_question = None
if search_mode == "Indexed papers":
    overview, implications = st.columns(2)
    if overview.button(PAPER_OVERVIEW_QUESTION, use_container_width=True):
        quick_question = PAPER_OVERVIEW_QUESTION
    if implications.button(PAPER_IMPLICATIONS_QUESTION, use_container_width=True):
        quick_question = PAPER_IMPLICATIONS_QUESTION
prompt = st.chat_input(placeholder) or quick_question

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
            if library_status and not library_status.get("models_ready", False):
                if library_status.get("model_warmup_status") == "failed":
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
