"""Local RAG's Streamlit interface.

Run with ``streamlit run ui.py`` after starting the API and ingestion worker.
"""
import time
import os

import pandas as pd
import requests
import streamlit as st

from config import ACCESS_HIERARCHY, DEFAULT_ACCESS_LEVEL

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")
API_TOKEN = os.environ.get("RAG_API_TOKEN")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
POLL_INTERVAL_SECONDS = float(os.environ.get("UI_POLL_INTERVAL_SECONDS", "1"))

st.set_page_config(page_title="Local RAG", page_icon="🔎", layout="wide")
st.markdown(
    """
    <style>
            .stApp { background: linear-gradient(135deg, #0b0f17 0%, #101827 58%, #0b1018 100%); }
      .block-container { max-width: 1120px; padding-top: 3.2rem; padding-bottom: 8rem; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(148, 163, 184, .18); }
      .rag-eyebrow { color: #38bdf8; font-size: .8rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
      .rag-hero { padding: 2.5rem 0 1.5rem; text-align: center; }
      .rag-hero h1 { margin: .35rem 0 .6rem; font-size: clamp(2.4rem, 6vw, 4.5rem); letter-spacing: -.06em; }
      .rag-hero p { color: #94a3b8; font-size: 1.05rem; margin: 0 auto; max-width: 38rem; }
      .status-card { border: 1px solid rgba(56, 189, 248, .35); background: rgba(14, 116, 144, .13); border-radius: 14px; padding: 1rem; margin: .4rem 0 1rem; }
      .status-card strong { color: #e0f2fe; display: block; margin-bottom: .25rem; }
      .status-card span { color: #bae6fd; font-size: .9rem; }
      .ready-card { border: 1px solid rgba(74, 222, 128, .28); background: rgba(20, 83, 45, .22); border-radius: 14px; padding: .85rem 1rem; margin: 1rem auto; max-width: 42rem; color: #bbf7d0; }
            .service-card { border: 1px solid rgba(148, 163, 184, .22); background: rgba(15, 23, 42, .5); border-radius: 12px; padding: .7rem .85rem; margin: .4rem 0 1.2rem; }
            .service-card strong { display: block; color: #e2e8f0; font-size: .86rem; margin-bottom: .2rem; }
            .service-card span { color: #94a3b8; font-size: .78rem; }
            .service-ok span { color: #86efac; }
            .service-waiting span { color: #fcd34d; }
            .upload-note { color: #94a3b8; font-size: .78rem; margin-top: -.25rem; }
      [data-testid="stChatInput"] { max-width: 760px; margin-left: auto; margin-right: auto; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str):
    """Read from the API without letting a temporary connection failure crash UI."""
    try:
        response = requests.get(f"{API_URL}{path}", headers=api_headers(), timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def api_post(path: str):
    """Call a state-changing API endpoint with the configured bearer token."""
    try:
        response = requests.post(f"{API_URL}{path}", headers=api_headers(), timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as error:
        st.error(f"Could not cancel ingestion: {error}")
        return None


def api_headers():
    return {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}


def format_bytes(value: int) -> str:
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.0f} MB"
    return f"{value / 1024:.0f} KB"


def render_service_status():
    """Show readiness before users attempt an upload or question."""
    try:
        response = requests.get(f"{API_URL}/ready", timeout=3)
        if response.ok:
            st.markdown(
                '<div class="service-card service-ok"><strong>System ready</strong>'
                '<span>API, database, and vector search are available.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="service-card service-waiting"><strong>Starting up</strong>'
                '<span>The search services are still warming up.</span></div>',
                unsafe_allow_html=True,
            )
    except requests.exceptions.RequestException:
        st.markdown(
            '<div class="service-card"><strong>API unavailable</strong>'
            '<span>Start the local stack, then refresh this page.</span></div>',
            unsafe_allow_html=True,
        )


def get_active_job():
    """Recover the active job after a browser refresh, if necessary."""
    job_id = st.session_state.get("ingestion_job_id")
    if job_id is not None:
        job = api_get(f"/jobs/{job_id}")
        if job and job.get("status") in {"pending", "processing"}:
            return job

    jobs = api_get("/jobs?limit=50") or []
    active_jobs = [job for job in jobs if job.get("status") in {"pending", "processing"}]
    if active_jobs:
        job = active_jobs[0]
        st.session_state.ingestion_job_id = job["id"]
        return job
    return None


def render_ingestion_status(job: dict):
    """Render progress for the asynchronous ingestion job."""
    status = job.get("status", "pending")
    done = job.get("chunks_done") or 0
    total = job.get("total_chunks")
    filename = job.get("source_name", "document")
    if status == "processing" and total:
        label = f"Indexing {filename}: {done:,} of {total:,} chunks"
        progress = min(done / total, 1.0)
    elif status == "processing":
        label, progress = f"Reading {filename} and preparing the index…", 0.05
    else:
        label, progress = f"{filename} is waiting for the ingestion worker…", 0.0
    st.markdown(f'<div class="status-card"><strong>Ingestion in progress</strong><span>{label}</span></div>', unsafe_allow_html=True)
    st.progress(progress)
    st.caption("Questions are disabled so answers only use the completed document index.")


def render_debug_panel(debug: dict):
    if not debug:
        return
    with st.expander("Retrieval details"):
        st.caption(
            f"Access: {debug['access_levels_applied']}  ·  Candidates: {debug['candidates_considered']}  ·  "
            f"Selected: {debug['candidates_selected_after_rerank']}  ·  Best rerank score: {debug.get('best_rerank_score')}"
        )
        if debug.get("low_confidence_triggered"):
            st.warning("No retrieved chunk was a strong match, so no answer was generated.")
        rows = debug.get("top_candidates", [])
        if not rows:
            st.write("No candidates retrieved.")
            return
        df = pd.DataFrame(rows).rename(columns={
            "hybrid_rank": "Rank", "selected_in_final_answer": "Used?", "source": "Source", "page": "Page",
            "article_ref": "Article", "fused_score": "Fused", "vector_rank": "Vector rank",
            "vector_score": "Vector score", "keyword_rank": "Keyword rank", "keyword_score": "Keyword score",
            "rerank_score": "Rerank score", "text_preview": "Preview",
        })
        st.dataframe(df, width="stretch", hide_index=True)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(time.time())

with st.sidebar:
    st.markdown("### Knowledge base")
    st.caption("Add a file, wait for indexing, then start asking grounded questions.")
    render_service_status()
    uploaded_file = st.file_uploader("Choose a document", type=["pdf", "docx", "txt", "md"])
    st.markdown(
        f'<div class="upload-note">PDF, DOCX, TXT, or MD · up to {format_bytes(MAX_UPLOAD_BYTES)} per file</div>',
        unsafe_allow_html=True,
    )
    access_level = st.selectbox(
        "Access level",
        ACCESS_HIERARCHY,
        index=ACCESS_HIERARCHY.index(DEFAULT_ACCESS_LEVEL),
    )
    active_job = get_active_job()
    oversized_upload = uploaded_file is not None and uploaded_file.size > MAX_UPLOAD_BYTES
    if oversized_upload:
        st.error(f"This file is {format_bytes(uploaded_file.size)}. The limit is {format_bytes(MAX_UPLOAD_BYTES)}.")
    upload_disabled = not uploaded_file or oversized_upload or active_job is not None or not API_TOKEN
    if st.button("Add to knowledge base", width="stretch", disabled=upload_disabled):
        try:
            with st.spinner("Uploading document…"):
                response = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                    data={"access_level": access_level}, headers=api_headers(), timeout=30,
                )
                response.raise_for_status()
            st.session_state.ingestion_job_id = response.json()["job_id"]
            st.rerun()
        except requests.exceptions.HTTPError as error:
            try:
                detail = error.response.json().get("detail", "The API rejected the upload.")
            except (ValueError, AttributeError):
                detail = "The API rejected the upload."
            st.error(detail)
        except requests.exceptions.RequestException as error:
            st.error(f"Could not start ingestion: {error}")

    if not API_TOKEN:
        st.warning("Authentication is not configured for this UI session.")

    if active_job:
        render_ingestion_status(active_job)
        if st.button("Stop ingestion", width="stretch", type="secondary"):
            if api_post(f"/jobs/{active_job['id']}/cancel"):
                st.session_state.ingestion_job_id = active_job["id"]
                st.rerun()
    elif st.session_state.get("ingestion_job_id"):
        completed_job = api_get(f"/jobs/{st.session_state.ingestion_job_id}")
        st.session_state.ingestion_job_id = None
        if completed_job and completed_job.get("status") == "failed":
            st.error(f"Ingestion failed: {completed_job.get('error', 'Unknown error')}")
        elif completed_job and completed_job.get("status") == "cancelled":
            st.info("Ingestion cancelled.")
        elif completed_job:
            st.success("Document indexed and ready to search.")

    st.divider()
    st.markdown("### Chat settings")
    if st.button("Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.session_state.conversation_id = str(time.time())
        st.rerun()

if not st.session_state.messages:
    st.markdown(
        """<section class="rag-hero"><div class="rag-eyebrow">Private document search</div>
        <h1>🔎 Local RAG</h1><p>Ask clear questions and get answers grounded in the documents you add.</p></section>""",
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for source in msg["sources"]:
                    st.write(f"- {source['document']}, page {source['page']}")
        render_debug_panel(msg.get("debug"))

chat_disabled = active_job is not None
placeholder = "Indexing your document — questions will unlock automatically." if chat_disabled else "Ask a question about your documents…"
query = st.chat_input(placeholder, disabled=chat_disabled)
if chat_disabled:
    st.markdown('<div class="ready-card">Your document is being indexed. The question box will unlock automatically when it is ready.</div>', unsafe_allow_html=True)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents…"):
            sources, debug = [], None
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"conversation_id": st.session_state.conversation_id, "message": query},
                    headers=api_headers(), timeout=180,
                )
                response.raise_for_status()
                result = response.json()
                answer, sources, debug = result["answer"], result["sources"], result.get("debug")
                st.write(answer)
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.write(f"- {source['document']}, page {source['page']}")
                timing = result["timing"]
                st.caption(f"Answered in {timing['total_ms']} ms · retrieval {timing['retrieval_ms']} ms · rerank {timing['rerank_ms']} ms")
                render_debug_panel(debug)
            except requests.exceptions.RequestException as error:
                answer = f"Error contacting the API: {error}"
                st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources, "debug": debug})

# Refresh only while a job is active; terminal states stop polling immediately.
if active_job:
    time.sleep(POLL_INTERVAL_SECONDS)
    st.rerun()
