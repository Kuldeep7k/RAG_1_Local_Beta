"""FastAPI app: an /upload endpoint that queues async ingestion, and a
/chat endpoint that runs hybrid retrieval -> rerank -> generate -> cite,
logging a trace of every request for observability."""
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from db import get_connection, init_db, cancel_job
from retrieval.hybrid import hybrid_search
from retrieval.rerank import rerank
from generation.prompts import build_prompt, build_sources
from generation.llm import generate
from security.permissions import (
    User, can_upload_access, can_upload_department,
)
from security.auth import get_authenticated_user
from config import (
    TOP_K_RETRIEVAL, TOP_K_RERANK, RERANK_CONFIDENCE_THRESHOLD,
    ALLOWED_UPLOAD_EXTENSIONS, MAX_UPLOAD_BYTES,
)

UPLOAD_DIR = Path("data/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize persistent storage before serving requests."""
    init_db()
    yield


app = FastAPI(title="Local RAG API", lifespan=lifespan)
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2000)


def _public_job(row) -> dict:
    result = dict(row)
    result.pop("file_path", None)
    if result.get("error"):
        result["error"] = "Ingestion failed; see server logs for details."
    return result


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    access_level: str = Form("employee"),
    department: str = Form(None),
    uploader: User = Depends(get_authenticated_user),
):
    """Saves the file and queues it for async ingestion. Returns immediately
    -- the actual parsing/embedding happens in worker.py."""
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="A filename is required")
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported document type")
    if not can_upload_access(uploader, access_level):
        raise HTTPException(status_code=400, detail="Invalid access level")
    department = (department or "").strip() or None
    if not can_upload_department(uploader, department):
        raise HTTPException(status_code=403, detail="Uploads may only use your department")

    conn = get_connection()
    try:
        active_job = conn.execute(
            """SELECT id FROM jobs
               WHERE uploader_id = ? AND source_name = ?
                 AND status IN ('pending', 'processing')
               ORDER BY id DESC LIMIT 1""",
            (uploader.user_id, filename),
        ).fetchone()
    finally:
        conn.close()
    if active_job is not None:
        raise HTTPException(
            status_code=409,
            detail=f"An ingestion job for {filename!r} is already active.",
        )

    dest_path = UPLOAD_DIR / f"{time.time_ns()}_{filename}"
    bytes_written = 0
    try:
        with open(dest_path, "wb") as destination:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Uploaded file is too large")
                destination.write(chunk)
    except Exception:
        dest_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    try:
        conn = get_connection()
        cur = conn.execute(
            "INSERT INTO jobs (file_path, source_name, uploader_id, access_level, department) VALUES (?, ?, ?, ?, ?)",
            (str(dest_path), filename, uploader.user_id, access_level, department),
        )
        job_id = cur.lastrowid
        conn.commit()
    except Exception:
        dest_path.unlink(missing_ok=True)
        raise
    finally:
        if "conn" in locals():
            conn.close()

    return {"job_id": job_id, "status": "queued", "file": filename}


@app.get("/jobs")
def list_jobs(limit: int = 10, user: User = Depends(get_authenticated_user)):
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE uploader_id = ? ORDER BY id DESC LIMIT ?", (user.user_id, limit)
        ).fetchall()
    finally:
        conn.close()
    return [_public_job(row) for row in rows]


@app.get("/jobs/{job_id}")
def job_status(job_id: int, user: User = Depends(get_authenticated_user)):
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ? AND uploader_id = ?", (job_id, user.user_id)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _public_job(row)


@app.post("/jobs/{job_id}/cancel")
def cancel_ingestion_job(job_id: int, user: User = Depends(get_authenticated_user)):
    try:
        previous_status = cancel_job(job_id, user.user_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=f"Job is already {exc.args[0]}") from exc
    return {"job_id": job_id, "status": "cancelled", "previous_status": previous_status}


@app.post("/chat")
def chat(request: ChatRequest, user: User = Depends(get_authenticated_user)):
    start = time.time()
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    try:
        # 1. Hybrid retrieval, pre-filtered by the user's access level.
        t0 = time.time()
        candidates = hybrid_search(
            request.message,
            top_k=TOP_K_RETRIEVAL,
            access_levels=user.allowed_access_levels(),
            department=user.department,
        )
        retrieval_ms = (time.time() - t0) * 1000

        # 2. Rerank every candidate after all retrieval constraints are applied.
        t0 = time.time()
        top_chunks = rerank(request.message, candidates, top_k=TOP_K_RERANK)
        rerank_ms = (time.time() - t0) * 1000

        best_rerank_score = top_chunks[0].get("rerank_score") if top_chunks else None
        low_confidence = not top_chunks
        if RERANK_CONFIDENCE_THRESHOLD is not None and best_rerank_score is not None:
            low_confidence = best_rerank_score < RERANK_CONFIDENCE_THRESHOLD

        t0 = time.time()
        if low_confidence:
            answer = (
                "I don't have enough information in the retrieved documents to "
                "answer this confidently. The closest matches I found don't "
                "actually address this question."
            )
            sources = []
        else:
            messages = build_prompt(request.message, top_chunks)
            answer = generate(messages)
            sources = build_sources(top_chunks)
        generation_ms = (time.time() - t0) * 1000
    except Exception as exc:
        logger.exception("RAG request failed for conversation %s", request.conversation_id)
        raise HTTPException(status_code=503, detail="Retrieval or generation service is unavailable") from exc

    latency_ms = (time.time() - start) * 1000
    timing = {
        "retrieval_ms": round(retrieval_ms, 1),
        "rerank_ms": round(rerank_ms, 1),
        "generation_ms": round(generation_ms, 1),
        "total_ms": round(latency_ms, 1),
    }
    logger.info("chat timing: %s", timing)

    # 4. Build a debug breakdown of the top candidates: where each one came
    # from (vector search, keyword search, or both), its score at every
    # stage, and whether it survived reranking into the final answer.
    # This is what lets you see WHY a chunk did or didn't get used.
    selected_ids = {c["chunk_id"] for c in top_chunks}

    def to_debug_row(rank, cand):
        rank_detail = cand.get("rank_detail", {})
        vector_info = rank_detail.get("vector")
        keyword_info = rank_detail.get("keyword")
        cand_rerank_score = cand.get("rerank_score")
        return {
            "hybrid_rank": rank,
            "selected_in_final_answer": cand["chunk_id"] in selected_ids,
            "source": cand["source"],
            "page": cand["page"],
            "article_ref": cand.get("article_ref"),
            "fused_score": round(cand.get("fused_score", 0), 4),
            "vector_rank": vector_info["rank"] if vector_info else None,
            "vector_score": round(vector_info["score"], 4) if vector_info else None,
            "keyword_rank": keyword_info["rank"] if keyword_info else None,
            "keyword_score": round(keyword_info["score"], 4) if keyword_info else None,
            "rerank_score": round(cand_rerank_score, 4) if cand_rerank_score is not None else None,
            "text_preview": cand["text"][:200],
        }

    debug_candidates = [to_debug_row(rank, cand) for rank, cand in enumerate(candidates[:10], start=1)]

    # The reranker can pick a final chunk from anywhere in `candidates`, not
    # just the fused-score top 10 shown above -- so make sure every selected
    # chunk is visible here even if its fused rank was outside that window.
    shown_chunk_ids = {c["chunk_id"] for c in candidates[:10]}
    for cand in candidates:
        if cand["chunk_id"] in selected_ids and cand["chunk_id"] not in shown_chunk_ids:
            debug_candidates.append(to_debug_row(None, cand))

    debug_info = {
        "access_levels_applied": user.allowed_access_levels(),
        "candidates_considered": len(candidates),
        "candidates_selected_after_rerank": len(top_chunks),
        "best_rerank_score": round(best_rerank_score, 4) if best_rerank_score is not None else None,
        "low_confidence_triggered": low_confidence,
        "top_candidates": debug_candidates,
    }

    # 5. Log the trace for observability/debugging.
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO traces
               (conversation_id, query, retrieved_chunk_ids, reranked_chunk_ids, answer, sources, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                request.conversation_id,
                request.message,
                json.dumps([c["chunk_id"] for c in candidates]),
                json.dumps([c["chunk_id"] for c in top_chunks]),
                answer,
                json.dumps(sources),
                latency_ms,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return {"answer": answer, "sources": sources, "timing": timing, "debug": debug_info}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def readiness():
    conn = None
    try:
        conn = get_connection()
        conn.execute("SELECT 1").fetchone()
        from qdrant_utils import get_qdrant_client
        get_qdrant_client().get_collections()
    except Exception:
        logger.warning("Readiness check failed", exc_info=True)
        raise HTTPException(status_code=503, detail="Dependencies are not ready")
    finally:
        if conn is not None:
            conn.close()
    return {"status": "ready"}
