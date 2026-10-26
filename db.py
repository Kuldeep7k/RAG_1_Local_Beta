"""SQLite schema setup and connection helper.

Holds: documents, chunks (mirrored into an FTS5 table for keyword search),
async ingestion jobs, and per-request observability traces.
"""
import sqlite3
from pathlib import Path

from config import DB_PATH


def get_connection() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    # timeout=30: if the DB is briefly locked by another connection (e.g. the
    # worker writing at the same moment as the API), wait up to 30s instead
    # of failing immediately with "database is locked".
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL mode lets readers and a writer work concurrently instead of
    # locking the whole file on every write -- important since the API
    # and the worker both open their own connections.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            access_level TEXT DEFAULT 'employee',
            department TEXT,
            status TEXT NOT NULL DEFAULT 'ready',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    existing_document_cols = {row["name"] for row in cur.execute("PRAGMA table_info(documents)").fetchall()}
    if "status" not in existing_document_cols:
        cur.execute("ALTER TABLE documents ADD COLUMN status TEXT NOT NULL DEFAULT 'ready'")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            text TEXT NOT NULL,
            page INTEGER,
            section TEXT,
            article_number INTEGER,
            article_ref TEXT,
            access_level TEXT DEFAULT 'employee',
            department TEXT,
            embedding_version TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(document_id)
        )
    """)

    # Migration: add article_number if this DB predates it.
    existing_chunk_cols = {row["name"] for row in cur.execute("PRAGMA table_info(chunks)").fetchall()}
    if "article_number" not in existing_chunk_cols:
        cur.execute("ALTER TABLE chunks ADD COLUMN article_number INTEGER")
    if "article_ref" not in existing_chunk_cols:
        cur.execute("ALTER TABLE chunks ADD COLUMN article_ref TEXT")
        # Existing numeric article metadata remains usable until the document
        # is re-indexed with full references such as 21A and 239AA.
        cur.execute("UPDATE chunks SET article_ref = CAST(article_number AS TEXT) WHERE article_number IS NOT NULL")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_article_ref ON chunks(article_ref)")

    # FTS5 virtual table mirrors chunks.text for BM25-style keyword search.
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED,
            text,
            content='chunks',
            content_rowid='rowid'
        )
    """)

    # Keep the FTS index in sync with the chunks table automatically.
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, chunk_id, text)
            VALUES (new.rowid, new.chunk_id, new.text);
        END
    """)
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, text)
            VALUES ('delete', old.rowid, old.chunk_id, old.text);
        END
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            source_name TEXT,
            access_level TEXT DEFAULT 'employee',
            department TEXT,
            status TEXT DEFAULT 'pending',
            error TEXT,
            total_chunks INTEGER,
            chunks_done INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migration: add progress columns if this DB was created before they existed.
    existing_job_cols = {row["name"] for row in cur.execute("PRAGMA table_info(jobs)").fetchall()}
    if "uploader_id" not in existing_job_cols:
        cur.execute("ALTER TABLE jobs ADD COLUMN uploader_id TEXT")
    if "source_name" not in existing_job_cols:
        cur.execute("ALTER TABLE jobs ADD COLUMN source_name TEXT")
    if "total_chunks" not in existing_job_cols:
        cur.execute("ALTER TABLE jobs ADD COLUMN total_chunks INTEGER")
    if "chunks_done" not in existing_job_cols:
        cur.execute("ALTER TABLE jobs ADD COLUMN chunks_done INTEGER DEFAULT 0")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            query TEXT,
            rewritten_query TEXT,
            retrieved_chunk_ids TEXT,
            reranked_chunk_ids TEXT,
            answer TEXT,
            sources TEXT,
            latency_ms REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DB_PATH}")


def update_job_progress(job_id: int, total_chunks: int = None, chunks_done: int = None):
    """Updates progress fields on a job row. Called periodically during
    ingestion so the API/UI can show real progress instead of a blind spinner."""
    conn = get_connection()
    if total_chunks is not None:
        conn.execute(
            "UPDATE jobs SET total_chunks = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (total_chunks, job_id),
        )
    if chunks_done is not None:
        conn.execute(
            "UPDATE jobs SET chunks_done = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (chunks_done, job_id),
        )
    conn.commit()
    conn.close()


def cancel_job(job_id: int, uploader_id: str) -> str:
    """Cancel an owned non-terminal job and return its previous status."""
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT status FROM jobs WHERE id = ? AND uploader_id = ?",
            (job_id, uploader_id),
        ).fetchone()
        if row is None:
            conn.rollback()
            raise LookupError("job not found")
        if row["status"] not in {"pending", "processing"}:
            conn.rollback()
            raise ValueError(row["status"])
        conn.execute(
            "UPDATE jobs SET status = 'cancelled', error = NULL, updated_at = CURRENT_TIMESTAMP "
            "WHERE id = ? AND uploader_id = ? AND status IN ('pending', 'processing')",
            (job_id, uploader_id),
        )
        conn.commit()
        return row["status"]
    finally:
        conn.close()


def is_job_cancelled(job_id: int) -> bool:
    """Return whether a job has been cancelled, using a short-lived read."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return row is not None and row["status"] == "cancelled"
    finally:
        conn.close()


def finish_job(job_id: int) -> bool:
    """Mark a processing job done, or report that it was cancelled meanwhile."""
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None or row["status"] != "processing":
            conn.rollback()
            return False
        conn.execute(
            "UPDATE jobs SET status = 'done', error = NULL, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (job_id,),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def claim_next_job():
    """Atomically claim the oldest pending job exactly once."""
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = 'pending' ORDER BY id ASC LIMIT 1",
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        updated = conn.execute(
            """UPDATE jobs SET status = 'processing',
               error = NULL, updated_at = CURRENT_TIMESTAMP
               WHERE id = ? AND status = 'pending'""",
            (row["id"],),
        )
        if updated.rowcount != 1:
            conn.rollback()
            return None
        conn.commit()
        return dict(conn.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone())
    finally:
        conn.close()


def fail_stale_jobs(stale_seconds: int) -> int:
    """Mark abandoned processing jobs failed; never return them to the queue."""
    conn = get_connection()
    try:
        updated = conn.execute(
            """UPDATE jobs SET status = 'failed',
               error = 'Worker stopped before ingestion completed.',
               updated_at = CURRENT_TIMESTAMP
               WHERE status = 'processing'
                 AND updated_at < datetime('now', ? || ' seconds')""",
            (f"-{stale_seconds}",),
        )
        conn.commit()
        return updated.rowcount
    finally:
        conn.close()


def prune_terminal_jobs() -> int:
    """Remove every terminal job so no historical job state remains."""
    conn = get_connection()
    try:
        deleted = conn.execute(
            "DELETE FROM jobs WHERE status IN ('done', 'failed', 'cancelled')"
        )
        conn.commit()
        return deleted.rowcount
    finally:
        conn.close()
