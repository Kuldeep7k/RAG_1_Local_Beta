"""Polls the SQLite jobs table and runs ingestion asynchronously, so
document uploads via the API never block on parsing/embedding. Run this
as a separate long-lived process alongside `uvicorn main:app`."""
import time
import os

from config import JOB_STALE_SECONDS
from db import (
    get_connection, init_db, claim_next_job, fail_stale_jobs,
    prune_terminal_jobs, finish_job,
)
from ingestion.ingest import ingest_file, cleanup_document, IngestionCancelled

POLL_INTERVAL_SECONDS = float(os.environ.get("WORKER_POLL_INTERVAL_SECONDS", "3"))


def process_job(job: dict):
    document_id = None
    try:
        document_id = ingest_file(
            job["file_path"], job["access_level"], job["department"],
            job_id=job["id"], source_name=job.get("source_name"),
        )
        if finish_job(job["id"]):
            print(f"[worker] Job {job['id']} done: {job['file_path']}")
        else:
            cleanup_document(document_id)
            print(f"[worker] Job {job['id']} cancelled during finalization")
    except IngestionCancelled:
        print(f"[worker] Job {job['id']} cancelled")
    except Exception as exc:
        print(f"[worker] Job {job['id']} failed permanently: {exc}")
        conn = get_connection()
        try:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ? AND status = 'processing'",
                ("failed", str(exc), job["id"]),
            )
            conn.commit()
        finally:
            conn.close()


def run():
    init_db()
    # Self-heal: any job left in 'processing' means the worker died or was
    # restarted mid-job. Mark it failed so it is not silently orphaned or
    # processed a second time.
    reset_count = fail_stale_jobs(JOB_STALE_SECONDS)
    if reset_count:
        print(f"[worker] Marked {reset_count} abandoned job(s) as failed.")
    pruned_count = prune_terminal_jobs()
    if pruned_count:
        print(f"[worker] Removed {pruned_count} completed historical job(s).")

    print("[worker] Watching for ingestion jobs... (Ctrl+C to stop)")
    while True:
        reset_count = fail_stale_jobs(JOB_STALE_SECONDS)
        if reset_count:
            print(f"[worker] Marked {reset_count} abandoned job(s) as failed.")
        while job := claim_next_job():
            process_job(job)

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run()
