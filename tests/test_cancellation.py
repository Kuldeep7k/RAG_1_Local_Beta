import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import db
import worker
from config import EMBEDDING_DIM
from ingestion import ingest as ingestion
from main import app


class CancellationTests(unittest.TestCase):
    def setUp(self):
        self.original_db_path = db.DB_PATH
        self.original_users = os.environ.get("RAG_AUTH_USERS")
        self.original_tokens = os.environ.get("RAG_AUTH_TOKENS")
        self.temp_dir = tempfile.TemporaryDirectory()
        db.DB_PATH = str(Path(self.temp_dir.name) / "test.db")
        os.environ["RAG_AUTH_USERS"] = json.dumps({
            "alice": {"role": "employee", "access_level": "employee", "department": "Legal"},
            "bob": {"role": "employee", "access_level": "employee", "department": "HR"},
        })
        os.environ["RAG_AUTH_TOKENS"] = json.dumps({
            "alice-token": "alice", "bob-token": "bob",
        })
        db.init_db()

    def tearDown(self):
        db.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()
        if self.original_users is None:
            os.environ.pop("RAG_AUTH_USERS", None)
        else:
            os.environ["RAG_AUTH_USERS"] = self.original_users
        if self.original_tokens is None:
            os.environ.pop("RAG_AUTH_TOKENS", None)
        else:
            os.environ["RAG_AUTH_TOKENS"] = self.original_tokens

    def add_job(self, status="pending", uploader_id="alice"):
        conn = db.get_connection()
        cursor = conn.execute(
            "INSERT INTO jobs (file_path, source_name, uploader_id, status) VALUES (?, ?, ?, ?)",
            ("file.txt", "file.txt", uploader_id, status),
        )
        conn.commit()
        job_id = cursor.lastrowid
        conn.close()
        return job_id

    def test_cancel_endpoint_enforces_owner_and_terminal_conflicts(self):
        pending_id = self.add_job("pending")
        processing_id = self.add_job("processing")
        done_id = self.add_job("done")
        cancelled_id = self.add_job("cancelled")
        other_user_id = self.add_job("pending", "bob")

        with TestClient(app) as client:
            headers = {"Authorization": "Bearer alice-token"}
            response = client.post(f"/jobs/{pending_id}/cancel", headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "cancelled")
            self.assertEqual(response.json()["previous_status"], "pending")
            self.assertEqual(client.post(f"/jobs/{pending_id}/cancel", headers=headers).status_code, 409)
            self.assertEqual(client.post(f"/jobs/{processing_id}/cancel", headers=headers).status_code, 200)
            self.assertEqual(client.post(f"/jobs/{done_id}/cancel", headers=headers).status_code, 409)
            self.assertEqual(client.post(f"/jobs/{cancelled_id}/cancel", headers=headers).status_code, 409)
            self.assertEqual(client.post(f"/jobs/{other_user_id}/cancel", headers=headers).status_code, 404)
            self.assertEqual(client.post("/jobs/9999/cancel", headers=headers).status_code, 404)

    def test_finish_job_cannot_overwrite_cancelled_state(self):
        job_id = self.add_job("processing")
        self.assertEqual(db.cancel_job(job_id, "alice"), "processing")
        self.assertFalse(db.finish_job(job_id))
        conn = db.get_connection()
        status = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()["status"]
        conn.close()
        self.assertEqual(status, "cancelled")

    def test_ingestion_cancellation_removes_partial_sqlite_state_and_skips_upsert(self):
        job_id = self.add_job("processing")
        chunks = [
            {"text": "first", "page": 1, "chunk_index": 0},
            {"text": "second", "page": 1, "chunk_index": 1},
        ]
        fake_client = type("FakeClient", (), {
            "get_collections": lambda self: type("Collections", (), {"collections": []})(),
            "create_collection": lambda *args, **kwargs: None,
            "upsert": unittest.mock.Mock(),
            "delete": unittest.mock.Mock(),
        })()
        with patch.object(ingestion, "EMBED_BATCH_SIZE", 1), \
             patch.object(ingestion, "get_qdrant_client", return_value=fake_client), \
             patch.object(ingestion, "load_document", return_value=["page"]), \
             patch.object(ingestion, "chunk_document", return_value=chunks), \
             patch.object(ingestion, "embed_batch", return_value=[[0.1] * EMBEDDING_DIM]), \
             patch.object(ingestion, "is_job_cancelled", side_effect=[False, False, True]):
            with self.assertRaises(ingestion.IngestionCancelled):
                ingestion.ingest_file("file.txt", job_id=job_id, source_name="file.txt")

        conn = db.get_connection()
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 0)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 0)
        conn.close()
        fake_client.upsert.assert_not_called()
        fake_client.delete.assert_called_once()

    def test_worker_preserves_cancelled_ingestion(self):
        job_id = self.add_job("cancelled")
        with patch.object(worker, "ingest_file", side_effect=ingestion.IngestionCancelled()):
            worker.process_job({
                "id": job_id, "file_path": "file.txt", "access_level": "employee",
                "department": None,
            })
        conn = db.get_connection()
        status = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()["status"]
        conn.close()
        self.assertEqual(status, "cancelled")


if __name__ == "__main__":
    unittest.main()
