import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import db
import worker


class WorkerFailureTests(unittest.TestCase):
    def test_processing_failure_is_terminal_and_not_requeued(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_db_path = db.DB_PATH
            db.DB_PATH = str(Path(temp_dir) / "test.db")
            try:
                db.init_db()
                connection = db.get_connection()
                cursor = connection.execute(
                    "INSERT INTO jobs (file_path, status) VALUES (?, 'processing')",
                    ("missing.pdf",),
                )
                job_id = cursor.lastrowid
                connection.commit()
                connection.close()

                with patch.object(
                    worker,
                    "ingest_file",
                    side_effect=RuntimeError("embedding provider unavailable"),
                ):
                    worker.process_job({
                        "id": job_id,
                        "file_path": "missing.pdf",
                        "access_level": "employee",
                        "department": None,
                    })

                connection = db.get_connection()
                job = connection.execute(
                    "SELECT status, error FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
                connection.close()
                self.assertEqual(job["status"], "failed")
                self.assertEqual(job["error"], "embedding provider unavailable")
                self.assertIsNone(db.claim_next_job())
            finally:
                db.DB_PATH = original_db_path


if __name__ == "__main__":
    unittest.main()
