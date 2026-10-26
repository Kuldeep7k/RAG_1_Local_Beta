import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from config import _env_int
from main import ChatRequest, _public_job
from generation.prompts import format_context
import db


class HardeningTests(unittest.TestCase):
    def test_environment_integer_bounds_are_enforced(self):
        self.assertEqual(_env_int("TEST_VALUE", 3, 1, 5), 3)
        with self.assertRaises(ValueError):
            _env_int("TEST_VALUE", 0, 1, 5)

    def test_remote_qdrant_requires_authentication(self):
        from config import _validate_qdrant_security

        with self.assertRaises(ValueError):
            _validate_qdrant_security("https://qdrant.example", None)
        _validate_qdrant_security("http://127.0.0.1:6333", None)

    def test_chat_request_rejects_unbounded_input(self):
        with self.assertRaises(ValidationError):
            ChatRequest(conversation_id="x", message="x" * 2001)
        with self.assertRaises(ValidationError):
            ChatRequest(conversation_id="", message="hello")
        with self.assertRaises(ValidationError):
            ChatRequest(user_id="caller-controlled", conversation_id="x", message="hello")

    def test_job_response_does_not_expose_path_or_internal_error(self):
        public_job = _public_job({
            "id": 7,
            "file_path": "data/raw/private.pdf",
            "source_name": "private.pdf",
            "error": "Traceback with provider details",
        })
        self.assertNotIn("file_path", public_job)
        self.assertEqual(public_job["error"], "Ingestion failed; see server logs for details.")

    def test_job_claim_is_atomic_and_single_use(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_db_path = db.DB_PATH
            db.DB_PATH = str(Path(temp_dir) / "test.db")
            try:
                db.init_db()
                conn = db.get_connection()
                conn.execute("INSERT INTO jobs (file_path) VALUES (?)", ("file.txt",))
                conn.commit()
                conn.close()

                first = db.claim_next_job()
                self.assertEqual(first["status"], "processing")
                self.assertNotIn("attempts", first)
                self.assertIsNone(db.claim_next_job())
            finally:
                db.DB_PATH = original_db_path

    def test_prompt_uses_same_source_label_as_system_citation_rule(self):
        context = format_context([{"source": "policy.txt", "page": 4, "text": "A fact."}])
        self.assertIn("[Source: policy.txt, page 4]", context)


if __name__ == "__main__":
    unittest.main()