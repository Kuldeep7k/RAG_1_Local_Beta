import json
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

import db
from main import app


USERS = {
    "alice": {"role": "employee", "access_level": "employee", "department": "Legal"},
    "bob": {"role": "employee", "access_level": "employee", "department": "HR"},
}
TOKENS = {"alice-local-token": "alice", "bob-local-token": "bob"}


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.original_users = os.environ.get("RAG_AUTH_USERS")
        self.original_tokens = os.environ.get("RAG_AUTH_TOKENS")
        os.environ["RAG_AUTH_USERS"] = json.dumps(USERS)
        os.environ["RAG_AUTH_TOKENS"] = json.dumps(TOKENS)

    def tearDown(self):
        if self.original_users is None:
            os.environ.pop("RAG_AUTH_USERS", None)
        else:
            os.environ["RAG_AUTH_USERS"] = self.original_users
        if self.original_tokens is None:
            os.environ.pop("RAG_AUTH_TOKENS", None)
        else:
            os.environ["RAG_AUTH_TOKENS"] = self.original_tokens

    def test_missing_and_invalid_credentials_are_rejected(self):
        with TestClient(app) as client:
            self.assertEqual(client.get("/jobs").status_code, 401)
            response = client.get("/jobs", headers={"Authorization": "Bearer wrong-token"})
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.json()["detail"], "Invalid bearer token")

    def test_authenticated_identity_controls_job_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            original_db_path = db.DB_PATH
            db.DB_PATH = os.path.join(directory, "test.db")
            try:
                db.init_db()
                connection = db.get_connection()
                connection.execute(
                    "INSERT INTO jobs (file_path, source_name, uploader_id) VALUES (?, ?, ?)",
                    ("private.txt", "private.txt", "alice"),
                )
                connection.commit()
                connection.close()

                with TestClient(app) as client:
                    headers = {"Authorization": "Bearer alice-local-token"}
                    response = client.get("/jobs?user_id=bob", headers=headers)
                    self.assertEqual(response.json()[0]["uploader_id"], "alice")
                    self.assertEqual(
                        client.get("/jobs/1", headers={"Authorization": "Bearer bob-local-token"}).status_code,
                        404,
                    )
            finally:
                db.DB_PATH = original_db_path

    def test_authentication_configuration_failure_is_not_a_bypass(self):
        os.environ.pop("RAG_AUTH_USERS", None)
        os.environ.pop("RAG_AUTH_TOKENS", None)
        with TestClient(app) as client:
            response = client.get("/jobs", headers={"Authorization": "Bearer any-token"})
            self.assertEqual(response.status_code, 503)
            self.assertIn("not configured", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()