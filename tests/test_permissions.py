import unittest
import json
import os

from security.permissions import (
    UnknownUserError, can_upload_access, can_upload_department, get_user,
)


class PermissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_users = os.environ.get("RAG_AUTH_USERS")
        os.environ["RAG_AUTH_USERS"] = json.dumps({
            "demo_user": {"role": "employee", "access_level": "employee", "department": None},
            "demo_manager": {"role": "manager", "access_level": "manager", "department": "HR"},
        })

    @classmethod
    def tearDownClass(cls):
        if cls.original_users is None:
            os.environ.pop("RAG_AUTH_USERS", None)
        else:
            os.environ["RAG_AUTH_USERS"] = cls.original_users

    def test_unknown_user_is_rejected(self):
        with self.assertRaises(UnknownUserError):
            get_user("not-a-real-user")

    def test_manager_scope_includes_department(self):
        user = get_user("demo_manager")
        self.assertEqual(user.department, "HR")
        self.assertEqual(user.allowed_access_levels(), ["public", "employee", "manager"])

    def test_uploader_cannot_assign_a_higher_access_level(self):
        user = get_user("demo_user")
        self.assertTrue(can_upload_access(user, "employee"))
        self.assertFalse(can_upload_access(user, "manager"))

    def test_uploader_cannot_assign_another_department(self):
        self.assertFalse(can_upload_department(get_user("demo_user"), "HR"))
        self.assertTrue(can_upload_department(get_user("demo_manager"), "HR"))
        self.assertTrue(can_upload_department(get_user("demo_user"), None))


if __name__ == "__main__":
    unittest.main()
