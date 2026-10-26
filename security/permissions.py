"""Configured user scopes used after authentication."""
import json
import os

# Access levels ordered from least to most privileged.
ACCESS_HIERARCHY = ["public", "employee", "manager", "executive"]

class UnknownUserError(ValueError):
    """Raised when an authenticated identity has no configured scope."""


def _users() -> dict[str, dict]:
    try:
        users = json.loads(os.environ["RAG_AUTH_USERS"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise UnknownUserError("Authentication user configuration is unavailable") from exc
    if not isinstance(users, dict):
        raise UnknownUserError("Authentication user configuration is invalid")
    return users


class User:
    def __init__(self, user_id: str):
        record = _users().get(user_id)
        if record is None:
            raise UnknownUserError(f"Unknown user: {user_id}")
        if not isinstance(record, dict) or record.get("access_level") not in ACCESS_HIERARCHY:
            raise UnknownUserError(f"Invalid configuration for user: {user_id}")
        self.user_id = user_id
        self.role = record.get("role", "employee")
        self.access_level = record["access_level"]
        self.department = record.get("department")

    def allowed_access_levels(self) -> list:
        """Returns every access level at or below this user's own level,
        so a manager can see employee + public content, etc."""
        idx = ACCESS_HIERARCHY.index(self.access_level)
        return ACCESS_HIERARCHY[: idx + 1]


def get_user(user_id: str) -> User:
    return User(user_id)


def can_upload_access(user: User, access_level: str) -> bool:
    """Return whether a user may assign the requested document level."""
    return access_level in user.allowed_access_levels()


def can_upload_department(user: User, department: str | None) -> bool:
    """Return whether an upload may carry the requested department scope."""
    return department is None or department == user.department
