"""Authentication dependency for the local FastAPI application."""
import hmac
import json
import os

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from security.permissions import UnknownUserError, get_user

_bearer = HTTPBearer(auto_error=False)


def _auth_config() -> tuple[dict[str, str], dict[str, dict]]:
    raw_tokens = os.environ.get("RAG_AUTH_TOKENS")
    raw_users = os.environ.get("RAG_AUTH_USERS")
    if not raw_tokens or not raw_users:
        raise HTTPException(status_code=503, detail="Authentication is not configured")
    try:
        tokens = json.loads(raw_tokens)
        users = json.loads(raw_users)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=503, detail="Authentication configuration is invalid") from exc
    if not isinstance(tokens, dict) or not isinstance(users, dict) or not tokens or not users:
        raise HTTPException(status_code=503, detail="Authentication configuration is invalid")
    if not all(isinstance(token, str) and isinstance(user_id, str) for token, user_id in tokens.items()):
        raise HTTPException(status_code=503, detail="Authentication configuration is invalid")
    return tokens, users


def get_authenticated_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
):
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer authentication is required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    tokens, _ = _auth_config()
    user_id = next(
        (configured_user_id for token, configured_user_id in tokens.items()
         if hmac.compare_digest(token, credentials.credentials)),
        None,
    )
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return get_user(user_id)
    except UnknownUserError as exc:
        raise HTTPException(status_code=503, detail="Authentication configuration is invalid") from exc