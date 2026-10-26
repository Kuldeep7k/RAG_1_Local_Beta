"""Single shared Qdrant client, reused across every ingestion and retrieval
call instead of opening a brand-new HTTP client (and TCP connection) every
time. Also centralizes API-key auth so it's set in exactly one place."""
from qdrant_client import QdrantClient

from config import QDRANT_URL, QDRANT_API_KEY

_client = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client
