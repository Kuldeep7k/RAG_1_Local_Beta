"""Central configuration for the local RAG system."""
import os
from urllib.parse import urlparse


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
	raw_value = os.environ.get(name, str(default))
	try:
		value = int(raw_value)
	except ValueError as exc:
		raise ValueError(f"{name} must be an integer") from exc
	if not minimum <= value <= maximum:
		raise ValueError(f"{name} must be between {minimum} and {maximum}")
	return value


def _validate_qdrant_security(url: str, api_key: str | None) -> None:
	if not api_key and urlparse(url).hostname not in {"localhost", "127.0.0.1", "::1"}:
		raise ValueError("QDRANT_API_KEY is required when QDRANT_URL is not local")


# --- Ollama (LLM) ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2:3b")
EMBED_BATCH_SIZE = _env_int("EMBED_BATCH_SIZE", 16, 1, 256)

# --- Local sentence-transformer embeddings ---
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# --- Qdrant (vector DB) ---
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "documents_minilm")
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 384, 1, 8192)
EMBEDDING_VERSION = os.environ.get("EMBEDDING_VERSION", "all-MiniLM-L6-v2")
INDEX_SCHEMA_VERSION = os.environ.get("INDEX_SCHEMA_VERSION", "2")
# Set via: $env:QDRANT_API_KEY="your-secret-key" before starting both
# qdrant.exe and this app. None means no auth (fine only if Qdrant is
# bound to 127.0.0.1 and never exposed beyond this machine).
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
_validate_qdrant_security(QDRANT_URL, QDRANT_API_KEY)

# --- SQLite (metadata, keyword search, jobs, traces) ---
DB_PATH = os.environ.get("DB_PATH", "data/rag.db")

# --- Chunking ---
# all-MiniLM-L6-v2 truncates inputs beyond its 256-token context. Keep a
# margin for special tokens so every chunk is embedded in full.
CHUNK_SIZE_TOKENS = _env_int("CHUNK_SIZE_TOKENS", 220, 32, 240)
CHUNK_OVERLAP_TOKENS = _env_int("CHUNK_OVERLAP_TOKENS", 35, 0, CHUNK_SIZE_TOKENS - 1)

# --- Retrieval ---
TOP_K_RETRIEVAL = _env_int("TOP_K_RETRIEVAL", 40, 1, 200)
TOP_K_RERANK = _env_int("TOP_K_RERANK", 8, 1, TOP_K_RETRIEVAL)
# Keep the current model as the baseline; compare alternatives with
# eval/benchmark_rerankers.py before changing this value.
RERANKER_MODEL = os.environ.get(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
# Cross-encoder scores are raw model logits, not confidence probabilities.
# Keep this disabled until it is calibrated against a representative labelled
# evaluation set. Set a numeric value only with documented evaluation data.
RERANK_CONFIDENCE_THRESHOLD = None

# --- Access control ---
ACCESS_HIERARCHY = ("public", "employee", "manager", "executive")
DEFAULT_ACCESS_LEVEL = os.environ.get("DEFAULT_ACCESS_LEVEL", "employee")
if DEFAULT_ACCESS_LEVEL not in ACCESS_HIERARCHY:
    raise ValueError(
        f"DEFAULT_ACCESS_LEVEL must be one of {', '.join(ACCESS_HIERARCHY)}"
    )
MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_BYTES", 25 * 1024 * 1024, 1, 250 * 1024 * 1024)
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
JOB_STALE_SECONDS = _env_int("JOB_STALE_SECONDS", 300, 30, 86400)
