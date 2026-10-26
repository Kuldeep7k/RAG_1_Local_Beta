"""Local sentence-transformer embedding provider."""
from threading import Lock

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_DIM, EMBEDDING_MODEL


_model = None
_model_lock = Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(EMBEDDING_MODEL)
                dimension = _model.get_embedding_dimension()
                if dimension != EMBEDDING_DIM:
                    raise ValueError(
                        f"Embedding model {EMBEDDING_MODEL!r} returned dimension "
                        f"{dimension}; expected {EMBEDDING_DIM}."
                    )
    return _model


def embed_text(text: str) -> list[float]:
    """Encode one text locally and return a normalized vector."""
    return embed_batch([text])[0]


def embed_batch(texts: list[str], on_progress=None) -> list[list[float]]:
    """Encode a batch locally without network calls or provider retries."""
    if not texts:
        return []
    vectors = _get_model().encode(
        texts,
        batch_size=len(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    result = vectors.tolist()
    if len(result) != len(texts):
        raise ValueError(
            f"Embedding model returned {len(result)} vectors for {len(texts)} texts"
        )
    if any(len(vector) != EMBEDDING_DIM for vector in result):
        raise ValueError(
            f"Embedding model returned a vector with unexpected dimension; "
            f"expected {EMBEDDING_DIM}."
        )
    if on_progress is not None:
        for index in range(1, len(result) + 1):
            on_progress(index)
    return result
