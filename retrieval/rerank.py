"""Cross-encoder reranking. Runs on CPU -- no VRAM cost -- and materially
improves precision by scoring (query, chunk) pairs jointly instead of
relying on separately-computed embedding similarity."""
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, TOP_K_RERANK

_models = {}


def _get_model(model_name: str) -> CrossEncoder:
    if model_name not in _models:
        _models[model_name] = CrossEncoder(model_name)
    return _models[model_name]


def rerank(
    query: str,
    candidates: list,
    top_k: int = TOP_K_RERANK,
    model_name: str = RERANKER_MODEL,
) -> list:
    """Re-scores candidates with a cross-encoder and returns the top_k,
    sorted by relevance to the query."""
    if not candidates:
        return []

    model = _get_model(model_name)
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]
