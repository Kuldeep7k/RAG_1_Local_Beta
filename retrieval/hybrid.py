"""Fuses vector and keyword search results using Reciprocal Rank Fusion
(RRF) -- a simple, effective way to combine differently-scaled rankings
without needing to normalize scores by hand."""
from retrieval.vector_search import vector_search
from retrieval.keyword_search import keyword_search
from retrieval.query import parse_retrieval_constraints
from retrieval.overview_search import document_overview_search
from config import TOP_K_RETRIEVAL
import re


_OVERVIEW_QUERY_RE = re.compile(
    r"\b(?:what\s+is|tell\s+me\s+about|summari[sz]e|overview|"
    r"what\s+does\s+this\s+(?:document|file)|what\s+is\s+this\s+(?:document|file))\b",
    re.IGNORECASE,
)


def reciprocal_rank_fusion(result_lists: list, list_names: list = None, k: int = 60) -> list:
    """score(chunk) = sum(1 / (k + rank)) across every list it appears in.
    Also records, per chunk, which list(s) it came from and at what
    rank/score -- purely for debugging/observability, doesn't affect
    ranking behavior."""
    names = list_names or [f"list_{i}" for i in range(len(result_lists))]
    fused_scores = {}
    chunk_data = {}
    rank_detail = {}

    for name, results in zip(names, result_lists):
        for rank, item in enumerate(results):
            chunk_id = item["chunk_id"]
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
            chunk_data[chunk_id] = item
            rank_detail.setdefault(chunk_id, {})[name] = {
                "rank": rank + 1,
                "score": item.get("score"),
            }

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {**chunk_data[chunk_id], "fused_score": score, "rank_detail": rank_detail[chunk_id]}
        for chunk_id, score in ranked
    ]


def hybrid_search(
    query: str,
    top_k: int = TOP_K_RETRIEVAL,
    access_levels: list = None,
    department: str = None,
) -> list:
    if _OVERVIEW_QUERY_RE.search(query):
        return document_overview_search(top_k, access_levels, department)

    constraints = parse_retrieval_constraints(query)
    vector_results = vector_search(
        query, top_k, access_levels, department, constraints.article_ref
    )
    keyword_results = keyword_search(
        query, top_k, access_levels, department, constraints.article_ref
    )
    return reciprocal_rank_fusion(
        [vector_results, keyword_results],
        list_names=["vector", "keyword"],
    )
